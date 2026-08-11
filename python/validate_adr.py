"""Robustness validation of ADR-NSGA-II.

Multi-seed comparison vs NSGA-II (and SparseEA on the sparse suite) across
LSMOP1-9 and SMOP1-5, reporting IGD mean(std), Wilcoxon significance, the
coefficient of variation (stability across seeds), and the mean self-adapted
sparsification share p_sparsify. Also saves Pareto-front plots for a set of
representative problems to check that ADR is not winning IGD at the cost of
diversity.

Run:  N_JOBS=-1 python validate_adr.py
Writes ./figures/adr_validation.txt and ./figures/adr_pf_*.png
"""
import os
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from saemo import problems as prob
from saemo import algorithms as alg, adr
from saemo.metrics import nondominated

N_RUNS = int(os.environ.get("N_RUNS", "10"))
POP, MAX_FE, DIM_L, DIM_S = 100, 20000, 1000, 100
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

LSMOPS = [f"LSMOP{i}" for i in range(1, 10)]
SMOPS = [f"SMOP{i}" for i in range(1, 6)]


def _task(spec):
    kind, pname, alg_name, seed = spec
    D = DIM_L if kind == "lsmop" else DIM_S
    p = prob.PROBLEMS[pname](n_var=D, n_obj=2)
    ref = p.pareto_front(300)
    if alg_name == "NSGA-II":
        r = alg.nsga2(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=ref)
        extra = np.nan
    elif alg_name == "SparseEA":
        r = alg.sparseea(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=ref)
        extra = np.nan
    else:  # ADR
        r = adr.adr_nsga2(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=ref, track_s=True)
        extra = r.get("p_sparsify", np.nan)
    F = r.F[nondominated(r.F)]
    return pname, alg_name, seed, r.history["igd"][-1], extra, F


def main():
    os.makedirs("figures", exist_ok=True)
    specs = []
    for pn in LSMOPS:
        for an in ("NSGA-II", "ADR"):
            specs += [("lsmop", pn, an, s) for s in range(N_RUNS)]
    for pn in SMOPS:
        for an in ("NSGA-II", "SparseEA", "ADR"):
            specs += [("smop", pn, an, s) for s in range(N_RUNS)]

    if N_JOBS == 1:
        recs = [_task(s) for s in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        w = os.cpu_count() if N_JOBS in (-1, 0) else N_JOBS
        with ProcessPoolExecutor(max_workers=w) as ex:
            recs = list(ex.map(_task, specs))

    igd = {}
    psp = {}
    fronts = {}
    for pname, an, seed, ig, extra, F in recs:
        igd.setdefault((pname, an), []).append(ig)
        if an == "ADR":
            psp.setdefault(pname, []).append(extra)
            fronts.setdefault(pname, []).append((ig, F))

    lines = ["ADR validation  (IGD mean(std), Wilcoxon ADR vs NSGA-II, CV=std/mean, p_sp=sparsify share)"]
    for group, names, others in [("LSMOP", LSMOPS, ["NSGA-II"]),
                                 ("SMOP", SMOPS, ["NSGA-II", "SparseEA"])]:
        lines.append(f"\n== {group} ==")
        for pn in names:
            adrv = np.array(igd[(pn, "ADR")])
            base = np.array(igd[(pn, "NSGA-II")])
            p = ranksums(adrv, base).pvalue
            win = "ADR<NSGA-II(better)" if adrv.mean() < base.mean() else "ADR worse"
            sig = "~" if p >= 0.05 else win
            cv = adrv.std() / adrv.mean() if adrv.mean() else np.nan
            extra = ""
            if "SparseEA" in others:
                sp = np.array(igd[(pn, "SparseEA")])
                extra = f"  SparseEA={sp.mean():.3e}"
            lines.append(f"  {pn:8s} NSGA-II={base.mean():.3e}  ADR={adrv.mean():.3e}({adrv.std():.1e})"
                         f"  CV={cv:.2f}  p_sp={np.mean(psp[pn]):.2f}  [{sig}]{extra}")
    txt = "\n".join(lines)
    print(txt)
    with open("figures/adr_validation.txt", "w") as f:
        f.write(txt)

    # PF plots for representative problems (median ADR run vs true PF)
    show = ["LSMOP1", "LSMOP5", "LSMOP9", "SMOP1", "SMOP4"]
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.2))
    for ax, pn in zip(axes, show):
        D = DIM_L if pn.startswith("LSMOP") else DIM_S
        R = prob.PROBLEMS[pn](n_var=D, n_obj=2).pareto_front(300)
        runs = sorted(fronts[pn], key=lambda z: z[0])
        F = runs[len(runs) // 2][1]                       # median-IGD run
        ax.plot(R[:, 0], R[:, 1], "k-", lw=1, label="true PF")
        ax.scatter(F[:, 0], F[:, 1], s=10, c="tab:red", alpha=0.7, label="ADR")
        ax.set_title(pn); ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/adr_pf_grid.png", dpi=150)
    print("\nWrote figures/adr_validation.txt and figures/adr_pf_grid.png")


if __name__ == "__main__":
    main()
