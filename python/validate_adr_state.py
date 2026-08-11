"""Validate the state-adaptive ADR (parameter-free) against the fixed-schedule
ADR and SparseEA on the sparse suite: does removing the hand-set coarse->fine
schedule (amplitude driven by convergence-stall & diversity state) match or beat
the tuned schedule?

Run:  N_JOBS=-1 python validate_adr_state.py
"""
import os
from functools import partial
import numpy as np
from scipy.stats import ranksums

from saemo.problems import SMOP1, SMOP2, SMOP3, SMOP4, SMOP5, SparsePO
from saemo import algorithms as alg, adr
from saemo.metrics import hypervolume, nondominated

N_RUNS = int(os.environ.get("N_RUNS", "10"))
POP, MAX_FE = 100, 20000
N_JOBS = int(os.environ.get("N_JOBS", "-1"))
SMOPS = {"SMOP1": SMOP1, "SMOP2": SMOP2, "SMOP3": SMOP3, "SMOP4": SMOP4, "SMOP5": SMOP5}


def _task(spec):
    pname, alg_name, seed = spec
    if pname == "SparsePO":
        p = SparsePO(dataNo=1); ref = None
    else:
        p = SMOPS[pname](n_var=100); ref = p.pareto_front(300)
    fn = {"ADR-fixed": adr.adr_nsga2, "ADR-State": adr.adr_state_nsga2,
          "SparseEA": alg.sparseea}[alg_name]
    r = fn(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=ref)
    F = r.F[nondominated(r.F)]
    return pname, alg_name, r.history["igd"][-1], F


def main():
    os.makedirs("figures", exist_ok=True)
    algos = ["ADR-fixed", "ADR-State", "SparseEA"]
    probs = list(SMOPS) + ["SparsePO"]
    specs = [(pn, an, s) for pn in probs for an in algos for s in range(N_RUNS)]
    if N_JOBS == 1:
        recs = [_task(s) for s in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        w = os.cpu_count() if N_JOBS in (-1, 0) else N_JOBS
        with ProcessPoolExecutor(max_workers=w) as ex:
            recs = list(ex.map(_task, specs))

    igd, fronts = {}, {}
    for pn, an, ig, F in recs:
        igd.setdefault((pn, an), []).append(ig)
        fronts.setdefault((pn, an), []).append(F)

    lines = ["State-adaptive ADR validation (SMOP: IGD lower=better; PO: HV higher=better)",
             "  vs fixed-schedule ADR (Wilcoxon)"]
    lines.append("\n== SMOP1-5 (IGD) ==")
    for pn in SMOPS:
        f = np.array(igd[(pn, "ADR-fixed")]); s = np.array(igd[(pn, "ADR-State")])
        sp = np.array(igd[(pn, "SparseEA")])
        p = ranksums(s, f).pvalue
        tag = "~" if p >= 0.05 else ("State better" if s.mean() < f.mean() else "State worse")
        lines.append(f"  {pn:8s} ADR-fixed={f.mean():.3e}  ADR-State={s.mean():.3e}({s.std():.1e}) [{tag}]"
                     f"  SparseEA={sp.mean():.3e}")

    # Sparse_PO: HV with joint normalization
    allpts = np.vstack([F for an in algos for F in fronts[("SparsePO", an)]])
    ide, nad = allpts.min(0), allpts.max(0)
    span = np.where(nad - ide == 0, 1.0, nad - ide)
    lines.append("\n== Sparse_PO (HV, joint-normalized) ==")
    hv = {an: np.array([hypervolume((F - ide) / span, np.array([1.1, 1.1]))
                        for F in fronts[("SparsePO", an)]]) for an in algos}
    p = ranksums(hv["ADR-State"], hv["ADR-fixed"]).pvalue
    tag = "~" if p >= 0.05 else ("State better" if hv["ADR-State"].mean() > hv["ADR-fixed"].mean() else "State worse")
    for an in algos:
        mark = f" [{tag}]" if an == "ADR-State" else ""
        lines.append(f"  {an:10s} HV={hv[an].mean():.4e}({hv[an].std():.1e}){mark}")

    txt = "\n".join(lines)
    print(txt)
    with open("figures/adr_state_validation.txt", "w") as f:
        f.write(txt)
    print("\nWrote figures/adr_state_validation.txt")


if __name__ == "__main__":
    main()
