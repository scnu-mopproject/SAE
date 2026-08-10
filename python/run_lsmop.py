"""Reproduce the original paper's LSMOP experiments (framework vs base optimizer).

Compares plain NSGA-II against SAE-NSGA-II (the SAE/APF framework embedding
NSGA-II, using the additive refinement operator of the LSMOP paper, Eq. 12) on
LSMOP1-9. Primary metric: IGD (lower is better).

Run:  N_JOBS=-1 python run_lsmop.py
Writes ./figures/lsmop_*.png and ./figures/lsmop_results.txt
"""
import os
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saemo import problems as prob
from saemo import algorithms as alg
from saemo import experiment as exp

os.makedirs("figures", exist_ok=True)

M = 2
DIM = int(os.environ.get("DIM", "1000"))
N_RUNS = int(os.environ.get("N_RUNS", "10"))
POP = 100
MAX_FE = 20000
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

ALGOS = {
    "NSGA-II": alg.nsga2,
    "SAE-NSGA-II": partial(alg.apf_nsga2, amplitude="fixed"),
}
LSMOPS = [f"LSMOP{i}" for i in range(1, 10)]


def _factory(name):
    cls = prob.PROBLEMS[name]
    return partial(cls, n_obj=M)          # picklable; run_comparison calls factory(D)


def main():
    per = {}
    lines = ["LSMOP (M=%d, D=%d) IGD mean(std), Wilcoxon: NSGA-II vs SAE-NSGA-II" % (M, DIM)]
    for name in LSMOPS:
        res = exp.run_comparison(_factory(name), ALGOS, [DIM], N=POP, max_fe=MAX_FE,
                                 n_runs=N_RUNS, hv_ref=None, n_jobs=N_JOBS)
        per[name] = res[DIM]
        row = exp.significance_table(res, metric="igd", reference="SAE-NSGA-II")
        base = per[name]["NSGA-II"]["igd"]; sae = per[name]["SAE-NSGA-II"]["igd"]
        sym = row[0]["vs_ref"] if row else "?"
        lines.append(f"  {name}: NSGA-II={base.mean():.3e}  SAE-NSGA-II={sae.mean():.3e}  ({sym})")
    print("\n".join(lines))
    with open("figures/lsmop_results.txt", "w") as f:
        f.write("\n".join(lines))

    # bar chart of IGD across LSMOP1-9
    x = np.arange(len(LSMOPS)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, a in enumerate(ALGOS):
        means = [per[p][a]["igd"].mean() for p in LSMOPS]
        errs = [per[p][a]["igd"].std() for p in LSMOPS]
        ax.bar(x + i * w, means, w, yerr=errs, capsize=2, label=a)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(LSMOPS, rotation=30)
    ax.set_ylabel("IGD"); ax.set_yscale("log")
    ax.set_title(f"IGD on LSMOP1-9 (M={M}, D={DIM})"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/lsmop_igd_bar.png", dpi=150)
    print("\nWrote figures/lsmop_igd_bar.png and figures/lsmop_results.txt")


if __name__ == "__main__":
    main()
