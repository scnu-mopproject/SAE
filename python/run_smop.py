"""SMOP experiment: compare a plain base (NSGA-II), a sparse base (SparseEA),
and the APF refinement embedded in the sparse base (the correct analogue of the
paper's APFSNSGAII), on the SMOP suite.

Run:  python run_smop.py
Writes figures to ./figures/ and a results table to ./figures/results.txt
"""
import os
import numpy as np

from saemo import problems as prob
from saemo import algorithms as alg
from saemo import baselines as bl
from saemo import experiment as exp

os.makedirs("figures", exist_ok=True)

HV_REF = np.array([1.1, 1.1])
N_RUNS = 10
POP = 100
MAX_FE = 20000

ALGOS = {
    "SparseEA": alg.sparseea,
    "MSKEA": bl.mskea,
    "MGCEA": bl.mgcea,
    "BLIGEA": bl.bligea,
    "APF-SparseEA": lambda p, **k: alg.apf_sparseea(p, amplitude="mod", **k),
}

SMOPS = {"SMOP1": prob.SMOP1, "SMOP2": prob.SMOP2, "SMOP3": prob.SMOP3,
         "SMOP4": prob.SMOP4, "SMOP5": prob.SMOP5}


def main():
    lines = []

    # ---- (A) all SMOP1-5 at D=100 ----
    per_problem = {}
    for pname, pcls in SMOPS.items():
        print(f"\n=== {pname} (D=100) ===")
        res = exp.run_comparison(lambda D: pcls(n_var=D), ALGOS, [100],
                                 N=POP, max_fe=MAX_FE, n_runs=N_RUNS, hv_ref=HV_REF)
        per_problem[pname] = res[100]
        lines.append(f"\n== {pname} (D=100) : IGD mean(std), Wilcoxon vs APF-SparseEA ==")
        for row in exp.significance_table(res, metric="igd", reference="APF-SparseEA"):
            lines.append(f"  {row['algo']:20s} {row['mean']:.4e}({row['std']:.1e}) "
                         f"p={row['p']:.1e} {row['vs_ref']}")

    # bar chart of IGD across SMOP1-5
    _bar_igd(per_problem, "figures/igd_bar_SMOP1-5.png")

    # ---- (B) SMOP1 scalability across dimensions ----
    print("\n=== SMOP1 scalability ===")
    scal = exp.run_comparison(lambda D: prob.SMOP1(n_var=D), ALGOS,
                              [100, 500, 1000], N=POP, max_fe=MAX_FE,
                              n_runs=max(4, N_RUNS // 2), hv_ref=HV_REF)
    exp.plot_pareto(scal, 500, "figures/smop1_pareto_D500.png",
                    title="SMOP1 Pareto front (D=500)")
    exp.plot_convergence(scal, 500, "figures/smop1_igd_conv_D500.png", metric="igd",
                         title="SMOP1 IGD convergence (D=500)")
    exp.plot_scalability(scal, "figures/smop1_igd_vs_D.png", metric="igd",
                         title="SMOP1 IGD vs. dimension")
    exp.plot_scalability(scal, "figures/smop1_hv_vs_D.png", metric="hv",
                         title="SMOP1 HV vs. dimension")

    with open("figures/results.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nFigures + results.txt written to ./figures/")


def _bar_igd(per_problem, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    probs = list(per_problem.keys())
    algos = list(next(iter(per_problem.values())).keys())
    x = np.arange(len(probs)); w = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, a in enumerate(algos):
        means = [np.mean(per_problem[p][a]["igd"]) for p in probs]
        errs = [np.std(per_problem[p][a]["igd"]) for p in probs]
        ax.bar(x + i * w, means, w, yerr=errs, capsize=2, label=a)
    ax.set_xticks(x + w * (len(algos) - 1) / 2); ax.set_xticklabels(probs)
    ax.set_ylabel("IGD"); ax.set_yscale("log")
    ax.set_title("IGD on SMOP1-5 (D=100)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
