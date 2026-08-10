"""End-to-end demo: compare NSGA-II, SparseEA and APF-NSGA-II on a sparse
bi-objective problem across dimensions, run statistics, and save figures.

Run:  python demo.py
Figures are written to ./figures/.
"""
import os
import numpy as np

from saemo.problems import SparseZDT1
from saemo import algorithms as alg
from saemo import experiment as exp

os.makedirs("figures", exist_ok=True)

HV_REF = np.array([1.1, 1.1])
DIMS = [100, 500, 1000, 2000]
N_RUNS = 10
POP = 100
MAX_FE = 20000

algos = {
    "NSGA-II": alg.nsga2,
    "SparseEA": alg.sparseea,
    "APF-NSGA-II": alg.apf_nsga2,
}

if __name__ == "__main__":
    results = exp.run_comparison(
        SparseZDT1, algos, DIMS, N=POP, max_fe=MAX_FE, n_runs=N_RUNS, hv_ref=HV_REF)

    print("\n== Wilcoxon rank-sum vs APF-NSGA-II (HV) ==")
    for row in exp.significance_table(results, metric="hv"):
        print(f"D={row['D']:5d} {row['algo']:12s} "
              f"HV={row['mean']:.4e} p={row['p']:.2e} {row['vs_ref']}")

    exp.plot_pareto(results, 1000, "figures/pareto_D1000.png")
    exp.plot_convergence(results, 1000, "figures/igd_conv_D1000.png", metric="igd")
    exp.plot_scalability(results, "figures/hv_vs_D.png", metric="hv")
    exp.plot_scalability(results, "figures/igd_vs_D.png", metric="igd")
    print("\nFigures saved to ./figures/")
