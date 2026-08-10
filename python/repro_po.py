"""Reproduce the paper's Sparse_PO experiment in Python.

Runs SNSGA-II, APF-SNSGA-II (= the paper's APFSNSGAII) and the sparse baselines
on the real portfolio problem, then compares by hypervolume with *joint*
objective normalization (real-world problem -> no analytic PF, HV only) and a
Wilcoxon rank-sum test vs APF-SNSGA-II.

Run:  N_JOBS=-1 python repro_po.py
"""
import os
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from saemo.problems import SparsePO
from saemo import algorithms as alg, baselines as bl, snsga
from saemo.metrics import hypervolume, nondominated

DATA_NO = 1          # 1 = 1000 assets, 2 = 5000 assets
N_RUNS = int(os.environ.get("N_RUNS", "10"))
POP = 100
MAX_FE = 20000
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

ALGOS = {
    "SparseEA": alg.sparseea,
    "MSKEA": bl.mskea,
    "MGCEA": bl.mgcea,
    "BLIGEA": bl.bligea,
    "SNSGA-II": snsga.snsgaii,
    "APF-SNSGA-II": partial(snsga.snsgaii, apf="mod"),
}


def _task(spec):
    name, algo, seed = spec
    prob = SparsePO(dataNo=DATA_NO)
    r = algo(prob, N=POP, max_fe=MAX_FE, seed=seed, hv_ref=None, ref_pf=None)
    return name, seed, r.F


def main():
    os.makedirs("figures", exist_ok=True)
    specs = [(name, fn, s) for name, fn in ALGOS.items() for s in range(N_RUNS)]
    if N_JOBS == 1:
        records = [_task(s) for s in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        workers = os.cpu_count() if N_JOBS in (-1, 0) else N_JOBS
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(_task, specs))

    fronts = {name: [] for name in ALGOS}
    for name, seed, F in records:
        fronts[name].append(F[nondominated(F)])

    # joint normalization across every obtained point
    allpts = np.vstack([F for lst in fronts.values() for F in lst])
    ideal = allpts.min(0)
    nadir = allpts.max(0)
    span = np.where(nadir - ideal == 0, 1.0, nadir - ideal)
    ref = np.array([1.1, 1.1])

    hv = {}
    for name in ALGOS:
        vals = []
        for F in fronts[name]:
            Fn = (F - ideal) / span
            vals.append(hypervolume(Fn, ref))
        hv[name] = np.array(vals)

    # report
    print(f"\n== Sparse_PO (dataNo={DATA_NO}, D={fronts['SNSGA-II'][0].shape[1] if False else '~'}), "
          f"HV (joint-normalized, larger=better), {N_RUNS} runs ==")
    ref_name = "APF-SNSGA-II"
    lines = []
    for name in ALGOS:
        p = ranksums(hv[name], hv[ref_name]).pvalue if name != ref_name else np.nan
        if name == ref_name:
            sym = ""
        elif p >= 0.05:
            sym = "~"
        elif hv[name].mean() > hv[ref_name].mean():
            sym = "+"   # better than APF-SNSGA-II
        else:
            sym = "-"   # worse than APF-SNSGA-II
        line = f"  {name:14s} HV={hv[name].mean():.4e} ({hv[name].std():.1e}) {sym}"
        print(line); lines.append(line)
    with open("figures/po_results.txt", "w") as f:
        f.write("Sparse_PO HV (joint-normalized), vs APF-SNSGA-II\n" + "\n".join(lines))

    # PF plot (median-HV run per algorithm), un-normalized objective space
    fig, ax = plt.subplots(figsize=(6, 4.6))
    markers = ["o", "s", "^", "D", "v", "P"]
    for (name, mk) in zip(ALGOS, markers):
        med = int(np.argsort(hv[name])[len(hv[name]) // 2])
        F = fronts[name][med]
        o = np.argsort(F[:, 0])
        ax.plot(F[o, 0], F[o, 1], marker=mk, ms=3, lw=0.8, alpha=0.8, label=name)
    ax.set_xlabel("$f_1$ = risk"); ax.set_ylabel("$f_2$ = 1 - return")
    ax.set_title(f"Sparse_PO fronts (median run, D={fronts['SNSGA-II'][0].shape[1]})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/po_fronts.png", dpi=150)
    print("\nWrote figures/po_fronts.png and figures/po_results.txt")


if __name__ == "__main__":
    main()
