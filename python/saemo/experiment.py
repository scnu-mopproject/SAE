"""Experiment runner, statistics, and plotting."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from .metrics import nondominated


def _run_task(spec):
    """Top-level worker (picklable) for parallel execution."""
    problem_factory, D, name, algo, seed, N, max_fe, hv_ref = spec
    prob = problem_factory(D)
    ref_pf = prob.pareto_front(300)
    r = algo(prob, N=N, max_fe=max_fe, seed=seed, hv_ref=hv_ref, ref_pf=ref_pf)
    return D, name, seed, r, ref_pf


def run_comparison(problem_factory, algos, dims, N=100, max_fe=20000,
                   n_runs=10, hv_ref=None, base_seed=0, n_jobs=1):
    """Run every algorithm on every dimension for n_runs seeds.

    Parameters
    ----------
    problem_factory : callable(D) -> Problem. Must be picklable when n_jobs>1
                      (pass a Problem subclass or functools.partial, NOT a lambda).
    algos           : dict name -> callable(problem, N, max_fe, seed, hv_ref, ref_pf).
                      Values must be picklable when n_jobs>1 (module-level funcs or
                      functools.partial, NOT lambdas).
    dims            : list of decision-variable counts
    n_jobs          : processes to use (>1 runs seeds/problems/algos in parallel;
                      -1 uses all CPU cores). Near-linear speedup on many-core hosts.
    Returns a nested results dict.
    """
    specs = [(problem_factory, D, name, fn, base_seed + s, N, max_fe, hv_ref)
             for D in dims for name, fn in algos.items() for s in range(n_runs)]

    if n_jobs == 1:
        records = [_run_task(s) for s in specs]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor
        workers = os.cpu_count() if n_jobs in (-1, 0) else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(_run_task, specs))

    out = {D: {} for D in dims}
    for D, name, seed, r, ref_pf in records:
        d = out[D].setdefault(name, {"igd": [], "hv": [], "runs": [], "ref_pf": ref_pf})
        d["igd"].append(r.history["igd"][-1])
        d["hv"].append(r.history["hv"][-1])
        d["runs"].append(r)
    for D in dims:
        for name in algos:
            d = out[D][name]
            d["igd"] = np.array(d["igd"]); d["hv"] = np.array(d["hv"])
            print(f"D={D:5d} {name:16s} IGD={d['igd'].mean():.4e}({d['igd'].std():.1e}) "
                  f"HV={d['hv'].mean():.4e}({d['hv'].std():.1e})")
    return out


def significance_table(results, metric="igd", reference="APF-NSGA-II"):
    """Wilcoxon rank-sum of each algorithm vs `reference` per dimension.
    Returns rows of dicts; '+/-/~' is from the reference's viewpoint."""
    rows = []
    better = "hv"          # for HV larger is better; for IGD smaller is better
    for D, byalg in results.items():
        ref_vals = byalg[reference][metric]
        for name, d in byalg.items():
            if name == reference:
                continue
            vals = d[metric]
            p = ranksums(vals, ref_vals).pvalue
            if p >= 0.05:
                sym = "~"
            elif (np.mean(vals) > np.mean(ref_vals)) == (metric == better):
                sym = "+"          # competitor better than reference
            else:
                sym = "-"          # competitor worse than reference
            rows.append({"D": D, "algo": name, "metric": metric,
                         "mean": np.mean(vals), "std": np.std(vals),
                         "p": p, "vs_ref": sym})
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_pareto(results, D, outfile, title=None):
    byalg = next(iter(results.values())) if D not in results else results[D]
    ref_pf = byalg[next(iter(byalg))]["ref_pf"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(ref_pf[:, 0], ref_pf[:, 1], "k-", lw=1.2, label="True PF", zorder=1)
    markers = ["o", "s", "^", "D", "v", "P"]
    for (name, d), mk in zip(byalg.items(), markers):
        # median run by final IGD
        med = int(np.argsort(d["igd"])[len(d["igd"]) // 2])
        F = d["runs"][med].F
        ax.scatter(F[:, 0], F[:, 1], s=16, marker=mk, alpha=0.7, label=name, zorder=2)
    ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")
    ax.set_title(title or f"Pareto front (D={D})")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_convergence(results, D, outfile, metric="igd", title=None):
    byalg = results[D]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, d in byalg.items():
        # align histories on the shortest length
        hs = [r.history for r in d["runs"]]
        L = min(len(h[metric]) for h in hs)
        fe = np.array(hs[0]["fe"][:L])
        vals = np.array([h[metric][:L] for h in hs])
        m = np.nanmean(vals, axis=0)
        ax.plot(fe, m, lw=1.6, label=name)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(metric.upper())
    if metric == "igd":
        ax.set_yscale("log")
    ax.set_title(title or f"{metric.upper()} convergence (D={D})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_scalability(results, outfile, metric="hv", title=None):
    """Metric vs dimension: the key 'does it stay flat as D grows' figure."""
    dims = sorted(results.keys())
    names = list(results[dims[0]].keys())
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    markers = ["o", "s", "^", "D", "v", "P"]
    for name, mk in zip(names, markers):
        mean = [np.mean(results[D][name][metric]) for D in dims]
        std = [np.std(results[D][name][metric]) for D in dims]
        ax.errorbar(dims, mean, yerr=std, marker=mk, capsize=3, lw=1.6, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Decision variables $D$")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} vs. dimension")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile
