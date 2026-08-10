"""Experiment runner, statistics, and plotting."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from .metrics import nondominated


def run_comparison(problem_factory, algos, dims, N=100, max_fe=20000,
                   n_runs=10, hv_ref=None, base_seed=0):
    """Run every algorithm on every dimension for n_runs seeds.

    Parameters
    ----------
    problem_factory : callable(D) -> Problem
    algos           : dict name -> callable(problem, N, max_fe, seed, hv_ref, ref_pf)
    dims            : list of decision-variable counts
    Returns a nested results dict.
    """
    out = {}
    for D in dims:
        prob = problem_factory(D)
        ref_pf = prob.pareto_front(300)
        out[D] = {}
        for name, fn in algos.items():
            igds, hvs, runs = [], [], []
            for s in range(n_runs):
                r = fn(prob, N=N, max_fe=max_fe, seed=base_seed + s,
                       hv_ref=hv_ref, ref_pf=ref_pf)
                igds.append(r.history["igd"][-1])
                hvs.append(r.history["hv"][-1])
                runs.append(r)
            out[D][name] = {
                "igd": np.array(igds), "hv": np.array(hvs),
                "runs": runs, "ref_pf": ref_pf,
            }
            print(f"D={D:5d} {name:16s} IGD={np.mean(igds):.4e}({np.std(igds):.1e}) "
                  f"HV={np.mean(hvs):.4e}({np.std(hvs):.1e})")
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
