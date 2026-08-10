"""Ablation study for the SAE/APF framework (Tier A + Tier B).

Isolates each mechanism and rules out trivial explanations (extra offspring,
random directions, random representatives) at an *identical* evaluation budget.

Configurations
  Full          : all mechanisms on
  -D            : no bi-directional sampling
  -R (no-shrink): fixed radius (no coarse-to-fine)        [= C4]
  -I (rand-rep) : random representatives (no IAPD)         [= C3]
  -T            : constant trigger probability
  C1 equal-rand : DST offspring -> equal # of random points (mechanism vs count)
  C1 equal-pert : DST offspring -> equal # of naive perturbations
  C2 rand-dir   : bound-anchored directions -> random directions
  C5 no-zeroprv : drop the zero-preserving frac() term (sparse only)

Run:  N_JOBS=-1 python run_ablation.py
Writes ./figures/ablation_*.txt
"""
import os
from functools import partial
import numpy as np
from scipy.stats import ranksums

from saemo.problems import LSMOP1, LSMOP5, SparsePO
from saemo import algorithms as alg, snsga
from saemo.metrics import hypervolume, nondominated

N_RUNS = int(os.environ.get("N_RUNS", "10"))
POP, MAX_FE = 100, 20000
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

# (label, abl-dict, extra-kwargs) applied on top of the "Full" method
CONFIGS = [
    ("Full",          {},                                  {}),
    ("-D (no-dir)",   {"directional": False},              {}),
    ("-R (no-shrink)", {"shrink": False},                  {}),
    ("-I (rand-rep)", {"iapd": False},                     {}),
    ("-T (const-trig)", {},                                {"trigger": False}),
    ("C1 equal-rand", {"control": "equal_random"},         {}),
    ("C1 equal-pert", {"control": "equal_perturb"},        {}),
    ("C2 rand-dir",   {"direction": "random"},             {}),
]
SPARSE_ONLY = ("C5 no-zeroprv", {}, {"amplitude": "fixed"})   # PO only


def _task(spec):
    kind, label, abl, extra, seed = spec
    if kind in ("lsmop1", "lsmop5"):
        p = (LSMOP1 if kind == "lsmop1" else LSMOP5)(n_var=1000, n_obj=2)
        ref = p.pareto_front(300)
        kw = dict(amplitude="fixed", trigger=True)
        kw.update(extra)
        r = alg.apf_nsga2(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=ref,
                          abl=(abl or None), name=label, **kw)
        F = r.F[nondominated(r.F)]
        return kind, label, seed, r.history["igd"][-1], F
    # Sparse_PO (real problem -> HV computed later)
    p = SparsePO(dataNo=1)
    apf_mode = extra.get("amplitude", "mod")       # C5 -> 'fixed'
    trig = extra.get("trigger", True)
    r = snsga.snsgaii(p, N=POP, max_fe=MAX_FE, seed=seed, ref_pf=None,
                      apf=apf_mode, abl=(abl or None), trigger=trig, name=label)
    F = r.F[nondominated(r.F)]
    return kind, label, seed, np.nan, F


def _run(kind, configs):
    specs = [(kind, lab, abl, extra, s)
             for (lab, abl, extra) in configs for s in range(N_RUNS)]
    if N_JOBS == 1:
        recs = [_task(s) for s in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        w = os.cpu_count() if N_JOBS in (-1, 0) else N_JOBS
        with ProcessPoolExecutor(max_workers=w) as ex:
            recs = list(ex.map(_task, specs))
    igd = {lab: [] for (lab, _, _) in configs}
    fronts = {lab: [] for (lab, _, _) in configs}
    for _, lab, _, ig, F in recs:
        igd[lab].append(ig); fronts[lab].append(F)
    return {lab: np.array(v) for lab, v in igd.items()}, fronts


def _report(title, igd, metric="IGD", better="min"):
    ref = igd["Full"]
    lines = [f"\n== {title} ({metric}, {N_RUNS} runs) : vs Full ==",
             f"{'config':16s} {'mean':>11s} {'std':>10s}  sig"]
    for lab, v in igd.items():
        if lab == "Full":
            sym = " (ref)"
        else:
            p = ranksums(v, ref).pvalue
            worse = (v.mean() > ref.mean()) if better == "min" else (v.mean() < ref.mean())
            sym = "~" if p >= 0.05 else ("worse-than-Full" if worse else "better-than-Full")
        lines.append(f"{lab:16s} {v.mean():11.4e} {v.std():10.1e}  {sym}")
    txt = "\n".join(lines)
    print(txt)
    return txt


def main():
    os.makedirs("figures", exist_ok=True)
    out = []
    for kind, title in [("lsmop1", "LSMOP1 D=1000 (SAE-NSGA-II)"),
                        ("lsmop5", "LSMOP5 D=1000 (SAE-NSGA-II)")]:
        igd, _ = _run(kind, CONFIGS)
        out.append(_report(title, igd))
    # PO: IGD not defined (real problem) -> use HV via joint normalization
    igd_po, fronts_po = _run("po", CONFIGS + [SPARSE_ONLY])
    allpts = np.vstack([F for lst in fronts_po.values() for F in lst])
    ide, nad = allpts.min(0), allpts.max(0)
    span = np.where(nad - ide == 0, 1.0, nad - ide)
    ref_pt = np.array([1.1, 1.1])
    hv = {lab: np.array([hypervolume((F - ide) / span, ref_pt) for F in lst])
          for lab, lst in fronts_po.items()}
    out.append(_report("Sparse_PO D=1000 (APF-SNSGA-II)", hv, metric="HV", better="max"))

    with open("figures/ablation_results.txt", "w") as f:
        f.write("\n".join(out))
    print("\nWrote figures/ablation_results.txt")


if __name__ == "__main__":
    main()
