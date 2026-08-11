"""ADR -- Adaptive Density-aware Refinement.

A minimal refinement layer (no IAPD, no fixed stage division, no manual mode
switch) added to a base MOEA. Each generation it injects two kinds of extra
offspring built from random population members:

  * exploration  : a long-range move (drives progress on dense large-scale
                   problems such as LSMOP);
  * sparsification: soft-threshold small components to exactly zero (CREATE
                   sparsity -- the proximal/ISTA step) then refine with a
                   relative, zero-preserving amplitude (supp(x*) subset of
                   supp(thresholded x)).

The injection budget is split between the two by their recent *survival rate*
in environmental selection -- so the operator self-adapts to the problem:
sparsification survives on sparse problems, exploration survives on dense ones.
No problem-specific tuning.
"""
from __future__ import annotations

import numpy as np

from . import core
from .algorithms import Result, _tracker
from .metrics import nondominated


def population_sparsity(X, eps=1e-6):
    return float(np.mean(np.abs(X) < eps))


def _explore(reps, problem, rng, amp_frac):
    """Long-range random-direction move from each representative."""
    Nw, D = reps.shape
    dirs = rng.standard_normal((Nw, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    step = rng.uniform(0, amp_frac * np.linalg.norm(problem.xu - problem.xl), (Nw, 1))
    return np.clip(reps + step * dirs, problem.xl, problem.xu)


def _sparsify(reps, problem, rng, t, sigma0, fmax=0.95):
    """Soft-threshold small components to 0 (create sparsity), then apply a
    relative, zero-preserving refinement (preserve the new support)."""
    Nw, D = reps.shape
    sp = reps.copy()
    for i in range(Nw):
        f = rng.uniform(0, fmax)                     # try a random sparsity level
        k = int(f * D)
        if k > 0:
            idx = np.argsort(np.abs(sp[i]))[:k]      # smallest-magnitude entries
            sp[i, idx] = 0.0                         # -> exactly zero
    amp = sigma0 * (1 - t)                            # coarse -> fine
    rho = np.abs(sp)                                  # relative -> zeros stay zero
    sp = sp + rng.uniform(-1, 1, sp.shape) * amp * rho
    return np.clip(sp, problem.xl, problem.xu)


def adr_nsga2(problem, N=100, max_fe=20000, seed=0, sigma0=3.0, frac_ref=0.1,
              hv_ref=None, ref_pf=None, record_every=1, name="ADR-NSGA-II",
              track_s=False):
    """NSGA-II + adaptive density-aware refinement (success-weighted dual mode)."""
    rng = np.random.default_rng(seed)
    D = problem.n_var
    X = problem.xl + rng.random((N, D)) * (problem.xu - problem.xl)
    F = problem.evaluate(X)
    fe = N
    front = core.fast_nondominated_sort(F)
    cd = core.crowding_distance(F, front)
    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)

    Nw = max(1, int(np.ceil(N * frac_ref)))
    succ_sp, succ_ex = 1.0, 1.0                      # Laplace-smoothed survival counts
    tot_sp, tot_ex = 2.0, 2.0
    s_log = []

    while fe < max_fe:
        pool = core.tournament_selection(2, N, front, -cd, rng=rng)
        base_off = core.operator_ga(X[pool], problem.xl, problem.xu, rng=rng)

        # split the injection budget by recent survival rate
        p_sp = (succ_sp / tot_sp) / (succ_sp / tot_sp + succ_ex / tot_ex)
        n_sp = int(round(p_sp * Nw)); n_ex = Nw - n_sp
        reps_sp = X[rng.integers(0, N, max(n_sp, 1))]
        reps_ex = X[rng.integers(0, N, max(n_ex, 1))]
        sp = _sparsify(reps_sp, problem, rng, fe / max_fe, sigma0)[:n_sp] if n_sp else np.zeros((0, D))
        ex = _explore(reps_ex, problem, rng, 1.0)[:n_ex] if n_ex else np.zeros((0, D))

        off = np.vstack([base_off, sp, ex])
        offF = problem.evaluate(off)
        fe += len(off)
        Xall = np.vstack([X, off]); Fall = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(Fall, N)

        # survival bookkeeping (offspring live at indices >= N+len(base_off))
        base0 = N + len(base_off)
        sp_ids = set(range(base0, base0 + len(sp)))
        ex_ids = set(range(base0 + len(sp), base0 + len(sp) + len(ex)))
        kept = set(keep.tolist())
        if len(sp):
            succ_sp += len(sp_ids & kept); tot_sp += len(sp)
        if len(ex):
            succ_ex += len(ex_ids & kept); tot_ex += len(ex)

        X, F = Xall[keep], Fall[keep]
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        record(fe, F)
        if track_s:
            s_log.append(population_sparsity(X))

    record(fe, F, force=True)
    nd = nondominated(F)
    r = Result(X=X[nd], F=F[nd], history=hist, name=name)
    r["p_sparsify"] = (succ_sp / tot_sp) / (succ_sp / tot_sp + succ_ex / tot_ex)
    if track_s:
        r["s_mean"] = float(np.mean(s_log)) if s_log else np.nan
    return r
