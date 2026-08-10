"""Sparse NSGA-II (SNSGAII) and APF-SNSGAII.

Faithful port of the PlatEMO S-NSGA-II sources (Kropp, Nejadhashemi & Deb,
IEEE TEVC 2024): real-encoded sparse algorithm with striped sparse
initialization (vssps), sparsity-aware SBX (ssbx), and value+sparsity
mutation (spm). ``apf=`` wraps it with the DST refinement layer, reproducing
the paper's APFSNSGAII.
"""
from __future__ import annotations

import numpy as np

from . import core
from .algorithms import Result, _tracker
from .baselines import tournament, _apf_setup, _apf_inject
from .metrics import nondominated


# --------------------------------------------------------------------------- #
# Sparse operators
# --------------------------------------------------------------------------- #
def _poly_mutate_core(g, lb, ub, eta, rng):
    span = ub - lb
    d1 = (g - lb) / span
    d2 = (ub - g) / span
    ex = 1.0 / (eta + 1)
    r = rng.random(g.shape)
    dq = np.empty_like(g)
    left = r < 0.5
    val = 2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)
    dq_left = val ** ex - 1
    val = 2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** (eta + 1)
    dq_right = 1 - val ** ex
    dq = np.where(left, dq_left, dq_right)
    return np.clip(g + dq * span, lb, ub)


def _sm2target(Pop, xl, xu, new_spars, rng):
    """Adjust each row's number of zeros to hit its target sparsity."""
    Pop = Pop.copy()
    N, D = Pop.shape
    spars = (Pop == 0).sum(1) / D
    xl = np.broadcast_to(xl, (D,)); xu = np.broadcast_to(xu, (D,))
    for i in range(N):
        if new_spars[i] == spars[i]:
            continue
        nz2add = int(round(D * (spars[i] - new_spars[i])))
        if nz2add > 0:                                   # need more non-zeros
            zloc = np.where(Pop[i] == 0)[0]
            k = min(nz2add, len(zloc))
            if k > 0:
                flip = rng.choice(zloc, k, replace=False)
                Pop[i, flip] = xl[flip] + rng.random(k) * (xu[flip] - xl[flip])
        elif nz2add < 0:                                 # need more zeros
            nzloc = np.where(Pop[i] != 0)[0]
            k = min(-nz2add, len(nzloc))
            if k > 0:
                flip = rng.choice(nzloc, k, replace=False)
                Pop[i, flip] = 0.0
    return Pop


def ssbx(parent, xl, xu, rng, proC=1.0, disC=20.0):
    """Sparsity-aware SBX: SBX on matched (both-zero/both-nonzero) positions;
    value swap on mismatched positions."""
    P = np.asarray(parent, float)
    half = len(P) // 2
    P1, P2 = P[:half], P[half:2 * half]
    z1, z2 = P1 == 0, P2 == 0
    matching = (z1 & z2) | (~z1 & ~z2)
    off1, off2 = P1.copy(), P2.copy()
    if matching.any():
        a, b = P1[matching], P2[matching]
        mu = rng.random(a.shape)
        beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)),
                        (2 - 2 * mu) ** (-1 / (disC + 1)))
        beta *= (-1) ** rng.integers(0, 2, a.shape)
        beta[rng.random(a.shape) < 0.5] = 1
        off1[matching] = (a + b) / 2 + beta * (a - b) / 2
        off2[matching] = (a + b) / 2 - beta * (a - b) / 2
    nm = ~matching
    if nm.any():
        a, b = P1[nm].copy(), P2[nm].copy()
        K = len(a)
        z2nz = rng.random()
        nflip = int(round(K * z2nz))                     # positions kept (not swapped)
        swapmask = np.ones(K, bool)
        if nflip > 0:
            swapmask[rng.choice(K, nflip, replace=False)] = False
        ta = a.copy()
        a[swapmask], b[swapmask] = b[swapmask], ta[swapmask]
        off1[nm], off2[nm] = a, b
    out = np.vstack([off1, off2])
    return np.clip(out, xl, xu)


def spm(Pop, xl, xu, rng, probMut=1.0, distrMut=20.0, probSMut=1.0, distrSMut=20.0):
    """Sparse polynomial mutation: value mutation on non-zeros + sparsity mutation."""
    Pop = np.asarray(Pop, float).copy()
    N, D = Pop.shape
    xlv = np.broadcast_to(xl, (D,)); xuv = np.broadcast_to(xu, (D,))
    nz = Pop != 0
    site = nz & (rng.random((N, D)) < probMut / D)
    if site.any():
        cols = np.where(site)[1]
        Pop[site] = _poly_mutate_core(Pop[site], xlv[cols], xuv[cols], distrMut, rng)
    # sparsity mutation
    mut = rng.random(N) < probSMut / D
    if mut.any():
        spars = (Pop == 0).sum(1) / D
        new = spars.copy()
        idx = np.where(mut)[0]
        new[idx] = _poly_mutate_core(spars[idx], np.zeros(len(idx)),
                                     np.ones(len(idx)), distrSMut, rng)
        new = np.clip(new, 0, 1)
        Pop = _sm2target(Pop, xl, xu, new, rng)
    return Pop


def vssps(problem, N, sLower, sUpper, rng):
    """Variable-sparsity striped population sampling: individuals span a range
    of sparsity levels; stripes tile the variables for coverage."""
    D = problem.n_var
    X = rng.uniform(problem.xl, problem.xu, (N, D))
    density = 1 - np.linspace(sLower, sUpper, N)          # high -> low density
    widths = np.clip(np.round(density * D).astype(int), 0, D)
    mask = np.zeros((N, D), bool)
    pos = 0
    for i in range(N):
        w = widths[i]
        if w > 0:
            mask[i, (np.arange(pos, pos + w) % D)] = True
            pos = (pos + w) % D
    X[~mask] = 0
    return X


# --------------------------------------------------------------------------- #
# Algorithm
# --------------------------------------------------------------------------- #
def snsgaii(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
            record_every=1, apf=None, sLower=0.75, sUpper=1.0):
    rng = np.random.default_rng(seed)
    X = vssps(problem, N, sLower, sUpper, rng)
    F = problem.evaluate(X); fe = N
    front = core.fast_nondominated_sort(F)
    cd = core.crowding_distance(F, front)
    RefV, Step = _apf_setup(problem, N) if apf else (None, None)

    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)
    while fe < max_fe:
        pool = tournament(2, N, front, -cd, rng=rng)
        off = ssbx(X[pool], problem.xl, problem.xu, rng)
        off = spm(off, problem.xl, problem.xu, rng)
        if apf:
            pD, _ = _apf_inject(problem, X, F, RefV, Step, rng, apf,
                                fe=fe, max_fe=max_fe)
            if pD is not None:
                off = np.vstack([off, pD])
        offF = problem.evaluate(off); fe += len(off)
        X = np.vstack([X, off]); F = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(F, N)
        X, F = X[keep], F[keep]
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        record(fe, F)
    record(fe, F, force=True)
    nd = nondominated(F)
    name = "SNSGA-II" if not apf else "APF-SNSGA-II"
    return Result(X=X[nd], F=F[nd], history=hist, name=name)


SNSGA = {"SNSGA-II": snsgaii}
