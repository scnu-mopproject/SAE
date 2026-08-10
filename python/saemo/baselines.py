"""Faithful Python ports of three PlatEMO sparse-MOEA baselines:

    MSKEA  (Ding et al., Swarm Evol. Comput. 2022)  -- multi-stage knowledge-guided
    MGCEA  (Tian et al., Swarm Evol. Comput. 2024)  -- multi-granularity clustering
    BLIGEA (Zou  et al., Swarm Evol. Comput. 2025)  -- bi-level interactive grouping

Ports the PlatEMO .m sources (mask+dec sparse encoding, SPEA2 environmental
selection). RNG differs from MATLAB, so results match structurally/statistically
rather than bit-for-bit. Each returns a saemo ``Result``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2

from . import core
from .algorithms import Result, _tracker
from .metrics import nondominated


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _nd_full(objs):
    """Full non-dominated sort, 1-indexed fronts (PlatEMO convention)."""
    return core.fast_nondominated_sort(np.asarray(objs, float)) + 1


def tournament(K, N, *keys, rng):
    """PlatEMO TournamentSelection(K,N,keys...): first key primary, smaller=better.
    Returns N indices (winners of K-ary tournaments)."""
    cols = np.column_stack([np.asarray(k, float).ravel() for k in keys])
    order = np.lexsort(cols.T[::-1])              # first column primary
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    pool = len(rank)
    cand = rng.integers(0, pool, (K, N))
    best = np.argmin(rank[cand], axis=0)
    return cand[best, np.arange(N)]


def operator_ga_half(problem, parentDec, rng, proC=1.0, disC=20.0, proM=1.0, disM=20.0):
    """PlatEMO OperatorGAhalf: SBX (one child) + polynomial mutation, N/2 out."""
    P = np.asarray(parentDec, float)
    N = len(P); half = N // 2
    p1, p2 = P[:half], P[half:2 * half]
    n, d = p1.shape
    beta = np.ones((n, d)); mu = rng.random((n, d))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta *= (-1) ** rng.integers(0, 2, (n, d))
    beta[rng.random((n, d)) < 0.5] = 1
    beta[np.tile(rng.random((n, 1)) > proC, (1, d))] = 1
    off = (p1 + p2) / 2 + beta * (p1 - p2) / 2
    off = core.polynomial_mutation(off, problem.xl, problem.xu, rng=rng,
                                   prob_m=proM / d, eta_m=disM)
    return np.clip(off, problem.xl, problem.xu)


def _truncation(objs, K, rng):
    """SPEA2 density truncation: mark K points for deletion (nearest-neighbour)."""
    n = len(objs)
    if K <= 0:
        return np.zeros(n, bool)
    D = cdist(objs, objs)
    np.fill_diagonal(D, np.inf)
    dele = np.zeros(n, bool)
    while dele.sum() < K:
        remain = np.where(~dele)[0]
        Temp = np.sort(D[np.ix_(remain, remain)], axis=1)
        rank = np.lexsort(Temp.T[::-1])           # sortrows ascending
        dele[remain[rank[0]]] = True
    return dele


def _spea2_ndsort_select(Objs, Dec, Mask, N, rng):
    """MSKEA / BLIGEA environmental selection: dedup, NDSort fronts, truncate the
    last front. Returns (Objs, Dec, Mask, FrontNo, CrowdDis)."""
    _, uni = np.unique(Objs, axis=0, return_index=True)
    uni = np.sort(uni)
    Objs, Dec, Mask = Objs[uni], Dec[uni], Mask[uni]
    n = len(Objs); N = min(N, n)
    front = _nd_full(Objs)
    # find MaxFNo where cumulative count reaches N
    order_fr = np.sort(np.unique(front))
    cum = 0
    for f in order_fr:
        cum += np.sum(front == f)
        if cum >= N:
            maxf = f
            break
    Next = front < maxf
    # normalize by front-1 range (as in the .m)
    f1 = Objs[front == 1]
    fmax, fmin = f1.max(0), f1.min(0)
    span = np.where(fmax - fmin == 0, 1.0, fmax - fmin)
    norm = (Objs - fmin) / span
    last = np.where(front == maxf)[0]
    K = len(last) - (N - Next.sum())
    dele = _truncation(norm[last], K, rng)
    Next[last[~dele]] = True
    Objs, Dec, Mask = Objs[Next], Dec[Next], Mask[Next]
    fr = _nd_full(Objs)
    cd = core.crowding_distance(Objs, fr - 1)
    return Objs, Dec, Mask, fr, cd


def _spea2_fitness(objs):
    """SPEA2 fitness R(i)+D(i) (MGCEA CalFitness)."""
    P = np.asarray(objs, float)
    n = len(P)
    dom = np.zeros((n, n), bool)
    for i in range(n):
        less = np.any(P[i] < P, axis=1)
        more = np.any(P[i] > P, axis=1)
        dom[i] = less & ~more                     # i dominates j
        dom[i, i] = False
    S = dom.sum(1)
    R = np.array([S[dom[:, i]].sum() for i in range(n)], float)
    _DP = cdist(P, P); np.fill_diagonal(_DP, np.inf); Dist = np.sort(_DP, axis=1)
    k = int(np.floor(np.sqrt(n)))
    Dens = 1.0 / (Dist[:, min(k, n - 1)] + 2)
    return R + Dens


def _spea2_fitness_select(Objs, Dec, Mask, N, rng):
    """MGCEA environmental selection (SPEA2 fitness < 1)."""
    _, uni = np.unique(Objs, axis=0, return_index=True)
    uni = np.sort(uni)
    Objs, Dec, Mask = Objs[uni], Dec[uni], Mask[uni]
    n = len(Objs); N = min(N, n)
    fit = _spea2_fitness(Objs)
    Next = fit < 1
    if Next.sum() < N:
        rank = np.argsort(fit)
        Next[rank[:N]] = True
    elif Next.sum() > N:
        idx = np.where(Next)[0]
        dele = _truncation(Objs[idx], Next.sum() - N, rng)
        Next[idx[dele]] = False
    return Objs[Next], Dec[Next], Mask[Next], fit[Next]


def _var_scores(problem, rng, reducer):
    """Single-variable NDSort scores + the temp population (evaluated once each)."""
    D = problem.n_var
    reps = 1 + 4 * int(np.any(problem.xu != problem.xl))   # 5 for real problems
    rows, Tdec, Tmask, Tobj = [], [], [], []
    fe = 0
    for _ in range(reps):
        Dec = rng.uniform(problem.xl, problem.xu, (D, D))
        Mask = np.eye(D)
        F = problem.evaluate(Dec * Mask); fe += D
        rows.append(_nd_full(F))
        Tdec.append(Dec); Tmask.append(Mask); Tobj.append(F)
    score = reducer(np.array(rows))
    return score, np.vstack(Tdec), np.vstack(Tmask), np.vstack(Tobj), fe


def _init_mask(problem, N, score, rng):
    D = problem.n_var
    Mask = np.zeros((N, D))
    for i in range(N):
        k = int(np.ceil(rng.random() * D))
        idx = tournament(2, k, score, rng=rng)
        Mask[i, idx] = 1
    return Mask


# --------------------------------------------------------------------------- #
# APF refinement hook: lets SAE/APF wrap any sparse base (base as SAE's "embedded
# optimizer"). Injects DST offspring each generation; because the base produces
# sparse decision vectors, the mod/abs amplitude preserves the zeros.
# --------------------------------------------------------------------------- #
def _apf_setup(problem, N, Rate=0.6, Sc=2):
    RefV, _ = core.uniform_points(N, problem.n_obj)
    Step = np.zeros(Sc + 2)
    for r in range(1, Sc + 1):
        Step[r] = (r - r ** 2 / Sc ** 2) * Rate
    Step[Sc + 1] = 1.0
    return RefV, Step


def _apf_inject(problem, X, F, RefV, Step, rng, amplitude,
                Rate=0.6, Sc=2, Ns=5, Up=0.8, fe=0, max_fe=1):
    from .algorithms import _dst_operator
    if rng.random() < min(Up, (fe / max_fe / 3 - 1) ** 2):
        perX = _dst_operator(problem, X, F, RefV, Rate, Step, Sc, Ns,
                             fe, max_fe, rng, amplitude=amplitude)
        if len(perX):
            return perX, (np.abs(perX) > 1e-12).astype(float)
    return None, None


# =========================================================================== #
# MSKEA
# =========================================================================== #
def mskea(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
          record_every=1, apf=None):
    rng = np.random.default_rng(seed)
    D = problem.n_var
    pv, TDec, TMask, TObj, fe = _var_scores(problem, rng, lambda a: a.sum(0))
    Dec0 = rng.uniform(problem.xl, problem.xu, (N, D))
    Mask0 = _init_mask(problem, N, pv, rng)
    O0 = problem.evaluate(Dec0 * Mask0); fe += N
    Objs = np.vstack([O0, TObj]); Dec = np.vstack([Dec0, TDec]); Mask = np.vstack([Mask0, TMask])
    Objs, Dec, Mask, FrontNo, CrowdDis = _spea2_ndsort_select(Objs, Dec, Mask, N, rng)
    sv = np.zeros(D); fv = np.zeros(D); last_num = 0
    RefV, Step = _apf_setup(problem, N) if apf else (None, None)

    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, Objs)
    while fe < max_fe:
        pool = tournament(2, 2 * N, FrontNo, -CrowdDis, rng=rng)
        delta = fe / max_fe
        if delta < 0.618:
            fv = np.std(Dec[FrontNo == 1], axis=0, ddof=0)
        f1mask = Mask[FrontNo == 1]
        tnum = len(f1mask); tvote = f1mask.sum(0)
        if tnum > 0:
            sv = (last_num / (last_num + tnum)) * sv + (tnum / (last_num + tnum)) * (tvote / tnum)
            last_num = tnum
        if delta < 0.618:
            pv = pv * (1 - sv) * np.sqrt(delta) + pv
        r = delta / 0.618
        if r < 0.618:
            oDec, oMask = _mskea_pvfv(problem, Dec[pool], Mask[pool], pv, fv, delta, rng)
        elif r >= 0.618 and delta < 0.618:
            if rng.random() < 0.5:
                oDec, oMask = _mskea_sv(problem, Dec[pool], Mask[pool], sv, rng)
            else:
                oDec, oMask = _mskea_pvfv(problem, Dec[pool], Mask[pool], pv, fv, delta, rng)
        else:
            oDec, oMask = _mskea_sv(problem, Dec[pool], Mask[pool], sv, rng)
        if apf:
            pD, pM = _apf_inject(problem, Dec * Mask, Objs, RefV, Step, rng, apf,
                                 fe=fe, max_fe=max_fe)
            if pD is not None:
                oDec = np.vstack([oDec, pD]); oMask = np.vstack([oMask, pM])
        oObj = problem.evaluate(oDec * oMask); fe += len(oDec)
        Objs = np.vstack([Objs, oObj]); Dec = np.vstack([Dec, oDec]); Mask = np.vstack([Mask, oMask])
        Objs, Dec, Mask, FrontNo, CrowdDis = _spea2_ndsort_select(Objs, Dec, Mask, N, rng)
        record(fe, Objs)
    record(fe, Objs, force=True)
    nd = nondominated(Objs)
    name = "MSKEA" if not apf else "APF-MSKEA"
    return Result(X=(Dec * Mask)[nd], F=Objs[nd], history=hist, name=name)


def _ts1(score, rng):
    return tournament(2, 1, score, rng=rng)[0] if len(score) else None


def _mskea_pvfv(problem, PDec, PMask, pv, fv, delta, rng):
    N = len(PDec); half = N // 2; D = problem.n_var
    P1, P2 = PMask[:half], PMask[half:2 * half]
    oMask = P1.copy()
    for i in range(half):
        if rng.random() < 0.5:
            idx = np.where(P1[i].astype(bool) & ~P2[i].astype(bool))[0]
            j = _ts1(-pv[idx], rng)
            if j is not None:
                oMask[i, idx[j]] = 0
        else:
            idx = np.where(~P1[i].astype(bool) & P2[i].astype(bool))[0]
            j = _ts1(pv[idx], rng)
            if j is not None:
                oMask[i, idx[j]] = P2[i, idx[j]]
    if rng.random() < (1 - delta):
        fvec = fv.copy(); fvec[fv > 0] = 1
        for i in range(half):
            idx = np.where(oMask[i] != fvec)[0]
            if rng.random() < 0.5:
                j = _ts1(-fv[idx], rng)
                if j is not None:
                    oMask[i, idx[j]] = 1
            else:
                j = _ts1(fv[idx], rng)
                if j is not None:
                    oMask[i, idx[j]] = 0
    else:
        for i in range(half):
            if rng.random() < 0.5:
                idx = np.where(oMask[i])[0]
                j = _ts1(-pv[idx], rng)
                if j is not None:
                    oMask[i, idx[j]] = 0
            else:
                idx = np.where(~oMask[i].astype(bool))[0]
                j = _ts1(pv[idx], rng)
                if j is not None:
                    oMask[i, idx[j]] = 1
    oDec = operator_ga_half(problem, PDec, rng)
    return oDec, oMask


def _mskea_sv(problem, PDec, PMask, sv, rng):
    N = len(PDec); half = N // 2; D = problem.n_var
    P1, P2 = PMask[:half], PMask[half:2 * half]
    oMask = P1.copy()
    rate0 = sv; rate1 = 1 - sv
    for i in range(half):
        diff = np.where(P1[i] != P2[i])[0]
        on = oMask[i, diff].astype(bool)
        rate = np.where(on, rate1[diff], rate0[diff])
        ex = rng.random(len(diff)) < rate
        oMask[i, diff[ex]] = 1 - oMask[i, diff[ex]]
    mut = rng.random((half, D)) < 1.0 / D
    for i in range(half):
        sub = np.where(mut[i])[0]
        if len(sub):
            on = oMask[i, sub].astype(bool)
            rate = np.where(on, rate1[sub], rate0[sub])
            ex = rng.random(len(sub)) < rate
            oMask[i, sub[ex]] = 1 - oMask[i, sub[ex]]
    oDec = operator_ga_half(problem, PDec, rng)
    return oDec, oMask


# =========================================================================== #
# MGCEA
# =========================================================================== #
def _mgcea_fitnesscal(problem, rng):
    D = problem.n_var
    Fit = np.zeros((5, D)); Tdec, Tmask, Tobj = [], [], []
    interval = (problem.xu - problem.xl) / 5
    fe = 0
    for i in range(5):
        for _ in range(2):
            lo = problem.xl + interval * i
            hi = problem.xl + interval * (i + 1)
            Dec = rng.uniform(lo, hi, (D, D)); Mask = np.eye(D)
            F = problem.evaluate(Dec * Mask); fe += D
            Fit[i] += _nd_full(F)
            Tdec.append(Dec); Tmask.append(Mask); Tobj.append(F)
    Tdec, Tmask, Tobj = np.vstack(Tdec), np.vstack(Tmask), np.vstack(Tobj)
    finit = Fit.sum(0)
    lbl, _ = kmeans2(finit[:, None], 2, minit="++", seed=int(rng.integers(1e9)))
    lab = np.argmin(np.abs(finit[:, None] - lbl.ravel()[None, :]), axis=1) + 1
    n1, n2 = np.sum(lab == 1), np.sum(lab == 2)
    v1, v2 = finit[lab == 1].sum(), finit[lab == 2].sum()
    if v1 < v2:
        sparse_rate = n1 / (n1 + n2)
    else:
        sparse_rate = n2 / (n1 + n2)
    return finit, sparse_rate, Tdec, Tmask, Tobj, fe


def _mgcea_update_layer(sparse_rate, stage, fitness, D, mask, rng):
    group_num = np.ceil(11 - stage) / 100 * D
    group_num = int(np.ceil(sparse_rate * 10 * group_num))
    group_num = max(1, group_num)
    if mask is None or mask.sum() == 0:
        order = np.argsort(fitness + rng.random(D))
    else:
        order = np.argsort(fitness + (mask == 0).sum(0) / 100000.0)
    layer_sorted = np.ceil(np.arange(1, D + 1) / group_num).astype(int)
    layer = np.zeros(D, int)
    layer[order] = layer_sorted
    return layer, layer.max()


def mgcea(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
          record_every=1, apf=None):
    rng = np.random.default_rng(seed)
    D = problem.n_var
    fitness, sparse_rate, TDec, TMask, TObj, fe = _mgcea_fitnesscal(problem, rng)
    Objs, Dec, Mask, FitS = _spea2_fitness_select(TObj, TDec, TMask, N, rng)
    near = int(np.ceil(fe / (max_fe / 10)))
    layer, layer_max = _mgcea_update_layer(sparse_rate, near, fitness, D, None, rng)
    RefV, Step = _apf_setup(problem, N) if apf else (None, None)

    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, Objs)
    while fe < max_fe:
        pool = tournament(2, 2 * N, FitS, rng=rng)
        stage = int(np.ceil(fe / (max_fe / 10)))
        if stage != near:
            near = stage
            layer, layer_max = _mgcea_update_layer(sparse_rate, stage, fitness, D, Mask, rng)
        oDec, oMask = _mgcea_operator(problem, Dec[pool], Mask[pool], layer, layer_max, rng)
        if apf:
            pD, pM = _apf_inject(problem, Dec * Mask, Objs, RefV, Step, rng, apf,
                                 fe=fe, max_fe=max_fe)
            if pD is not None:
                oDec = np.vstack([oDec, pD]); oMask = np.vstack([oMask, pM])
        oObj = problem.evaluate(oDec * oMask); fe += len(oDec)
        Objs = np.vstack([Objs, oObj]); Dec = np.vstack([Dec, oDec]); Mask = np.vstack([Mask, oMask])
        Objs, Dec, Mask, FitS = _spea2_fitness_select(Objs, Dec, Mask, N, rng)
        record(fe, Objs)
    record(fe, Objs, force=True)
    nd = nondominated(Objs)
    name = "MGCEA" if not apf else "APF-MGCEA"
    return Result(X=(Dec * Mask)[nd], F=Objs[nd], history=hist, name=name)


def _mgcea_operator(problem, PDec, PMask, layer, layer_max, rng):
    N = len(PDec); half = N // 2; D = problem.n_var
    P1, P2 = PMask[:half], PMask[half:2 * half]
    oMask = P1.copy()
    sr1 = P1.sum(1); sr2 = P2.sum(1)
    rate = sr1 / (sr1 + sr2 + 1e-12)
    idx = rng.random((half, D)) > rate[:, None] / 2
    oMask[idx] = P2[idx]
    for i in range(half):
        up, down = 1, layer_max
        for _ in range(layer_max):
            tu_layer = np.where(layer == up)[0]
            tu = tu_layer[oMask[i, tu_layer] == 0]
            td_layer = np.where(layer == down)[0]
            td = td_layer[oMask[i, td_layer] == 1]
            if rng.random() < 0.5:
                if len(tu) and rng.random() < 0.5:
                    sel = rng.choice(tu, int(np.ceil(len(tu) / 2)), replace=False)
                    oMask[i, sel] = 1
                if rng.random() < 0.5:
                    up += 1
                else:
                    break
            else:
                if len(td) and rng.random() < 0.5:
                    sel = rng.choice(td, int(np.ceil(len(td) / 2)), replace=False)
                    oMask[i, sel] = 0
                if rng.random() < 0.5:
                    down -= 1
                else:
                    break
            if up >= down:
                break
    oDec = operator_ga_half(problem, PDec, rng)
    return oDec, oMask


# =========================================================================== #
# BLIGEA
# =========================================================================== #
def bligea(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
           record_every=1, apf=None):
    rng = np.random.default_rng(seed)
    D = problem.n_var
    fitness, TDec, TMask, TObj, fe = _var_scores(
        problem, rng, lambda a: np.median(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0))
    Dec0 = rng.uniform(problem.xl, problem.xu, (N, D))
    Mask0 = _init_mask(problem, N, fitness, rng)
    O0 = problem.evaluate(Dec0 * Mask0); fe += N
    t = float(np.mean(fitness))
    Objs = np.vstack([O0, TObj]); Dec = np.vstack([Dec0, TDec]); Mask = np.vstack([Mask0, TMask])
    Objs, Dec, Mask, FrontNo, CrowdDis = _spea2_ndsort_select(Objs, Dec, Mask, N, rng)
    sv = np.zeros(D); last_num = 0
    RefV, Step = _apf_setup(problem, N) if apf else (None, None)

    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, Objs)
    while fe < max_fe:
        it = fe / max_fe
        f1 = Mask[FrontNo == 1]
        tnum = len(f1); tvote = f1.sum(0)
        if tnum > 0:
            sv = (last_num / (last_num + tnum)) * sv + (tnum / (last_num + tnum)) * (tvote / tnum)
            last_num = tnum
        fv = np.std(Dec[FrontNo == 1], axis=0, ddof=0)
        fitness = np.maximum(1, fitness - np.sqrt(t) * it * sv)
        fitness = np.minimum(fitness.max(), fitness + np.sqrt(t) * it * (1 - sv))
        pool = tournament(2, 2 * N, FrontNo, -CrowdDis, rng=rng)
        oDec, oMask = _bligea_operator(problem, Dec[pool], Mask[pool], fitness, it, sv, fv, rng)
        if apf:
            pD, pM = _apf_inject(problem, Dec * Mask, Objs, RefV, Step, rng, apf,
                                 fe=fe, max_fe=max_fe)
            if pD is not None:
                oDec = np.vstack([oDec, pD]); oMask = np.vstack([oMask, pM])
        oObj = problem.evaluate(oDec * oMask); fe += len(oDec)
        Objs = np.vstack([Objs, oObj]); Dec = np.vstack([Dec, oDec]); Mask = np.vstack([Mask, oMask])
        Objs, Dec, Mask, FrontNo, CrowdDis = _spea2_ndsort_select(Objs, Dec, Mask, N, rng)
        record(fe, Objs)
    record(fe, Objs, force=True)
    nd = nondominated(Objs)
    name = "BLIGEA" if not apf else "APF-BLIGEA"
    return Result(X=(Dec * Mask)[nd], F=Objs[nd], history=hist, name=name)


def _mask_group(it, mask, n_groups, fv):
    D = mask.shape[1]
    vpg = max(1, D // n_groups)
    vars_ = mask.sum(0)
    order = np.lexsort(np.vstack([vars_, fv]))[::-1]   # sortrows([fv;vars]','descend')
    out = -np.ones(D, int)
    for i in range(n_groups - 1):
        out[order[i * vpg:(i + 1) * vpg]] = i + 1
    out[order[(n_groups - 1) * vpg:]] = n_groups
    return out


def _bligea_operator(problem, PDec, PMask, fitness, it, sv, fv, rng):
    N = len(PDec); half = N // 2; D = problem.n_var
    P1d, P2d = PDec[:half], PDec[half:2 * half]
    P1, P2 = PMask[:half], PMask[half:2 * half]
    oMask = P1.copy()
    n_groups = max(1, int(np.floor(D * it)))
    Ksel = max(1, int(np.ceil(np.mean(sv) * D)))
    for i in range(half):
        if rng.random() < 0.5:
            idx = np.where(P1[i].astype(bool) & ~P2[i].astype(bool))[0]
            if len(idx):
                j = tournament(Ksel, 1, -fitness[idx], rng=rng)[0]
                oMask[i, idx[j]] = 0
        else:
            idx = np.where(~P1[i].astype(bool) & P2[i].astype(bool))[0]
            if len(idx):
                j = tournament(Ksel, 1, fitness[idx], rng=rng)[0]
                oMask[i, idx[j]] = P2[i, idx[j]]
    grp = _mask_group(it, oMask, n_groups, fv)
    gmax = max(2, grp.max())
    half_g = int(np.ceil(gmax / 2))
    use_fit = rng.random() > 0.5
    for i in range(half):
        c1 = rng.integers(1, half_g + 1)
        c2 = rng.integers(1, half_g + 1) + gmax // 2
        S1 = grp == c1; S2 = grp == c2
        if use_fit:
            s1 = (S1 * fitness).sum(); s2 = (S2 * fitness).sum()
        else:
            s1 = (S1 * (1 / (0.1 + sv))).sum(); s2 = (S2 * (1 / (0.1 + sv))).sum()
        if rng.random() < it:        # flip to 0
            ind = S2 if s1 < s2 else S1
            oMask[i, ind] = 0
        else:                        # flip to 1
            ind = S1 if s1 < s2 else S2
            oMask[i, ind] = 1
    oDec = _bligea_dec(problem, P1d, P2d, oMask, sv, it, rng)
    return oDec, oMask


def _bligea_dec(problem, P1, P2, oMask, sv, it, rng, disC=20.0, disM=20.0, proC=1.0):
    n, D = P1.shape
    beta = np.ones((n, D)); mu = rng.random((n, D))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta *= (-1) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1
    beta[np.tile(rng.random((n, 1)) > proC, (1, D))] = 1
    off = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    off = np.clip(off, problem.xl, problem.xu)
    # group-wise polynomial mutation (GLP CreateGroups over active sv-variables)
    n_groups = 4
    idx = (sv > 0) & (sv < 1)
    count = int(idx.sum()); vpg = max(1, count // n_groups)
    grp = -np.ones(D, int)
    if count > 0:
        vmean = off.mean(0)
        order = np.argsort(-vmean[idx])
        rankpos = np.full(D, -1); rankpos[np.where(idx)[0][order]] = np.arange(count) + 1
        for g in range(1, n_groups + 1):
            grp[(rankpos >= vpg * (g - 1) + 1) & (rankpos < vpg * g)] = g
    chosen = rng.integers(1, n_groups + 1, n)
    Site = grp[None, :] == chosen[:, None]
    mu = np.tile(rng.random((n, 1)), (1, D))
    L = np.broadcast_to(problem.xl, off.shape); U = np.broadcast_to(problem.xu, off.shape)
    span = U - L
    t1 = Site & (mu <= 0.5)
    off[t1] += span[t1] * ((2 * mu[t1] + (1 - 2 * mu[t1]) *
              (1 - (off[t1] - L[t1]) / span[t1]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t2 = Site & (mu > 0.5)
    off[t2] += span[t2] * (1 - (2 * (1 - mu[t2]) + 2 * (mu[t2] - 0.5) *
              (1 - (U[t2] - off[t2]) / span[t2]) ** (disM + 1)) ** (1 / (disM + 1)))
    return np.clip(off, problem.xl, problem.xu)


BASELINES = {"MSKEA": mskea, "MGCEA": mgcea, "BLIGEA": bligea}
