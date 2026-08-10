"""Algorithms: NSGA-II, SparseEA, and APF-NSGA-II (the proposed method).

APF-NSGA-II is a faithful port of ``APFNSGAII.m`` + ``DSTOperator.m``.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from . import core
from .metrics import igd, hypervolume, nondominated


class Result(dict):
    """Dict with attribute access: r.F, r.X, r.history."""
    __getattr__ = dict.get


def _tracker(problem, hv_ref, ref_pf, record_every):
    """Closure that records (fe, igd, hv) snapshots every `record_every` calls
    (the final call is always recorded)."""
    hist = {"fe": [], "igd": [], "hv": []}
    state = {"n": 0}

    def record(fe, F, force=False):
        state["n"] += 1
        if not force and record_every > 1 and (state["n"] % record_every):
            return
        nd = F[nondominated(F)]
        hist["fe"].append(int(fe))
        hist["igd"].append(igd(nd, ref_pf) if ref_pf is not None else np.nan)
        hist["hv"].append(hypervolume(nd, hv_ref) if hv_ref is not None else np.nan)

    return hist, record


# --------------------------------------------------------------------------- #
# NSGA-II (real-coded). Equivalent to the "SNSGAII" baseline on real problems.
# --------------------------------------------------------------------------- #
def nsga2(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
          record_every=1):
    rng = np.random.default_rng(seed)
    X = problem.xl + rng.random((N, problem.n_var)) * (problem.xu - problem.xl)
    F = problem.evaluate(X)
    fe = N
    front = core.fast_nondominated_sort(F)
    cd = core.crowding_distance(F, front)
    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)

    while fe < max_fe:
        pool = core.tournament_selection(2, N, front, -cd, rng=rng)
        off = core.operator_ga(X[pool], problem.xl, problem.xu, rng=rng)
        offF = problem.evaluate(off)
        fe += len(off)
        X = np.vstack([X, off])
        F = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(F, N)
        X, F = X[keep], F[keep]
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        record(fe, F)

    record(fe, F, force=True)
    nd = nondominated(F)
    return Result(X=X[nd], F=F[nd], history=hist, name="NSGA-II")


# --------------------------------------------------------------------------- #
# SparseEA (Tian et al., IEEE TEVC 2020) -- mask + dec encoding.
# --------------------------------------------------------------------------- #
def _sparse_fitness(problem, rng):
    """Front rank of each single-variable solution (lower = better variable)."""
    D = problem.n_var
    Dec = np.ones((D, D)) * (problem.xl + problem.xu) / 2.0
    Mask = np.eye(D)
    F = problem.evaluate(Dec * Mask)
    return core.fast_nondominated_sort(F).astype(float), D


def _sparse_offspring(pDec, pMask, qDec, qMask, fitness, problem, rng):
    """Produce one offspring (dec, mask) from parents p and q (SparseEA operator)."""
    D = problem.n_var
    # --- real part: SBX + PM on Dec ---
    child = core.operator_ga(np.vstack([pDec, qDec]), problem.xl, problem.xu, rng=rng)
    oDec = child[0]
    # --- binary mask crossover ---
    oMask = pMask.copy()
    if rng.random() < 0.5:
        idx = np.where((pMask == 1) & (qMask == 0))[0]
        if len(idx):
            j = idx[_bin_tour(fitness[idx], rng, worst=True)]
            oMask[j] = 0
    else:
        idx = np.where((pMask == 0) & (qMask == 1))[0]
        if len(idx):
            j = idx[_bin_tour(fitness[idx], rng, worst=False)]
            oMask[j] = 1
    # --- binary mask mutation ---
    if rng.random() < 0.5:
        idx = np.where(oMask == 1)[0]
        if len(idx):
            j = idx[_bin_tour(fitness[idx], rng, worst=True)]
            oMask[j] = 0
    else:
        idx = np.where(oMask == 0)[0]
        if len(idx):
            j = idx[_bin_tour(fitness[idx], rng, worst=False)]
            oMask[j] = 1
    return oDec, oMask


def _bin_tour(scores, rng, worst):
    """Index (into scores) of a binary-tournament winner.
    worst=True -> pick the larger (worse) score; else the smaller (better)."""
    a, b = rng.integers(0, len(scores), 2)
    if worst:
        return a if scores[a] >= scores[b] else b
    return a if scores[a] <= scores[b] else b


def sparseea(problem, N=100, max_fe=20000, seed=0, hv_ref=None, ref_pf=None,
             record_every=1):
    rng = np.random.default_rng(seed)
    D = problem.n_var
    fitness, _ = _sparse_fitness(problem, rng)
    fe = D
    # initialization
    Dec = problem.xl + rng.random((N, D)) * (problem.xu - problem.xl)
    Mask = np.zeros((N, D))
    for i in range(N):
        k = int(np.ceil(rng.random() * D))
        chosen = set()
        for _ in range(k):
            a, b = rng.integers(0, D, 2)
            chosen.add(a if fitness[a] <= fitness[b] else b)
        Mask[i, list(chosen)] = 1
    X = Dec * Mask
    F = problem.evaluate(X)
    fe += N
    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)

    while fe < max_fe:
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        pool = core.tournament_selection(2, N, front, -cd, rng=rng)
        oDec = np.zeros((N, D))
        oMask = np.zeros((N, D))
        for i in range(N):
            p, q = pool[i], pool[rng.integers(0, N)]
            oDec[i], oMask[i] = _sparse_offspring(
                Dec[p], Mask[p], Dec[q], Mask[q], fitness, problem, rng)
        off = oDec * oMask
        offF = problem.evaluate(off)
        fe += N
        Dec = np.vstack([Dec, oDec]); Mask = np.vstack([Mask, oMask])
        F = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(F, N)
        Dec, Mask, F = Dec[keep], Mask[keep], F[keep]
        record(fe, F)

    record(fe, F, force=True)
    Xf = Dec * Mask
    nd = nondominated(F)
    return Result(X=Xf[nd], F=F[nd], history=hist, name="SparseEA")


# --------------------------------------------------------------------------- #
# APF-NSGA-II  (proposed).  Port of APFNSGAII.m + DSTOperator.m.
# --------------------------------------------------------------------------- #
def _representative_selection(Obj, RefV, theta):
    """Port of GenerateRepresetativeSolution.m (IAPD-based selection)."""
    Obj = np.asarray(Obj, float)
    np_, M = Obj.shape
    lo, hi = Obj.min(0), Obj.max(0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    Obj = (Obj - lo) / span
    Nr = len(RefV)

    def cos_sim(A, B):
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return An @ Bn.T

    rr = np.clip(cos_sim(RefV, RefV), -1, 1)
    np.fill_diagonal(rr, 0.0)
    gamma = np.arccos(rr).min(axis=1)                    # min angle to other refs
    C = np.clip(cos_sim(Obj, RefV), -1, 1)               # cosine sim obj<->ref
    associate = C.argmax(axis=1)

    best = np.full(Nr, -1, dtype=int)
    used = np.zeros(np_, bool)
    for i in range(Nr):
        members = np.where(associate == i)[0]
        if len(members) > 1:
            apd = (1 + M * theta * C[members, i] / (gamma[i] + 1e-12)) \
                  * np.linalg.norm(Obj[members], axis=1)
            sel = members[apd.argmin()]
            best[i] = sel; used[sel] = True
        elif len(members) == 1:
            best[i] = members[0]; used[members[0]] = True
    for i in range(Nr):                                  # fill empty clusters
        if best[i] == -1:
            order = np.argsort(-C[:, i])
            for j in order:
                if not used[j]:
                    best[i] = j; used[j] = True
                    break
            if best[i] == -1:
                best[i] = order[0]
    return best


# Ablation configuration. None -> the full method. Keys:
#   directional (D): bi-directional bound-anchored sampling on/off
#   shrink      (R): geometric coarse-to-fine radius on/off (off = fixed radius)
#   iapd        (I): IAPD representative selection on/off (off = random reps)
#   direction   : 'bound' (L->X,U->X) or 'random' (C2 control)
#   control     : None | 'equal_random' | 'equal_perturb'  (C1 equal-offspring)
ABL_FULL = dict(directional=True, shrink=True, iapd=True,
                direction="bound", control=None)


def _abl(abl):
    cfg = dict(ABL_FULL)
    if abl:
        cfg.update(abl)
    return cfg


def _dst_operator(problem, X, F, RefV, Rate, Step, Sc, Ns, fe, max_fe, rng,
                  amplitude="mod", abl=None):
    """Port of DSTOperator.m with ablation switches (see ABL_FULL).
    amplitude='mod' matches the MATLAB code; 'abs' is continuous zero-preserving;
    'fixed' is the additive LSMOP-paper radius (also the C5 no-zero-preservation
    control on sparse problems)."""
    cfg = _abl(abl)
    L, U = problem.xl, problem.xu
    D = problem.n_var
    N = len(X)
    Nw = min(int(np.ceil(N / 10)), N)

    # cluster reference vectors -> Nw directional reference vectors
    if len(RefV) >= Nw:
        centroids, _ = kmeans2(RefV, Nw, minit="++", seed=int(rng.integers(1e9)))
    else:
        centroids = RefV
    t = fe / max_fe
    if cfg["iapd"]:
        best = _representative_selection(F, centroids, 0.1 ** t)          # I on
    else:
        best = rng.integers(0, N, min(Nw, N))                            # C3: random reps
    BestX = X[best]
    Nw = len(BestX)

    def frac(v):
        if amplitude == "mod":
            return np.mod(v, 1.0)          # sparse: preserves zeros
        if amplitude == "abs":
            return np.abs(v)               # sparse, continuous
        return np.ones_like(v)             # 'fixed'/no-frac: additive radius

    def rand_dir(n):
        d = rng.standard_normal((n, D))
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)

    perX1, perX2 = [], []
    for i in range(1, Sc + 2):                           # substages 1..Sc+1
        if Step[i - 1] < t <= Step[i]:
            expo = (i - 1) if cfg["shrink"] else 0       # R off -> fixed radius
            amp = 3.0 * 0.5 ** expo
            r = amp * frac(BestX)
            perX1.append(rng.uniform(BestX - r, BestX + r))
            if cfg["directional"] and t < Rate:          # D on, coarse phase
                if cfg["direction"] == "random":         # C2: random directions
                    dirL, dirU = rand_dir(Nw), rand_dir(Nw)
                else:
                    dL, dU = BestX - L, BestX - U
                    nL = np.linalg.norm(dL, axis=1, keepdims=True) + 1e-12
                    nU = np.linalg.norm(dU, axis=1, keepdims=True) + 1e-12
                    dirL, dirU = dL / nL, dU / nU
                interval = np.linalg.norm(U - L)
                for _ in range(Ns):
                    aL = rng.uniform(0, interval, (Nw, 1))
                    aU = rng.uniform(0, interval, (Nw, 1))
                    yL = L + aL * dirL
                    yU = U + aU * dirU
                    for y in (yL, yU):
                        ry = amp * frac(y)
                        perX2.append(rng.uniform(y - ry, y + ry))
    parts = perX1 + perX2
    if not parts:
        return np.zeros((0, D))
    PerX = np.vstack(parts)
    PerX = np.clip(PerX, L, U)

    # C1 equal-offspring control: keep the SAME count but replace the structured
    # offspring with naive ones (isolates mechanism vs "more/extra offspring").
    if cfg["control"] == "equal_random":
        PerX = rng.uniform(L, U, (len(PerX), D))
    elif cfg["control"] == "equal_perturb":
        idx = rng.integers(0, Nw, len(PerX))
        PerX = np.clip(BestX[idx] + rng.uniform(-3.0, 3.0, (len(PerX), D)), L, U)
    return PerX


def apf_nsga2(problem, N=100, max_fe=20000, seed=0, Rate=0.6, Sc=2, Ns=5,
              Up=0.8, amplitude="mod", hv_ref=None, ref_pf=None, record_every=1,
              abl=None, trigger=True, name=None):
    """Proposed framework with NSGA-II embedded. ``abl`` = ablation config
    (see ABL_FULL); ``trigger`` False = constant injection probability (T off)."""
    rng = np.random.default_rng(seed)
    RefV, _ = core.uniform_points(N, problem.n_obj)
    Step = np.zeros(Sc + 2)
    for r in range(1, Sc + 1):
        Step[r] = (r - r ** 2 / Sc ** 2) * Rate
    Step[Sc + 1] = 1.0

    X = problem.xl + rng.random((N, problem.n_var)) * (problem.xu - problem.xl)
    F = problem.evaluate(X)
    fe = N
    front = core.fast_nondominated_sort(F)
    cd = core.crowding_distance(F, front)
    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)

    while fe < max_fe:
        pool = core.tournament_selection(2, N, front, -cd, rng=rng)
        off = core.operator_ga(X[pool], problem.xl, problem.xu, rng=rng)
        prob = Up if not trigger else min(Up, (fe / max_fe / 3 - 1) ** 2)
        if rng.random() < prob:
            perX = _dst_operator(problem, X, F, RefV, Rate, Step, Sc, Ns,
                                 fe, max_fe, rng, amplitude=amplitude, abl=abl)
            if len(perX):
                off = np.vstack([off, perX])
        offF = problem.evaluate(off)
        fe += len(off)
        X = np.vstack([X, off])
        F = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(F, N)
        X, F = X[keep], F[keep]
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        record(fe, F)

    record(fe, F, force=True)
    nd = nondominated(F)
    tag = name or ("APF-NSGA-II" + ("" if amplitude == "mod" else f"({amplitude})"))
    return Result(X=X[nd], F=F[nd], history=hist, name=tag)


def apf_sparseea(problem, N=100, max_fe=20000, seed=0, Rate=0.6, Sc=2, Ns=5,
                 Up=0.8, amplitude="mod", hv_ref=None, ref_pf=None, record_every=1):
    """APF refinement embedded in a *sparse* base (SparseEA), i.e. the correct
    analogue of the paper's ``APFSNSGAII`` (APF over a sparsity-producing base).

    Each generation, with the same probability gate, the DST operator produces
    extra offspring from the current (already sparse) decision vectors; because
    those vectors contain exact zeros, the mod/abs amplitude term preserves the
    sparse pattern. DST offspring are folded back as (mask = x!=0, dec = x)."""
    rng = np.random.default_rng(seed)
    D = problem.n_var
    RefV, _ = core.uniform_points(N, problem.n_obj)
    Step = np.zeros(Sc + 2)
    for r in range(1, Sc + 1):
        Step[r] = (r - r ** 2 / Sc ** 2) * Rate
    Step[Sc + 1] = 1.0

    fitness, _ = _sparse_fitness(problem, rng)
    fe = D
    Dec = problem.xl + rng.random((N, D)) * (problem.xu - problem.xl)
    Mask = np.zeros((N, D))
    for i in range(N):
        k = int(np.ceil(rng.random() * D))
        chosen = set()
        for _ in range(k):
            a, b = rng.integers(0, D, 2)
            chosen.add(a if fitness[a] <= fitness[b] else b)
        Mask[i, list(chosen)] = 1
    F = problem.evaluate(Dec * Mask)
    fe += N
    hist, record = _tracker(problem, hv_ref, ref_pf, record_every)
    record(fe, F)

    while fe < max_fe:
        X = Dec * Mask
        front = core.fast_nondominated_sort(F)
        cd = core.crowding_distance(F, front)
        pool = core.tournament_selection(2, N, front, -cd, rng=rng)
        oDec = np.zeros((N, D)); oMask = np.zeros((N, D))
        for i in range(N):
            p, q = pool[i], pool[rng.integers(0, N)]
            oDec[i], oMask[i] = _sparse_offspring(
                Dec[p], Mask[p], Dec[q], Mask[q], fitness, problem, rng)
        # ---- APF/DST extra offspring on the sparse decision vectors ----
        if rng.random() < min(Up, (fe / max_fe / 3 - 1) ** 2):
            perX = _dst_operator(problem, X, F, RefV, Rate, Step, Sc, Ns,
                                 fe, max_fe, rng, amplitude=amplitude)
            if len(perX):
                oDec = np.vstack([oDec, perX])
                oMask = np.vstack([oMask, (np.abs(perX) > 1e-12).astype(float)])
        off = oDec * oMask
        offF = problem.evaluate(off)
        fe += len(off)
        Dec = np.vstack([Dec, oDec]); Mask = np.vstack([Mask, oMask])
        F = np.vstack([F, offF])
        keep = core.nsga2_environmental_selection(F, N)
        Dec, Mask, F = Dec[keep], Mask[keep], F[keep]
        record(fe, F)

    record(fe, F, force=True)
    Xf = Dec * Mask
    nd = nondominated(F)
    tag = "APF-SparseEA" + ("" if amplitude == "mod" else f"({amplitude})")
    return Result(X=Xf[nd], F=F[nd], history=hist, name=tag)


ALGORITHMS = {"NSGA-II": nsga2, "SparseEA": sparseea, "APF-NSGA-II": apf_nsga2,
              "APF-SparseEA": apf_sparseea}
