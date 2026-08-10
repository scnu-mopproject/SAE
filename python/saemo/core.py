"""Core evolutionary-multiobjective utilities.

Genetic operators (SBX + polynomial mutation) mirror PlatEMO's ``OperatorGA``
defaults so that ported algorithms behave like their MATLAB counterparts.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Genetic operators (real-coded), matching PlatEMO OperatorGA defaults.
# --------------------------------------------------------------------------- #
def sbx_crossover(parent1, parent2, xl, xu, prob_c=1.0, eta_c=20.0, rng=None):
    """Simulated binary crossover on two parent matrices (rows = individuals)."""
    rng = rng or np.random.default_rng()
    p1, p2 = np.asarray(parent1, float), np.asarray(parent2, float)
    n, d = p1.shape
    beta = np.ones((n, d))
    mu = rng.random((n, d))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (eta_c + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (eta_c + 1))
    beta *= (-1) ** rng.integers(0, 2, size=(n, d))
    beta[rng.random((n, d)) < 0.5] = 1                       # per-gene swap chance
    beta[np.tile(rng.random((n, 1)) > prob_c, (1, d))] = 1   # per-pair crossover chance
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    off = np.vstack([c1, c2])
    return np.clip(off, xl, xu)


def polynomial_mutation(pop, xl, xu, prob_m=None, eta_m=20.0, rng=None):
    """Polynomial mutation. Default per-gene prob = 1/d (PlatEMO default)."""
    rng = rng or np.random.default_rng()
    x = np.array(pop, float)
    n, d = x.shape
    prob_m = 1.0 / d if prob_m is None else prob_m
    xl = np.broadcast_to(xl, x.shape)
    xu = np.broadcast_to(xu, x.shape)
    site = rng.random((n, d)) < prob_m
    mu = rng.random((n, d))
    delta1 = (x - xl) / (xu - xl)
    delta2 = (xu - x) / (xu - xl)
    span = xu - xl
    # lower tail
    lo = site & (mu <= 0.5)
    val = 2 * mu + (1 - 2 * mu) * (1 - delta1) ** (eta_m + 1)
    x[lo] += (val[lo] ** (1 / (eta_m + 1)) - 1) * span[lo]
    # upper tail
    hi = site & (mu > 0.5)
    val = 2 * (1 - mu) + 2 * (mu - 0.5) * (1 - delta2) ** (eta_m + 1)
    x[hi] += (1 - val[hi] ** (1 / (eta_m + 1))) * span[hi]
    return np.clip(x, xl, xu)


def operator_ga(parents, xl, xu, rng=None, **kw):
    """SBX + polynomial mutation producing len(parents) offspring."""
    rng = rng or np.random.default_rng()
    parents = np.asarray(parents, float)
    n = len(parents)
    half = n // 2
    p1 = parents[:half]
    p2 = parents[half:2 * half]
    off = sbx_crossover(p1, p2, xl, xu, rng=rng,
                        prob_c=kw.get("prob_c", 1.0), eta_c=kw.get("eta_c", 20.0))
    off = polynomial_mutation(off, xl, xu, rng=rng,
                              prob_m=kw.get("prob_m"), eta_m=kw.get("eta_m", 20.0))
    return off[:n]


# --------------------------------------------------------------------------- #
# Non-dominated sorting and crowding distance (NSGA-II).
# --------------------------------------------------------------------------- #
def fast_nondominated_sort(F, n_stop=None):
    """Return front index per row (0 = best front)."""
    F = np.asarray(F, float)
    n = len(F)
    n_stop = n if n_stop is None else n_stop
    dominated_count = np.zeros(n, dtype=int)
    dominates = [[] for _ in range(n)]
    for i in range(n):
        di = F[i]
        le = np.all(F >= di, axis=1)
        lt = np.any(F > di, axis=1)
        dom_by_i = le & lt & (np.arange(n) != i)      # i dominates these
        dominates[i] = np.where(dom_by_i)[0].tolist()
        ge = np.all(F <= di, axis=1)
        gt = np.any(F < di, axis=1)
        dominated_count[i] = np.sum(ge & gt & (np.arange(n) != i))
    front = -np.ones(n, dtype=int)
    current = np.where(dominated_count == 0)[0].tolist()
    f = 0
    counted = 0
    while current:
        front[current] = f
        counted += len(current)
        nxt = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        current = nxt
        f += 1
        if counted >= n_stop:
            break
    front[front == -1] = f      # leftovers get a worst-front label
    return front


def crowding_distance(F, front):
    """Crowding distance per individual, computed within each front."""
    F = np.asarray(F, float)
    n, m = F.shape
    cd = np.zeros(n)
    for fr in np.unique(front):
        idx = np.where(front == fr)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        sub = F[idx]
        for k in range(m):
            order = np.argsort(sub[:, k])
            cd[idx[order[0]]] = cd[idx[order[-1]]] = np.inf
            span = sub[order[-1], k] - sub[order[0], k]
            if span == 0:
                continue
            cd[idx[order[1:-1]]] += (sub[order[2:], k] - sub[order[:-2], k]) / span
    return cd


def nsga2_environmental_selection(F, n):
    """Return indices of the n survivors (NSGA-II selection)."""
    F = np.asarray(F, float)
    front = fast_nondominated_sort(F, n_stop=n)
    cd = crowding_distance(F, front)
    order = np.lexsort((-cd, front))     # sort by front asc, then crowding desc
    return order[:n]


def tournament_selection(k, n_out, *keys, rng=None):
    """k-ary tournament. ``keys`` are ranked ascending (smaller = better)."""
    rng = rng or np.random.default_rng()
    n = len(keys[0])
    cand = rng.integers(0, n, size=(n_out, k))
    key = np.zeros((n_out, k))
    # lexicographic: combine keys into a comparable score via ranks
    ranks = np.zeros((n, len(keys)))
    for c, key_arr in enumerate(keys):
        ranks[:, c] = np.argsort(np.argsort(key_arr))
    score = ranks @ (len(keys) ** np.arange(len(keys) - 1, -1, -1))
    best = cand[np.arange(n_out), np.argmin(score[cand], axis=1)]
    return best


# --------------------------------------------------------------------------- #
# Uniform reference vectors (Das-Dennis simplex-lattice).
# --------------------------------------------------------------------------- #
def uniform_points(n, m):
    """Approximately ``n`` uniform points on the (m-1)-simplex; returns (P, n)."""
    from scipy.special import comb
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1
    import itertools
    def das_dennis(h, m):
        pts = []
        for c in itertools.combinations(range(1, h + m), m - 1):
            c = np.array((0,) + c + (h + m,))
            pts.append((c[1:] - c[:-1] - 1) / h)
        return np.array(pts, float)
    W = das_dennis(h1, m)
    if h1 < m:                      # add an inner layer for many-objective cases
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            W2 = das_dennis(h2, m) / 2 + 0.5 / m
            W = np.vstack([W, W2])
    W = np.maximum(W, 1e-6)
    return W, len(W)
