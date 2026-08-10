"""Performance indicators: IGD and hypervolume (HV)."""
from __future__ import annotations

import numpy as np


def igd(F, ref_pf):
    """Inverted Generational Distance: mean distance from each reference PF
    point to its nearest obtained objective vector (smaller is better)."""
    F = np.asarray(F, float)
    R = np.asarray(ref_pf, float)
    if len(F) == 0:
        return np.inf
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1).mean()


def _hv_2d(F, ref):
    """Exact 2-D hypervolume dominated by the non-dominated set F w.r.t. ref."""
    F = np.asarray(F, float)
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    # keep the non-dominated staircase
    keep, best_y = [], np.inf
    for p in F:
        if p[1] < best_y:
            keep.append(p)
            best_y = p[1]
    F = np.array(keep)
    vol, prev_x = 0.0, F[0, 0]
    prev_x = F[:, 0]
    widths = np.append(np.diff(F[:, 0]), ref[0] - F[-1, 0])
    heights = ref[1] - F[:, 1]
    # accumulate rectangles of the staircase
    vol = 0.0
    for i in range(len(F)):
        vol += (ref[0] - F[i, 0]) * 0.0  # placeholder to keep structure clear
    # simpler exact sweep:
    vol = 0.0
    prev_x = ref[0]
    for i in range(len(F) - 1, -1, -1):
        vol += (prev_x - F[i, 0]) * (ref[1] - F[i, 1])
        prev_x = F[i, 0]
    return vol


def hypervolume(F, ref, n_sample=100000, rng=None):
    """Hypervolume dominated by F relative to reference point ``ref``.

    Exact for 2 objectives; Monte-Carlo estimate for 3+.
    Objectives are assumed to be minimized.
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.ndim == 1:
        F = F[None, :]
    m = F.shape[1]
    if m == 2:
        return _hv_2d(F, ref)
    rng = rng or np.random.default_rng(0)
    lo = F.min(axis=0)
    lo = np.minimum(lo, ref)
    box = ref - lo
    if np.any(box <= 0):
        return 0.0
    S = lo + rng.random((n_sample, m)) * box
    dominated = np.any(np.all(F[None, :, :] <= S[:, None, :], axis=2), axis=1)
    return dominated.mean() * np.prod(box)


def nondominated(F):
    """Boolean mask of non-dominated rows."""
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dom[i] = False
        keep[dom] = False
    return keep
