"""Test problems.

Includes *sparse* bi-objective problems whose Pareto set is sparse (most
decision variables are zero at optimality) and whose Pareto front is known
analytically, so IGD is exact and the whole pipeline is verifiable.

NOTE ON EXACT BENCHMARKS
------------------------
The ``SparseZDT*`` problems below are self-contained and analytically exact.
They are meant for validating the pipeline and demonstrating the
sparsity-preserving behaviour of the DST operator. They are NOT the PlatEMO
SMOP1-8 / Sparse_PO problems. To reproduce the paper's exact numbers, share
the PlatEMO problem ``.m`` files and they can be ported verbatim into this
same ``Problem`` interface.
"""
from __future__ import annotations

import numpy as np


class Problem:
    """Base class. Subclasses set n_var, n_obj, xl, xu and implement _evaluate."""

    def __init__(self, n_var, n_obj, xl, xu):
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.xl = np.broadcast_to(np.asarray(xl, float), (n_var,)).copy()
        self.xu = np.broadcast_to(np.asarray(xu, float), (n_var,)).copy()

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return self._evaluate(np.clip(X, self.xl, self.xu))

    def _evaluate(self, X):                       # pragma: no cover
        raise NotImplementedError

    def pareto_front(self, n=200):                # for IGD reference
        raise NotImplementedError

    def pareto_sparsity(self):
        """Fraction of decision variables that are zero on the Pareto set."""
        return None


class _SparseZDT(Problem):
    """Sparse bi-objective family: Pareto set has only variable 0 non-zero;
    all remaining variables must be 0. So the true sparsity is (D-1)/D."""

    def __init__(self, n_var=100):
        super().__init__(n_var, 2, xl=0.0, xu=1.0)

    def _g(self, X):
        # zero iff x_i == 0 for every i >= 1  ->  drives the tail to sparse zeros
        return X[:, 1:].sum(axis=1) * (9.0 / (self.n_var - 1))

    def pareto_sparsity(self):
        return (self.n_var - 1) / self.n_var


class SparseZDT1(_SparseZDT):
    """Convex PF: f2 = 1 - sqrt(f1), Pareto set = {x0 in [0,1], rest = 0}."""

    def _evaluate(self, X):
        g = self._g(X)
        f1 = X[:, 0]
        f2 = (1 + g) * (1 - np.sqrt(f1 / (1 + g)))
        return np.column_stack([f1, f2])

    def pareto_front(self, n=200):
        f1 = np.linspace(0, 1, n)
        return np.column_stack([f1, 1 - np.sqrt(f1)])


class SparseZDT2(_SparseZDT):
    """Concave PF: f2 = 1 - f1^2, Pareto set = {x0 in [0,1], rest = 0}."""

    def _evaluate(self, X):
        g = self._g(X)
        f1 = X[:, 0]
        f2 = (1 + g) * (1 - (f1 / (1 + g)) ** 2)
        return np.column_stack([f1, f2])

    def pareto_front(self, n=200):
        f1 = np.linspace(0, 1, n)
        return np.column_stack([f1, 1 - f1 ** 2])


# =========================================================================== #
# SMOP1-8 (Tian et al., IEEE TEVC 2020) -- faithful port of PlatEMO .m files.
#   position vars x_1..x_{M-1} in [0,1];  distance vars in [-1,2].
#   K = ceil(theta*(D-M+1)) "relevant" distance vars (optimum at pi/3),
#   the remaining distance vars are 0 at the Pareto set  ->  sparse.
# =========================================================================== #
def _g1(x, t):
    return (x - t) ** 2


def _g2(x, t):
    return 2 * (x - t) ** 2 + np.sin(2 * np.pi * (x - t)) ** 2


def _g3(x, t):
    return 4 - (x - t) - 4.0 / np.exp(100 * (x - t) ** 2)


class _SMOP(Problem):
    """Base for the SMOP suite. Subclasses implement ``_g_of(dist)`` and set
    ``shape`` in {"linear", "concave"}."""
    shape = "linear"

    def __init__(self, n_var=100, n_obj=2, theta=0.1):
        M = n_obj
        xl = np.concatenate([np.zeros(M - 1), -np.ones(n_var - M + 1)])
        xu = np.concatenate([np.ones(M - 1), 2 * np.ones(n_var - M + 1)])
        super().__init__(n_var, M, xl, xu)
        self.theta = theta
        self.K = int(np.ceil(theta * (n_var - M + 1)))

    # --- objective assembly shared by every SMOP ---
    def _shape_terms(self, pos):
        n = pos.shape[0]
        ones = np.ones((n, 1))
        if self.shape == "linear":
            A = np.cumprod(np.hstack([ones, pos]), axis=1)[:, ::-1]
            B = np.hstack([ones, 1 - pos[:, ::-1]])
        else:  # concave
            A = np.cumprod(np.hstack([ones, 1 - np.cos(pos * np.pi / 2)]), axis=1)[:, ::-1]
            B = np.hstack([ones, 1 - np.sin(pos[:, ::-1] * np.pi / 2)])
        return A * B

    def _evaluate(self, X):
        M, D = self.n_obj, self.n_var
        pos = X[:, :M - 1]
        dist = X[:, M - 1:]
        g = self._g_of(dist)
        pop = (1 + g / (D - M + 1))[:, None] * self._shape_terms(pos)
        return pop

    def _g_of(self, dist):                       # pragma: no cover
        raise NotImplementedError

    def pareto_sparsity(self):
        return (self.n_var - self.n_obj + 1 - self.K) / (self.n_var - self.n_obj + 1)

    def pareto_front(self, n=300):
        if self.n_obj != 2:
            raise NotImplementedError("PF reference implemented for M=2")
        if self.shape == "linear":
            f1 = np.linspace(0, 1, n)
            return np.column_stack([f1, 1 - f1])
        x1 = np.linspace(0, 1, n)
        return np.column_stack([1 - np.cos(x1 * np.pi / 2), 1 - np.sin(x1 * np.pi / 2)])


class SMOP1(_SMOP):
    shape = "linear"
    def _g_of(self, dist):
        K = self.K
        return _g1(dist[:, :K], np.pi / 3).sum(1) + _g2(dist[:, K:], 0).sum(1)


class SMOP2(_SMOP):
    shape = "linear"
    def _g_of(self, dist):
        K = self.K
        return _g2(dist[:, :K], np.pi / 3).sum(1) + _g3(dist[:, K:], 0).sum(1)


class SMOP3(_SMOP):
    shape = "linear"
    def _g_of(self, dist):
        K = self.K
        g = _g1(dist[:, :K], np.pi / 3).sum(1)
        rest = dist[:, K:]
        n_rest = rest.shape[1]
        for i in range(int(np.ceil(n_rest / 10))):
            chunk = rest[:, i * 10:(i + 1) * 10]
            temp = 50 - _g1(chunk, 0).sum(1)
            m = temp < 50
            g[m] += temp[m]
        return g


class SMOP4(_SMOP):
    shape = "concave"
    def _g_of(self, dist):
        M, D = self.n_obj, self.n_var
        K = self.K
        g = np.sort(_g3(dist, 0), axis=1)
        return g[:, :D - M - K + 1].sum(1)


class SMOP5(_SMOP):
    shape = "concave"
    def _g_of(self, dist):
        K = self.K
        gg = _g1(dist, np.pi / 3) * _g2(dist, 0)
        return gg.sum(1) + np.abs(K - np.sum(dist != 0, axis=1))


# =========================================================================== #
# Sparse_PO -- portfolio optimization (Tian et al., IEEE TCyb 2021).
#   Faithful port of PlatEMO Sparse_PO.m + Dataset_PO.mat.
#   f1 = w' Risk w  (risk),  f2 = 1 - sum(w . Yield)  (1 - return),
#   with weights normalized so that sum|w| <= 1. Real-world -> HV only.
# =========================================================================== #
class SparsePO(Problem):
    def __init__(self, dataNo=1, data_path=None):
        import os
        import scipy.io as sio
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "data", "Dataset_PO.mat")
        ds = sio.loadmat(data_path)["Dataset"][0, 0]
        Data = ds[{1: "data1000", 2: "data5000"}[dataNo]].astype(float)
        self.Yield = np.log(Data[:, 1:]) - np.log(Data[:, :-1])     # D x (T-1)
        self.Risk = np.cov(self.Yield)                              # D x D
        D = self.Yield.shape[0]
        super().__init__(D, 2, xl=-1.0, xu=1.0)

    def _evaluate(self, X):
        s = np.maximum(np.abs(X).sum(1, keepdims=True), 1.0)
        W = X / s
        f1 = np.einsum("ij,jk,ik->i", W, self.Risk, W)             # w' Risk w
        f2 = 1.0 - (W @ self.Yield).sum(1)                          # 1 - return
        return np.column_stack([f1, f2])

    def pareto_front(self, n=300):
        return None            # real-world problem: no analytic PF (use HV)


# =========================================================================== #
# LSMOP1-9 (Cheng et al., IEEE TCyb 2017) -- faithful port of PlatEMO.
#   D = 100*M by default; position vars x_1..x_{M-1} in [0,1], distance vars
#   in [0,10]; chaotic variable grouping; shapes: linear (1-4), convex (5-8),
#   disconnected (9).
# =========================================================================== #
def _ls_sphere(x):
    return (x ** 2).sum(1)


def _ls_griewank(x):
    n = x.shape[1]
    return (x ** 2).sum(1) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))), axis=1) + 1


def _ls_schwefel(x):
    return np.abs(x).max(1) if x.shape[1] else np.zeros(len(x))


def _ls_rastrigin(x):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x) + 10).sum(1)


def _ls_rosenbrock(x):
    if x.shape[1] < 2:
        return np.zeros(len(x))
    return (100 * (x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1) ** 2).sum(1)


def _ls_ackley(x):
    n = x.shape[1]
    return (20 - 20 * np.exp(-0.2 * np.sqrt((x ** 2).sum(1) / n))
            - np.exp(np.cos(2 * np.pi * x).sum(1) / n) + np.e)


class _LSMOP(Problem):
    shape = "linear"                     # 'linear' | 'convex' | 'disconnected'
    gfun = (_ls_sphere, _ls_sphere)      # (odd-group fn, even-group fn)

    def __init__(self, n_var=None, n_obj=2, nk=5):
        M = n_obj
        D = 100 * M if n_var is None else n_var
        xl = np.zeros(D)
        xu = np.concatenate([np.ones(M - 1), 10 * np.ones(D - M + 1)])
        super().__init__(D, M, xl, xu)
        self.nk = nk
        c = [3.8 * 0.1 * (1 - 0.1)]
        for _ in range(M - 1):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = np.array(c)
        self.sublen = np.floor(c / c.sum() * (D - M + 1) / nk).astype(int)
        self.len = np.concatenate([[0], np.cumsum(self.sublen * nk)]).astype(int)

    def _link(self, X):
        M, D = self.n_obj, self.n_var
        Xd = X.copy()
        idx = np.arange(M, D + 1) / D
        factor = 1 + idx if self.shape == "linear" else 1 + np.cos(idx * np.pi / 2)
        Xd[:, M - 1:] = factor * Xd[:, M - 1:] - Xd[:, 0:1] * 10
        return Xd

    def _G(self, Xd):
        M, N = self.n_obj, len(Xd)
        G = np.zeros((N, M))
        A, B = self.gfun
        for g in range(M):
            fn = A if g % 2 == 0 else B
            acc = np.zeros(N)
            for jj in range(self.nk):
                s = self.len[g] + (M - 1) + jj * self.sublen[g]
                acc = acc + fn(Xd[:, s:s + self.sublen[g]])
            G[:, g] = acc
        return G

    def _evaluate(self, X):
        M, N = self.n_obj, len(X)
        Xd = self._link(X)
        G = self._G(Xd)
        pos = X[:, :M - 1]
        ones = np.ones((N, 1))
        if self.shape == "disconnected":
            Gs = 1 + (G / self.sublen[None, :] / self.nk).sum(1)
            obj = np.zeros((N, M))
            obj[:, :M - 1] = pos
            inner = (pos / (1 + Gs[:, None]) * (1 + np.sin(3 * np.pi * pos))).sum(1)
            obj[:, M - 1] = (1 + Gs) * (M - inner)
            return obj
        G = G / self.sublen[None, :] / self.nk
        if self.shape == "linear":
            A = np.cumprod(np.hstack([ones, pos]), axis=1)[:, ::-1]
            Bm = np.hstack([ones, 1 - pos[:, ::-1]])
            return (1 + G) * A * Bm
        Gc = 1 + G + np.hstack([G[:, 1:], np.zeros((N, 1))])
        A = np.cumprod(np.hstack([ones, np.cos(pos * np.pi / 2)]), axis=1)[:, ::-1]
        Bm = np.hstack([ones, np.sin(pos[:, ::-1] * np.pi / 2)])
        return Gc * A * Bm

    def pareto_front(self, n=300):
        M = self.n_obj
        if self.shape == "linear":
            if M == 2:
                f1 = np.linspace(0, 1, n)
                return np.column_stack([f1, 1 - f1])
            return core_uniform(n, M)
        if self.shape == "convex":
            if M == 2:
                w = np.linspace(0, 1, n)
                P = np.column_stack([w, 1 - w])
                return P / np.linalg.norm(P, axis=1, keepdims=True)
            R = core_uniform(n, M)
            return R / np.linalg.norm(R, axis=1, keepdims=True)
        # disconnected (LSMOP9), M=2
        interval = [0, 0.251412, 0.631627, 0.859401]
        med = (interval[1] - interval[0]) / ((interval[3] - interval[2]) + (interval[1] - interval[0]))
        X = np.linspace(0, 1, n)
        X = np.where(X <= med, X * (interval[1] - interval[0]) / med + interval[0],
                     (X - med) * (interval[3] - interval[2]) / (1 - med) + interval[2])
        f2 = 2 * (2 - X / 2 * (1 + np.sin(3 * np.pi * X)))
        R = np.column_stack([X, f2])
        return R[_nd_mask(R)]


def core_uniform(n, m):
    from .core import uniform_points
    return uniform_points(n, m)[0]


def _nd_mask(F):
    from .metrics import nondominated
    return nondominated(F)


class LSMOP1(_LSMOP):
    shape = "linear"; gfun = (_ls_sphere, _ls_sphere)


class LSMOP2(_LSMOP):
    shape = "linear"; gfun = (_ls_griewank, _ls_schwefel)


class LSMOP3(_LSMOP):
    shape = "linear"; gfun = (_ls_rastrigin, _ls_rosenbrock)


class LSMOP4(_LSMOP):
    shape = "linear"; gfun = (_ls_ackley, _ls_griewank)


class LSMOP5(_LSMOP):
    shape = "convex"; gfun = (_ls_sphere, _ls_sphere)


class LSMOP6(_LSMOP):
    shape = "convex"; gfun = (_ls_rosenbrock, _ls_schwefel)


class LSMOP7(_LSMOP):
    shape = "convex"; gfun = (_ls_ackley, _ls_rosenbrock)


class LSMOP8(_LSMOP):
    shape = "convex"; gfun = (_ls_griewank, _ls_sphere)


class LSMOP9(_LSMOP):
    shape = "disconnected"; gfun = (_ls_sphere, _ls_ackley)


PROBLEMS = {
    "SparseZDT1": SparseZDT1, "SparseZDT2": SparseZDT2,
    "SMOP1": SMOP1, "SMOP2": SMOP2, "SMOP3": SMOP3, "SMOP4": SMOP4, "SMOP5": SMOP5,
    "SparsePO": SparsePO,
    "LSMOP1": LSMOP1, "LSMOP2": LSMOP2, "LSMOP3": LSMOP3, "LSMOP4": LSMOP4,
    "LSMOP5": LSMOP5, "LSMOP6": LSMOP6, "LSMOP7": LSMOP7, "LSMOP8": LSMOP8,
    "LSMOP9": LSMOP9,
}
