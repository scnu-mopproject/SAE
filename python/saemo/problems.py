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


PROBLEMS = {
    "SparseZDT1": SparseZDT1, "SparseZDT2": SparseZDT2,
    "SMOP1": SMOP1, "SMOP2": SMOP2, "SMOP3": SMOP3, "SMOP4": SMOP4, "SMOP5": SMOP5,
    "SparsePO": SparsePO,
}
