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


PROBLEMS = {"SparseZDT1": SparseZDT1, "SparseZDT2": SparseZDT2}
