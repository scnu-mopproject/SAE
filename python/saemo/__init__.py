"""saemo: a compact Python framework to reproduce and compare the SAE/APF
sparse large-scale multi-objective method against baselines."""
from . import core, metrics, problems, algorithms          # noqa: F401
from .problems import PROBLEMS, Problem                     # noqa: F401
from .algorithms import nsga2, sparseea, apf_nsga2, ALGORITHMS  # noqa: F401
from .metrics import igd, hypervolume, nondominated         # noqa: F401

__all__ = ["core", "metrics", "problems", "algorithms",
           "PROBLEMS", "Problem", "nsga2", "sparseea", "apf_nsga2",
           "ALGORITHMS", "igd", "hypervolume", "nondominated"]
