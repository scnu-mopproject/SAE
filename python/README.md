# saemo — Python reimplementation of the SAE/APF sparse MOEA framework

A compact, dependency-light (numpy / scipy / matplotlib) framework that
reproduces the proposed **APF-NSGA-II** method (the MATLAB `APFNSGAII.m` +
`DSTOperator.m`) in Python, together with baseline algorithms and the plotting
you need to compare results — no MATLAB / PlatEMO required.

## Install & run
```bash
pip install -r requirements.txt
python demo.py            # runs a comparison and writes ./figures/*.png
```

## What is implemented (faithful)
| Component | Source | Status |
|---|---|---|
| `apf_nsga2` (proposed) | `APFNSGAII.m` + `DSTOperator.m` | line-by-line port |
| `nsga2` | standard NSGA-II (real-coded) | ✅ |
| `sparseea` | SparseEA (Tian et al., TEVC 2020) | ✅ |
| IGD / HV (2-D exact, 3-D+ Monte-Carlo) | — | ✅ |
| SBX + polynomial mutation | PlatEMO `OperatorGA` defaults | ✅ |
| `SparseZDT1/2` sparse test problems | analytic PF (this repo) | ✅ |

### The proposed operator
`saemo/algorithms.py::_dst_operator` ports `DSTOperator.m` exactly, including
the amplitude term. Two modes:
- `amplitude="mod"` — matches the MATLAB code (`3*0.5^(i-1).*mod(x,1)`).
- `amplitude="abs"` — the continuous, scale-covariant alternative
  (`3*0.5^(i-1).*|x|`) with the **same zero-preservation** property; recommended
  for general sparse problems where nonzero variables can be arbitrary reals.

Switch via `apf_nsga2(problem, amplitude="abs")`.

## What is NOT included (and why)
`MSKEA`, `MGCEA`, `BLIGEA`, and the exact PlatEMO `SMOP1-8` / `Sparse_PO`
problems are **not** reimplemented. Approximate reimplementations would be
unfair baselines / wrong benchmarks and would invalidate any comparison. To
add them faithfully, drop the corresponding PlatEMO `.m` files in and port them
into the `Problem` / algorithm interface (the MATLAB reads cleanly into this
structure). Each new problem only needs `_evaluate(X)->F`, `xl`, `xu`, and
optionally `pareto_front(n)` for IGD.

## Extending
```python
from saemo.problems import Problem
class MyProblem(Problem):
    def __init__(self, D): super().__init__(D, n_obj=2, xl=0.0, xu=1.0)
    def _evaluate(self, X): ...      # return F, shape (len(X), n_obj)
    def pareto_front(self, n=200): ...
```

## Key diagnostic finding
On a sparse problem, `apf_nsga2` embedding *plain* NSGA-II attains the best
IGD/HV but does **not** produce sparse decision vectors, because the `mod`/`abs`
operator only *preserves* zeros that the embedded algorithm already created —
plain NSGA-II creates none. To exhibit sparsity preservation the framework must
embed a sparsity-producing optimizer (sparse-encoded NSGA-II, MSKEA, …). This
is exactly the mechanism argument for the paper.
