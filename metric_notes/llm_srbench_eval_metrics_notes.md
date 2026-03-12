# Metric Requirements: LLM-SRBench

Source: Paper Section 4, https://arxiv.org/abs/2504.10415
Code: bench/pipelines.py (compute_output_base_metrics)

## Why Custom Metrics Are Needed

Standard text metrics (BLEU, ROUGE, exact_match) are inadequate for symbolic equations
because mathematically equivalent expressions have different string forms:
- `x^2 + 2*x + 1` vs `(x + 1)^2`
- `sin(x)^2 + cos(x)^2` vs `1`
- `a*x + b*x` vs `(a+b)*x`

## Evaluation Approach

The benchmark evaluates equations by **executing** them against held-out test data:

1. Parse the model's predicted equation into an executable form (sympy or lambda)
2. Evaluate on in-distribution (ID) test data points
3. Evaluate on out-of-distribution (OOD) test data points
4. Compute numerical metrics

## Metrics (from bench/pipelines.py)

| Metric | Formula | Notes |
|--------|---------|-------|
| **NMSE** | `mean((y - y_pred)^2) / var(y)` | Primary metric; normalized MSE |
| **R²** | `1 - sum((y - y_pred)^2) / sum((y - y_mean)^2)` | Coefficient of determination |
| **MSE** | `mean((y - y_pred)^2)` | Raw mean squared error |
| **Kendall's Tau** | Rank correlation between y and y_pred | Monotonicity measure |
| **MAPE** | Mean absolute percentage error | Relative error measure |

## Test Data

Both ID and OOD test sets are in the HDF5 file (`lsr_bench_data.hdf5`):
- LSR-Transform: `/{name}/test` (ID), `/{name}/ood_test` (OOD)
- LSR-Synth: `/lsr_synth/{domain}/{name}/id_test_data` (ID), `ood_test_data` (OOD)

Column 0 is always the output variable; columns 1+ are inputs.

## Implementation Notes

A custom HELM metric would need to:
1. Parse predicted equation string to a callable (e.g., via `sympy.sympify` + `lambdify`)
2. Handle parse failures gracefully (score = 0)
3. Load test data from HDF5
4. Compute NMSE/R² on both ID and OOD test sets
5. Handle numerical issues (NaN, Inf predictions)

### Equation Parsing

The original benchmark uses three formats:
- `expression`: Symbolic string (e.g., `"sin(x) + x**2"`)
- `program_format`: Python function code
- `lambda_format`: Callable Python function

For HELM, the model outputs a symbolic string which must be parsed via sympy.

### Parameter Optimization

Some equations contain free parameters (constants). The original benchmark uses
scipy.optimize.minimize with BFGS to find optimal parameter values before scoring.
A simplified evaluation could skip parameter optimization and only evaluate
equations with fixed numerical constants.

## Baseline Performance

From the paper (Table 2):
- Best LLM method (LaSR with GPT-4): ~15% R² > 0.99 on LSR-Transform
- Best overall (SGA with GPT-4): ~30% R² > 0.99 on LSR-Transform
- Classical SR methods (PySR): ~50% R² > 0.99 on LSR-Transform
- This is a very challenging benchmark
