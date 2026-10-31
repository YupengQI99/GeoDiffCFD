# GeoDiffCFD

GeoDiffCFD is a JAX code for differentiable two-dimensional flow design with
Brinkman penalization. Geometry parameters are mapped to a material field, the
flow is computed on a staggered grid, and the gradient of a flow objective with
respect to the geometry parameters is obtained by automatic or implicit
differentiation.

This repository contains the code and the numerical records for the paper

> Y. Qi, Y. Wu, W. Lou, P. Yan, G. Yang, X. Chu. GeoDiffCFD: A differentiable
> Brinkman-penalization framework for two-dimensional flow design.

## What is included

- Geometry routes: analytic and sampled signed-distance fields, density
  variables, and area-preserving Fourier boundaries.
- Periodic Stokes-Brinkman flow on a staggered grid with mobility-weighted
  pressure projection, and non-periodic Navier-Stokes-Brinkman flow in a
  channel.
- Reverse differentiation through a finite number of iterations or a transient
  trajectory, and implicit differentiation of converged periodic states.
- Bound-constrained shape optimization, including the three-body wake design.

## Installation

Python 3.11 is used for the archived runs.

```bash
git clone https://github.com/YupengQI99/GeoDiffCFD.git
cd GeoDiffCFD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -c requirements-reproduce.txt
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
```

The Python package is imported as `jax_cad`, the name it had during
development.

## Objective and gradient in a few lines

```python
import jax
from jax_cad.unified import DifferentiableProgram, FourierObstacleBenchmark

jax.config.update("jax_enable_x64", True)
benchmark = FourierObstacleBenchmark()
solver, diagnostics = benchmark.implicit_solver()
program = DifferentiableProgram(
    benchmark.design_map(), solver, benchmark.objective
)
value, gradient = jax.jit(program.value_and_grad)(benchmark.zero_coefficients())
print(value, gradient)
```

## Layout

| Path | Content |
| --- | --- |
| `jax_cad/` | solvers, geometry maps, differentiation and optimization |
| `examples/` | scripts 22-36: periodic studies and forward verification |
| `tools/` | drivers for the GPU studies, the coefficient-bound sweep, and record checks |
| `configs/` | run configurations used by the drivers |
| `output/data/` | records of the periodic studies written by `examples/` |
| `runs/` | records of the forward verification and the GPU runs |
| `tests/` | unit and regression tests |

## Data behind the tables and figures

Numbers refer to the preprint. Each record stores its configuration, the
software versions and the revision of the development repository (not
public) that produced it. The code here is taken from that repository at
revision `203efc0`, the last revision used for the reported runs.

| Item | Records | Produced by |
| --- | --- | --- |
| Fig. 2, Tables 2-3 (Poiseuille, lid-driven cavity) | `runs/poiseuille_forward_verification/`, `runs/cavity_re100_forward_verification/` | `examples/28_poiseuille_verification.py`, `examples/29_cavity_re100.py` |
| Fig. 3, Table 4 (finite horizon) | `output/data/31_steady_result_to_geometry.json`, `output/data/34_field_snapshots.json` | `examples/31_steady_result_to_geometry.py`, `examples/34_field_snapshots.py` |
| Table 5 (geometry routes) | `output/data/24_multiple_geometry_routes.json` | `examples/24_multiple_geometry_routes.py` |
| Table 6 (implicit and unrolled backends) | `output/data/31_steady_result_to_geometry.json` | `examples/31_steady_result_to_geometry.py` |
| Fig. 4, Tables 7-8 (regularized refinement) | `output/data/35_regularized_refinement*.json` | `examples/35_regularized_refinement.py` |
| Fig. 5 (interface resolution) | `output/data/34_field_snapshots.json` | `examples/34_field_snapshots.py` |
| Fig. 6, Table 9 (gradient convergence) | `output/data/32_geometry_gradient_convergence*.json` | `examples/32_geometry_gradient_convergence.py` |
| Table 10 (DFG 2D-1 cylinder) | `runs/dfg_cylinder_2d1_forward_verification/19cd3ae/` | `examples/30_cylinder_dfg2d1.py`, `tools/run_cylinder_verification_captured.py` |
| Table 11 (DFG 2D-2 and wall offset) | `runs/multibody_review_checks_20260926/dfg2d2.csv`, `runs/wall_shift_stage1_20260927/results.csv`, `runs/dfg2d1_wall_shift_20260928/results.csv` | `tools/run_dfg2d2.py`, `tools/run_wall_shift_dfg.py`, `tools/run_dfg2d1_wall_shift.py` |
| Tables 12, 14, Figs. 7-8 (periodic design) | `output/data/31_steady_result_to_geometry.json`, `output/data/25_result_to_geometry_optimization.json`, `output/data/34_field_snapshots.json` | `examples/31_steady_result_to_geometry.py`, `examples/25_result_to_geometry_optimization.py`, `examples/34_field_snapshots.py` |
| Table 13 (coefficient bound) | `runs/periodic_bound_sweep/results.json` | `tools/run_bound_sweep.py` |
| Table 15 (nested-grid design) | `output/data/36_closed_loop_refinement*.json` | `examples/36_closed_loop_refinement.py` |
| Fig. 9, Table 16 design-grid row (three-body optimization) | `runs/multibody_wake_diagnostics/optimization_v1/`, `runs/multibody_wake_continuation_20260921/` | `tools/guard_multibody_wake.py`, `tools/multibody_wake_continuation.py` |
| Table 16 other rows, Fig. 10 (fixed-design re-evaluation) | `runs/multibody_wake_final_20260923/` (final design), `runs/multibody_wake_diagnostics/*_initial_v1/` (initial design) | `tools/guard_multibody_wake.py` with `configs/multibody_wake/` |
| Table 17 (penalty and wall-offset sensitivity) | `runs/multibody_penalty_sensitivity_20260926/objectives.csv`, `runs/multibody_review_checks_20260926/timecheck.csv`, `runs/multibody_offset_20260928/pairs.csv` | `tools/run_multibody_penalty.py`, `tools/run_multibody_offset.py` |
| Table 18 (flow inside the bodies) | `runs/multibody_penalty_sensitivity_20260926/summary.json`, `runs/multibody_offset_20260928/diagnostics.csv` | `tools/run_multibody_penalty.py`, `tools/run_multibody_offset.py` |
| Figs. 11-12 (three-body flow fields) | `fields.npz` in `runs/multibody_wake_diagnostics/optimization_v1/evaluation-000/` and `runs/multibody_wake_continuation_20260921/continuation_01_retry/evaluation-005/` | `tools/guard_multibody_wake.py` |
| Table A1 (cylinder with normal constraints) | `runs/mac_normal_response_diagnostics/` | `tools/run_normal_cylinder.py`, `tools/fe_normal_constraint.py` |

`runs/derived/` holds two small summaries quoted in the text: the DFG 2D-2
drag and Strouhal number on two grids with the design-model parameters, and
the explicit-scheme stability index of the three-body runs. Each entry names
the history file it was computed from.

The GPU studies stored force histories, but not every flow field: fields of
up to 7.5 MB per evaluation were kept only where a figure needs them. The
summaries list SHA-256 checksums of all raw files they were computed from.

Some records carry fields such as `paper_eligible` or other acceptance flags.
They were set by the run-checking scripts used during development and do not
indicate whether a result is reported in the paper.

## Running the studies

Periodic studies run on a CPU in minutes to hours:

```bash
python examples/24_multiple_geometry_routes.py
python examples/32_geometry_gradient_convergence.py --help
python tools/run_bound_sweep.py --bound 0.3
```

The example scripts overwrite the records in `output/data/`, so run them in a
separate clone if you want to compare with the archived values.

The three-body, DFG 2D-2 and wall-offset runs need an NVIDIA GPU. They use
float64 and `XLA_FLAGS=--xla_backend_optimization_level=0`, which the guard
scripts set. A forward run of the initial three-body design is started with

```bash
python tools/guard_multibody_wake.py --mode forward \
  --config configs/multibody_wake/medium_initial.json \
  --output-dir /absolute/path/to/new-run --timeout-seconds 1800
```

See `docs/REPRODUCING.md` for the other drivers.

## Tests

```bash
python tools/test_release.py
python tools/update_data_checksums.py --check
```

The first command runs each test file in its own process. It does not run the
optimization studies, and GPU tests are skipped on machines without a GPU.

## License

MIT, see [LICENSE](LICENSE). If you use the code, please cite the paper above
(see [CITATION.cff](CITATION.cff)).
