# Reproducing the studies

All runs use float64 (`JAX_ENABLE_X64=true`). The numerical stack of the
archived runs is pinned in `requirements-reproduce.txt`.

## Periodic studies (CPU)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
python examples/24_multiple_geometry_routes.py
python examples/31_steady_result_to_geometry.py
python examples/32_geometry_gradient_convergence.py --help
python examples/35_regularized_refinement.py --help
python examples/36_closed_loop_refinement.py --help
python tools/run_bound_sweep.py --bound 0.3
```

The example scripts write their records to `output/data/` and overwrite the
archived ones, so run them in a separate clone. `tools/run_bound_sweep.py`
writes to `output/bound_sweep_results.json` unless `--output` is given; with
`--bound 0.3` it also compares the result with the archived optimization in
`output/data/31_steady_result_to_geometry.json`.

## Forward verification

Examples 26-30 run the channel residual, Poiseuille, cavity and DFG 2D-1
cases. `tools/run_cylinder_verification_captured.py` runs a configuration
from `configs/forward_verification/` and stores the result with its
environment. The finite-element comparisons (`tools/dolfinx_dfg2d1.py`,
`tools/fe_normal_constraint.py`) need DOLFINx, PETSc and MPI, which are not
installed by `pip install -e .`.

## Three-body design and fixed-design re-evaluations (GPU)

These runs need an NVIDIA GPU. The guard scripts set float64,
`XLA_FLAGS=--xla_backend_optimization_level=0` and material arrays as runtime
arguments, check that the source tree is clean, and stop a run on high GPU
temperature, low memory or low disk space. The output directory must be new
and outside the repository.

```bash
python tools/guard_multibody_wake.py --mode forward \
  --config configs/multibody_wake/medium_initial.json \
  --output-dir /absolute/path/to/new-run --timeout-seconds 1800
```

The optimization was run with `--mode optimize` and
`configs/multibody_wake/coarse_optimization.json`, then continued from its
result with `configs/multibody_wake/continuation_01.json`. The records are in
`runs/multibody_wake_diagnostics/optimization_v1/` and
`runs/multibody_wake_continuation_20260921/`. The fixed-design
re-evaluations of the final design use the `converged_*.json` configurations.

## Penalty, time-step, DFG 2D-2 and wall-offset checks (GPU)

| Study | Configurations | Guard | Records |
| --- | --- | --- | --- |
| penalty and interface width (A, B, C) | `configs/multibody_penalty/` | `tools/guard_multibody_penalty.py` | `runs/multibody_penalty_sensitivity_20260926/` |
| half time step (A, C) | `configs/multibody_penalty_timecheck/` | `tools/guard_multibody_penalty.py` | `runs/multibody_review_checks_20260926/` |
| DFG 2D-2 at P, A, B, C | `configs/dfg2d2_pairing/` | `tools/guard_dfg2d2.py` | `runs/multibody_review_checks_20260926/` |
| DFG 2D-2 with wall offset | `configs/wall_shift_dfg/` | `tools/guard_wall_shift_dfg.py` | `runs/wall_shift_stage1_20260927/` |
| DFG 2D-1 with wall offset | `configs/dfg2d1_wall_shift/` | `tools/guard_dfg2d1_wall_shift.py` | `runs/dfg2d1_wall_shift_20260928/` |
| three-body designs with wall offset | `configs/multibody_offset/` | `tools/guard_multibody_offset.py` | `runs/multibody_offset_20260928/` |

The planar-shear calibration of the wall offset is in
`tools/quintic_wall_calibration.py` and `jax_cad/differentiable/wall_shift.py`.
These runs wrote their raw histories and fields to an external disk, given in
`tools/multibody_penalty_config.py` and `tools/multibody_penalty_storage.py`;
change that path before running them. The compact records in `runs/` keep the
values derived from the raw files and the SHA-256 checksums of those files.
`tools/archive_*.py --check --output <record directory>` checks a compact
record against its `SHA256SUMS`.
