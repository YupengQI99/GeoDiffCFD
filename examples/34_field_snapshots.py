"""Field snapshots that illustrate two results of the manuscript.

1. Horizon: the streamwise velocity of the reference circle after 100 explicit
   steps (``t = 0.15``, the horizon of the previous version) next to the
   converged field, with their difference.
2. Interface resolution: the geometry-sensitivity density and the converged
   velocity in a window around the obstacle on the three grids of the
   convergence study at the reference point ``(epsilon/h, alpha_max) = (1.5, 2400)``.
3. Sensitivity fields of the closed loop: for the initial circle and the
   optimized shape of record 31, the material cotangent ``g_alpha = dJ/dalpha``
   at the converged state (through the implicit solver) and the level-set
   sensitivity density ``-alpha_max H'_eps(phi) g_alpha`` of Eq. (sdf_pullback),
   which is the per-cell contribution to every native geometry derivative.

The record stores cell-centred arrays and full provenance.
Run from the repository root with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import simulate
from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import FourierObstacleBenchmark
from jax_cad.unified.materials import compact_delta

HORIZON_STEPS = 100
LEVELS = {"coarse": (32, 20), "medium": (64, 40), "fine": (128, 80)}


def _sensitivity(bench: FourierObstacleBenchmark, alpha, level_set):
    """Cotangent of the physical objective with respect to alpha and its level-set density."""

    solver, _ = bench.implicit_solver()

    def objective_of_alpha(a):
        return bench.physical_loss(solver(_Fields(a)))

    g_alpha = jax.grad(objective_of_alpha)(alpha)
    density = -bench.alpha_max * compact_delta(level_set, bench.epsilon) * g_alpha
    return np.asarray(g_alpha), np.asarray(density)


class _Fields:
    """Minimal stand-in exposing only ``momentum_resistance`` to the solver."""

    def __init__(self, alpha):
        self.momentum_resistance = alpha


def _cell_arrays(bench: FourierObstacleBenchmark, state, fields) -> dict[str, object]:
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    g_alpha, density = _sensitivity(bench, fields.momentum_resistance, fields.level_set)
    return {
        "pressure": np.asarray(state.p).tolist(),
        "objective_cotangent_alpha": g_alpha.tolist(),
        "level_set_sensitivity_density": density.tolist(),
        "sensitivity_density_definition": "-alpha_max * H_eps'(phi) * dJ_flow/dalpha per cell; cells with |phi| >= eps carry exactly zero",
        "nx": bench.nx,
        "ny": bench.ny,
        "dx": bench.grid.dx,
        "dy": bench.grid.dy,
        "alpha": np.asarray(fields.momentum_resistance).tolist(),
        "level_set": np.asarray(fields.level_set).tolist(),
        "u_cell": (0.5 * (u + np.roll(u, 1, axis=0))).tolist(),
        "v_cell": (0.5 * (v + np.roll(v, 1, axis=1))).tolist(),
        "mean_streamwise_velocity": float(u.mean()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "output" / "data" / "34_field_snapshots.json"
    started = provenance_record()["utc"]

    # ---------------------------------------------------------------- horizon
    bench = FourierObstacleBenchmark()
    design_map = bench.design_map()
    solver, primal = bench.implicit_solver()
    fields = design_map(bench.zero_coefficients())
    record31 = json.loads((root / "output" / "data" / "31_steady_result_to_geometry.json").read_text(encoding="utf-8"))
    final_coefficients = jnp.asarray(record31["optimization"]["final_coefficients"], dtype=jnp.float64)
    final_fields = design_map(final_coefficients)
    final_state, final_diag = primal(final_fields.momentum_resistance)
    transient = block_tree(
        simulate(bench.initial_state(), fields.momentum_resistance, bench.grid, bench.params(), n_steps=HORIZON_STEPS)
    )
    steady, diagnostics = primal(fields.momentum_resistance)
    horizon = {
        "steps": HORIZON_STEPS,
        "final_time": HORIZON_STEPS * bench.dt,
        "transient": _cell_arrays(bench, transient, fields),
        "converged": _cell_arrays(bench, steady, fields),
        "converged_primal_residual": float(diagnostics.residual_norm),
        "mean_velocity_ratio_transient_over_converged": float(jnp.mean(transient.u) / jnp.mean(steady.u)),
    }

    # ----------------------------------------------------- interface resolution
    resolution = {}
    for level, (nx, ny) in LEVELS.items():
        b = FourierObstacleBenchmark(nx=nx, ny=ny)
        fields_l = b.design_map()(b.zero_coefficients())
        _, primal_l = b.implicit_solver()
        state_l, diag_l = primal_l(fields_l.momentum_resistance)
        entry = _cell_arrays(b, state_l, fields_l)
        entry["primal_residual"] = float(diag_l.residual_norm)
        entry["epsilon"] = b.epsilon
        entry["cells_across_interface_band"] = 2.0 * b.epsilon / b.h
        entry["brinkman_layer_over_h"] = float((b.viscosity / b.alpha_max) ** 0.5 / b.h)
        resolution[level] = entry
        print(f"{level}: mean u {entry['mean_streamwise_velocity']:.8f} residual {entry['primal_residual']:.1e}", flush=True)

    provenance = provenance_record()
    provenance["start_utc"] = started
    provenance["script"] = "examples/34_field_snapshots.py"
    closed_loop = {
        "source_record": "output/data/31_steady_result_to_geometry.json",
        "final_coefficients": np.asarray(final_coefficients).tolist(),
        "initial": horizon["converged"],
        "final": {**_cell_arrays(bench, final_state, final_fields), "primal_residual": float(final_diag.residual_norm)},
    }
    record = {
        "claim": (
            "Field snapshots: finite-horizon versus converged velocity, interface-band resolution across three grids, "
            "and the material cotangent and level-set sensitivity density of the closed loop."
        ),
        "configuration": bench.to_dict(),
        "horizon": horizon,
        "interface_resolution": resolution,
        "closed_loop_sensitivity": closed_loop,
        "provenance": provenance,
    }
    write_json(path, record)
    print(json.dumps({"mean_velocity_ratio_transient_over_converged": horizon["mean_velocity_ratio_transient_over_converged"]}, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
