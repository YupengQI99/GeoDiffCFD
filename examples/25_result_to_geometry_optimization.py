"""Finite-horizon closed loop: transient result -> unrolled gradient -> geometry.

Companion of ``examples/31_steady_result_to_geometry.py``.  The same committed
Fourier-obstacle benchmark, material map, objective and projected-gradient
optimizer are used, but the physical result is the state after a fixed horizon
of 100 explicit steps (``t = 0.15``), differentiated by reverse mode through
the unrolled ``jax.lax.scan``.  This exercises the finite-horizon sensitivity
backend of the manuscript; the record explicitly reports how far the horizon
is from steady state so that the two backends are not confused.

Run from the repository root with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import divergence_norms
from jax_cad.optimization.projected_descent import projected_gradient_descent
from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    FourierObstacleBenchmark,
    directional_taylor_test,
    fourier_boundary_radius,
    fourier_shape_area,
)

N_STEPS = 100
TAYLOR_DIRECTION = (0.30, -0.40, 0.20, 0.10, -0.25, 0.15, 0.12, -0.08)
TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "output" / "data" / "25_result_to_geometry_optimization.json"
    started_utc = provenance_record()["utc"]
    wall_started = time.perf_counter()

    bench = FourierObstacleBenchmark()
    grid = bench.grid
    design_map = bench.design_map()
    solver = bench.transient_solver(N_STEPS)
    program = DifferentiableProgram(design_map, solver, bench.objective)
    value_and_grad = jax.jit(program.value_and_grad)
    value = jax.jit(program.value)
    implicit_solver, primal = bench.implicit_solver()

    coefficients0 = bench.zero_coefficients()
    warm = block_tree(value_and_grad(coefficients0))
    del warm

    direction = jnp.asarray(TAYLOR_DIRECTION, dtype=jnp.float64)
    taylor = directional_taylor_test(program.value, coefficients0, direction, epsilons=TAYLOR_EPSILONS)

    def value_and_grad_host(x):
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g)

    def value_host(x):
        return float(value(jnp.asarray(x)))

    loop_started = time.perf_counter()
    result = projected_gradient_descent(
        value_and_grad_host,
        value_host,
        np.asarray(coefficients0),
        lower=-bench.coefficient_bound,
        upper=bench.coefficient_bound,
        initial_step=bench.initial_step,
        max_iterations=bench.max_iterations,
        relative_gradient_tolerance=bench.relative_gradient_tolerance,
        step_growth=2.0,
    )
    loop_seconds = time.perf_counter() - loop_started
    for item in result.trace:
        state_k = solver(design_map(jnp.asarray(item["coefficients"])))
        item["mean_streamwise_velocity"] = float(jnp.mean(state_k.u))

    final_coefficients = jnp.asarray(result.coefficients)
    final_fields = design_map(final_coefficients)
    final_state = solver(final_fields)
    initial_state = solver(design_map(coefficients0))
    final_value, final_gradient = block_tree(value_and_grad(final_coefficients))
    div_rms, div_max = divergence_norms(final_state, grid)
    initial_area = fourier_shape_area(coefficients0, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    final_area = fourier_shape_area(final_coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    angles = jnp.array([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi], dtype=jnp.float64)
    radii = fourier_boundary_radius(angles, final_coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    steady_initial, _ = primal(design_map(coefficients0).momentum_resistance)
    steady_final, _ = primal(final_fields.momentum_resistance)
    first = result.trace[0]
    accepted = [item for item in result.trace if item["accepted"]]

    provenance = provenance_record()
    provenance["start_utc"] = started_utc
    provenance["wall_seconds"] = time.perf_counter() - wall_started
    provenance["script"] = "examples/25_result_to_geometry_optimization.py"

    record = {
        "claim": (
            "A finite-horizon flow result is reverse-differentiated through the unrolled solver to native "
            "Fourier geometry coefficients and used to update an exactly area-preserving shape."
        ),
        "configuration": {**bench.to_dict(), "n_steps": N_STEPS, "final_time": N_STEPS * bench.dt, "sensitivity_backend": "unrolled reverse mode through jax.lax.scan"},
        "horizon_context": {
            "final_time": N_STEPS * bench.dt,
            "circle_mean_u_at_horizon": float(jnp.mean(initial_state.u)),
            "circle_steady_mean_u": float(jnp.mean(steady_initial.u)),
            "final_shape_mean_u_at_horizon": float(jnp.mean(final_state.u)),
            "final_shape_steady_mean_u": float(jnp.mean(steady_final.u)),
        },
        "derivative_verification": {
            "direction": list(TAYLOR_DIRECTION),
            "direction_normalized": True,
            "epsilons": np.asarray(taylor.epsilons).tolist(),
            "zero_order_errors": np.asarray(taylor.zero_order_error).tolist(),
            "first_order_remainders": np.asarray(taylor.first_order_error).tolist(),
            "zero_order_rates": np.asarray(taylor.zero_order_rate).tolist(),
            "first_order_rates": np.asarray(taylor.first_order_rate).tolist(),
            "directional_derivative": float(taylor.directional_derivative),
        },
        "first_geometry_step": {
            "step_length": first["step_length"],
            "linear_prediction": first["linear_prediction"],
            "realised_change": first["realised_change"],
            "ratio": first["realised_change"] / first["linear_prediction"] if first["linear_prediction"] else None,
        },
        "optimization": {
            "algorithm": "normalized projected steepest descent with box projection and Armijo backtracking",
            "termination": result.termination,
            "converged": result.converged,
            "iterations": result.iterations,
            "accepted_updates": len(accepted),
            "initial_objective": first["objective"],
            "final_objective": float(final_value),
            "objective_reduction_percent": 100.0 * (first["objective"] - float(final_value)) / abs(first["objective"]),
            "initial_mean_streamwise_velocity": float(jnp.mean(initial_state.u)),
            "final_mean_streamwise_velocity": float(jnp.mean(final_state.u)),
            "mean_velocity_increase_percent": 100.0 * (float(jnp.mean(final_state.u)) / float(jnp.mean(initial_state.u)) - 1.0),
            "initial_projected_gradient_norm": result.initial_projected_gradient_norm,
            "final_projected_gradient_norm": result.final_projected_gradient_norm,
            "final_gradient_norm": float(jnp.linalg.norm(final_gradient)),
            "active_bounds": [bool(abs(c) >= bench.coefficient_bound - 1e-12) for c in np.asarray(final_coefficients)],
            "initial_area": float(initial_area),
            "final_area": float(final_area),
            "relative_area_error": float(abs(final_area - initial_area) / initial_area),
            "final_coefficients": np.asarray(final_coefficients).tolist(),
            "final_cardinal_radii": np.asarray(radii).tolist(),
            "final_divergence_rms": float(div_rms),
            "final_divergence_max": float(div_max),
            "loop_wall_seconds": loop_seconds,
        },
        "trace": result.trace,
        "provenance": provenance,
    }
    write_json(data_path, record)
    print(json.dumps({k: record[k] for k in ("horizon_context", "first_geometry_step", "optimization")}, indent=2))
    print(f"\nSaved: {data_path}")


if __name__ == "__main__":
    main()
