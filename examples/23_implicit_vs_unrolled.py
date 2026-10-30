"""Compare unrolled and implicit differentiation of the same steady solver.

Both programs share the same geometry map, fixed-point update, and objective.
Only the derivative backend changes:

- unrolled: reverse-mode through 500 pseudo-time steps;
- implicit: convergence-based forward solve plus matrix-free adjoint solve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import (
    MACParams,
    cell_centres,
    create_initial_state,
    create_periodic_grid,
    mean_flow_loss,
    simulate,
    step,
)
from jax_cad.run_artifacts import provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    make_implicit_fixed_point_solver,
    make_periodic_circle_brinkman_map,
)


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def _time_call(fn, argument, repeats: int = 3):
    output = fn(argument)
    _block_tree(output)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn(argument)
        _block_tree(output)
        times.append(time.perf_counter() - start)
    return output, times


def main() -> None:
    grid = create_periodic_grid(6, 6, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.2,
        dt=0.03,
        force_x=1.0,
        include_advection=False,
        mobility_weighted=True,
        pressure_tol=1e-12,
        pressure_maxiter=500,
    )
    state0 = create_initial_state(grid, dtype=jnp.float64)
    design_map = make_periodic_circle_brinkman_map(
        x,
        y,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=15.0,
        epsilon=0.8 * min(grid.dx, grid.dy),
    )

    def fixed_point_update(state, alpha):
        return step(state, alpha, grid, params)

    implicit_solve, solve_with_diagnostics = make_implicit_fixed_point_solver(
        fixed_point_update,
        max_iterations=2000,
        atol=1e-9,
        rtol=1e-9,
        linear_tol=1e-7,
        linear_atol=1e-9,
        linear_maxiter=80,
        linear_restart=20,
    )

    implicit_program = DifferentiableProgram(
        design_map=design_map,
        solver=lambda alpha: implicit_solve(alpha, state0),
        objective=lambda state, alpha, theta: mean_flow_loss(state, alpha),
    )
    unrolled_program = DifferentiableProgram(
        design_map=design_map,
        solver=lambda alpha: simulate(state0, alpha, grid, params, n_steps=500),
        objective=lambda state, alpha, theta: mean_flow_loss(state, alpha),
    )

    theta = jnp.array([0.91, 0.47, 0.16], dtype=jnp.float64)
    alpha = design_map(theta)
    _, diagnostics = solve_with_diagnostics(alpha, state0)

    implicit_vg = jax.jit(jax.value_and_grad(implicit_program.value))
    unrolled_vg = jax.jit(jax.value_and_grad(unrolled_program.value))

    (value_i, grad_i), time_i = _time_call(implicit_vg, theta)
    (value_u, grad_u), time_u = _time_call(unrolled_vg, theta)

    relative_gradient_error = float(
        jnp.linalg.norm(grad_i - grad_u) / jnp.maximum(jnp.linalg.norm(grad_u), 1e-14)
    )
    report = {
        "scope": "small unit-scale backend check; the manuscript's cross-backend evidence on the flagship problem is examples/31_steady_result_to_geometry.py",
        "grid": [grid.nx, grid.ny],
        "configuration": {
            "viscosity": params.viscosity,
            "dt": params.dt,
            "alpha_max": 15.0,
            "epsilon_over_h": 0.8,
            "mobility_weighted": params.mobility_weighted,
            "unrolled_steps": 500,
            "timing_samples": 3,
        },
        "theta": np.asarray(theta).tolist(),
        "fixed_point_iterations": int(diagnostics.iterations),
        "fixed_point_residual": float(diagnostics.residual_norm),
        "converged": bool(diagnostics.converged),
        "implicit": {
            "objective": float(value_i),
            "gradient": np.asarray(grad_i).tolist(),
            "warm_runtime_seconds": time_i,
            "median_runtime_seconds": float(np.median(time_i)),
        },
        "unrolled_500_steps": {
            "objective": float(value_u),
            "gradient": np.asarray(grad_u).tolist(),
            "warm_runtime_seconds": time_u,
            "median_runtime_seconds": float(np.median(time_u)),
        },
        "relative_gradient_difference": relative_gradient_error,
        "provenance": {**provenance_record(), "script": "examples/23_implicit_vs_unrolled.py"},
    }

    out = Path(__file__).resolve().parents[1] / "output" / "data"
    path = out / "23_implicit_vs_unrolled.json"
    write_json(path, report)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
