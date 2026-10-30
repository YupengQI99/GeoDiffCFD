"""Use one solver and one objective with three geometry/design routes.

The analytic circle, a sampled signed-distance field with a scalar offset, and
density topology logits all terminate in the same ``MaterialFields`` contract.
The shared physics is the converged mobility-weighted periodic Stokes--Brinkman
fixed point, differentiated implicitly, so that every route is tested on the
same steady state.  Each route receives a double-precision directional Taylor
test of its complete design-to-result derivative.
"""

from __future__ import annotations

import json
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
    divergence_norms,
    mean_flow_loss,
    steady_residual,
    steady_residual_norms,
    step,
)
from jax_cad.run_artifacts import provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    directional_taylor_test,
    make_implicit_fixed_point_solver,
)
from jax_cad.unified.maps import circle_sdf_periodic
from jax_cad.unified.materials import (
    build_cone_filter_kernel,
    make_density_material_map,
    make_periodic_circle_material_map,
    make_sampled_sdf_material_map,
)

TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)


def _taylor(result) -> dict[str, object]:
    return {
        "epsilons": np.asarray(result.epsilons).tolist(),
        "first_order_remainders": np.asarray(result.first_order_error).tolist(),
        "first_order_rates": np.asarray(result.first_order_rate).tolist(),
        "zero_order_rates": np.asarray(result.zero_order_rate).tolist(),
        "directional_derivative": float(result.directional_derivative),
    }


def main() -> None:
    grid = create_periodic_grid(16, 12, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.08,
        dt=0.002,
        force_x=1.0,
        include_advection=False,
        mobility_weighted=True,
        pressure_tol=1e-12,
        pressure_maxiter=500,
    )
    state0 = create_initial_state(grid, dtype=jnp.float64)
    alpha_max = 80.0
    epsilon = 0.8 * min(grid.dx, grid.dy)
    solve, primal = make_implicit_fixed_point_solver(
        lambda state, alpha: step(state, alpha, grid, params),
        max_iterations=200_000,
        atol=1e-12,
        rtol=1e-12,
        linear_tol=1e-10,
        linear_atol=1e-12,
        linear_maxiter=20_000,
        linear_restart=60,
    )

    def solver(fields):
        return solve(fields.momentum_resistance, state0)

    def objective(state, fields, design):
        del design
        return mean_flow_loss(state, fields.momentum_resistance)

    def state_diagnostics(fields):
        state, diagnostics = primal(fields.momentum_resistance, state0)
        div_rms, div_max = divergence_norms(state, grid)
        res_rms, res_max = steady_residual_norms(
            steady_residual(state, fields.momentum_resistance, grid, params)
        )
        return {
            "fixed_point_iterations": int(diagnostics.iterations),
            "fixed_point_residual": float(diagnostics.residual_norm),
            "converged": bool(diagnostics.converged),
            "divergence_rms": float(div_rms),
            "divergence_max": float(div_max),
            "steady_residual_rms": float(res_rms),
            "steady_residual_max": float(res_max),
        }

    theta = jnp.array([0.91, 0.47, 0.17], dtype=jnp.float64)

    analytic_map = make_periodic_circle_material_map(
        x, y, lx=grid.lx, ly=grid.ly, alpha_max=alpha_max, epsilon=epsilon,
        conductivity_fluid=1.0, conductivity_solid=4.0,
        heat_capacity_fluid=1.0, heat_capacity_solid=2.0,
    )
    analytic_program = DifferentiableProgram(analytic_map, solver, objective)
    value_analytic, grad_theta = analytic_program.value_and_grad(theta)
    taylor_analytic = directional_taylor_test(
        analytic_program.value, theta, jnp.array([0.3, -0.2, 0.5]), epsilons=TAYLOR_EPSILONS
    )

    phi0 = circle_sdf_periodic(x, y, theta, lx=grid.lx, ly=grid.ly)
    sampled_map = make_sampled_sdf_material_map(phi0, alpha_max=alpha_max, epsilon=epsilon)
    sampled_program = DifferentiableProgram(sampled_map, solver, objective)
    offset0 = jnp.array(0.0, dtype=jnp.float64)
    value_sampled, grad_offset = sampled_program.value_and_grad(offset0)
    taylor_sampled = directional_taylor_test(
        sampled_program.value, offset0, jnp.array(1.0), epsilons=TAYLOR_EPSILONS
    )

    kernel = build_cone_filter_kernel(1.8 * min(grid.dx, grid.dy), dx=grid.dx, dy=grid.dy, dtype=jnp.float64)
    density_settings = {"beta": 2.0, "eta": 0.5, "ramp_q": 0.2, "filter_radius_over_h": 1.8}
    density_map = make_density_material_map(
        kernel, beta=density_settings["beta"], eta=density_settings["eta"],
        alpha_max=alpha_max, ramp_q=density_settings["ramp_q"],
    )
    density_program = DifferentiableProgram(density_map, solver, objective)
    logits = 2.0 * phi0 / epsilon
    value_density, grad_density = density_program.value_and_grad(logits)
    direction = jnp.sin(jnp.arange(logits.size, dtype=jnp.float64)).reshape(logits.shape)
    taylor_density = directional_taylor_test(
        density_program.value, logits, direction, epsilons=TAYLOR_EPSILONS
    )

    report = {
        "claim": "Three design routes share one material contract, one converged solver and one objective.",
        "shared_solver": {
            "grid": [grid.nx, grid.ny],
            "domain": [grid.lx, grid.ly],
            "spacing": [grid.dx, grid.dy],
            "viscosity": params.viscosity,
            "dt": params.dt,
            "alpha_max": alpha_max,
            "epsilon": epsilon,
            "epsilon_over_h": 0.8,
            "mobility_weighted": params.mobility_weighted,
            "pressure_tol": params.pressure_tol,
            "sensitivity_backend": "implicit fixed point (custom VJP + GMRES)",
            "fixed_point_tolerances": {"atol": 1e-12, "rtol": 1e-12},
            "adjoint_tolerances": {"tol": 1e-10, "atol": 1e-12},
            "objective": "negative domain-mean streamwise velocity at unit body force",
        },
        "analytic_sdf": {
            "design_size": int(theta.size),
            "theta": np.asarray(theta).tolist(),
            "objective": float(value_analytic),
            "gradient": np.asarray(grad_theta).tolist(),
            "taylor_test": _taylor(taylor_analytic),
            "state": state_diagnostics(analytic_map(theta)),
        },
        "sampled_sdf_offset": {
            "design_size": 1,
            "objective": float(value_sampled),
            "gradient": float(grad_offset),
            "matches_analytic_objective": bool(jnp.isclose(value_sampled, value_analytic, rtol=1e-12, atol=1e-13)),
            "gradient_sign_relation": "d/d(offset) = -d/d(radius) because a positive offset expands fluid",
            "gradient_matches_negative_radius_derivative": bool(
                jnp.isclose(grad_offset, -grad_theta[2], rtol=1e-10, atol=1e-13)
            ),
            "taylor_test": _taylor(taylor_sampled),
        },
        "density_topology": {
            "design_size": int(logits.size),
            "settings": density_settings,
            "objective": float(value_density),
            "gradient_norm": float(jnp.linalg.norm(grad_density)),
            "taylor_test": _taylor(taylor_density),
            "state": state_diagnostics(density_map(logits)),
        },
        "provenance": {**provenance_record(), "script": "examples/24_multiple_geometry_routes.py"},
    }

    out = Path(__file__).resolve().parents[1] / "output" / "data"
    path = out / "24_multiple_geometry_routes.json"
    write_json(path, report)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
