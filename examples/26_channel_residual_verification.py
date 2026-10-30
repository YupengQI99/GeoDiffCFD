"""Verify non-periodic compatible projection and native-geometry pullback."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    apply_velocity_boundaries,
    cell_centres,
    create_channel_grid,
    create_initial_state,
    interpolate_alpha_to_faces,
    parabolic_inlet,
    project_velocity,
    simulate,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "data" / "26_channel_residual_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    grid = create_channel_grid(18, 11, 2.0, 1.0)
    inlet = parabolic_inlet(grid, 0.7, dtype=jnp.float64)
    ii = jnp.arange((grid.nx + 1) * grid.ny, dtype=jnp.float64).reshape(
        grid.nx + 1, grid.ny
    )
    jj = jnp.arange(grid.nx * (grid.ny + 1), dtype=jnp.float64).reshape(
        grid.nx, grid.ny + 1
    )
    u, v = apply_velocity_boundaries(
        0.2 * jnp.sin(0.17 * ii),
        0.15 * jnp.cos(0.11 * jj),
        inlet,
        grid,
    )
    alpha = jnp.linspace(
        0.0, 20.0, grid.nx * grid.ny, dtype=jnp.float64
    ).reshape(grid.nx, grid.ny)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)

    tolerance_sweep = []
    for tolerance in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        params = ChannelParams(
            viscosity=0.1,
            dt=0.01,
            pressure_tolerance=tolerance,
            pressure_atol=0.0,
            pressure_maxiter=2000,
        )
        mobility_u = 1.0 / (1.0 + params.dt * alpha_u)
        mobility_v = 1.0 / (1.0 + params.dt * alpha_v)
        _, diagnostics = project_velocity(
            u, v, mobility_u, mobility_v, inlet, grid, params
        )
        tolerance_sweep.append(
            {
                "pressure_tolerance": tolerance,
                "pressure_residual_relative": float(
                    diagnostics.pressure_residual_rel
                ),
                "divergence_rms": float(diagnostics.divergence_rms),
                "divergence_max": float(diagnostics.divergence_max),
                "mass_flux_mismatch": float(diagnostics.mass_flux_mismatch),
            }
        )

    gradient_grid = create_channel_grid(12, 8, 2.0, 1.0)
    gradient_inlet = parabolic_inlet(
        gradient_grid, 0.6, dtype=jnp.float64
    )
    state0 = create_initial_state(
        gradient_grid, gradient_inlet, dtype=jnp.float64
    )
    x, y = cell_centres(gradient_grid, dtype=jnp.float64)
    params = ChannelParams(
        viscosity=0.08,
        dt=0.0015,
        pressure_tolerance=1e-11,
        pressure_atol=1e-13,
        pressure_maxiter=1000,
    )
    epsilon = 0.7 * min(gradient_grid.dx, gradient_grid.dy)

    def pressure_drop_objective(theta):
        cx, cy, radius = theta
        phi = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-30) - radius
        resistance = 80.0 * jax.nn.sigmoid(-phi / epsilon)
        final_state, _ = simulate(
            state0,
            resistance,
            gradient_inlet,
            gradient_grid,
            params,
            n_steps=8,
        )
        # The outlet pressure is zero by construction.
        return jnp.mean(final_state.p[0, :])

    theta = jnp.array([0.85, 0.51, 0.17], dtype=jnp.float64)
    direction = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    value, gradient = jax.value_and_grad(pressure_drop_objective)(theta)
    epsilons = jnp.array([4e-3, 2e-3, 1e-3, 5e-4], dtype=jnp.float64)
    directional = jnp.vdot(gradient, direction)
    remainders = jnp.asarray(
        [
            jnp.abs(
                pressure_drop_objective(theta + h * direction)
                - value
                - h * directional
            )
            for h in epsilons
        ]
    )
    rates = jnp.log(remainders[:-1] / remainders[1:]) / jnp.log(
        epsilons[:-1] / epsilons[1:]
    )

    report = {
        "scope": (
            "Small CPU verification of the non-periodic variable-mobility "
            "projection; not a final engineering benchmark."
        ),
        "projection": {
            "grid": [grid.nx, grid.ny],
            "domain": [grid.lx, grid.ly],
            "spacing": [grid.dx, grid.dy],
            "tolerance_sweep": tolerance_sweep,
        },
        "geometry_pullback": {
            "grid": [gradient_grid.nx, gradient_grid.ny],
            "theta": np.asarray(theta).tolist(),
            "objective": float(value),
            "gradient": np.asarray(gradient).tolist(),
            "directional_derivative": float(directional),
            "epsilons": np.asarray(epsilons).tolist(),
            "first_order_remainders": np.asarray(remainders).tolist(),
            "first_order_rates": np.asarray(rates).tolist(),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
