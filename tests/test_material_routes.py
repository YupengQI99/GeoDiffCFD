from __future__ import annotations

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
)
from jax_cad.unified import DifferentiableProgram, directional_taylor_test
from jax_cad.unified.maps import circle_sdf_periodic
from jax_cad.unified.materials import (
    build_cone_filter_kernel,
    make_density_material_map,
    make_periodic_circle_material_map,
    make_sampled_sdf_material_map,
    periodic_density_filter,
)


def _shared_solver(grid, params, n_steps=14):
    state0 = create_initial_state(grid, dtype=jnp.float64)

    def solve(fields):
        return simulate(
            state0,
            fields.momentum_resistance,
            grid,
            params,
            n_steps=n_steps,
        )

    return solve


def _objective(state, fields, design):
    del design
    return mean_flow_loss(state, fields.momentum_resistance)


def test_periodic_filter_preserves_constant_density_on_rectangular_grid():
    grid = create_periodic_grid(9, 7, 2.0, 1.0)
    kernel = build_cone_filter_kernel(
        2.2 * min(grid.dx, grid.dy),
        dx=grid.dx,
        dy=grid.dy,
        dtype=jnp.float64,
    )
    density = jnp.full((grid.nx, grid.ny), 0.37, dtype=jnp.float64)
    filtered = periodic_density_filter(density, kernel)
    np.testing.assert_allclose(np.asarray(filtered), 0.37, rtol=0.0, atol=2e-15)


def test_analytic_and_sampled_sdf_routes_share_material_contract():
    grid = create_periodic_grid(12, 8, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(viscosity=0.1, dt=0.004, force_x=1.0, include_advection=False)
    solver = _shared_solver(grid, params)
    epsilon = 0.8 * min(grid.dx, grid.dy)
    theta = jnp.array([0.83, 0.46, 0.16], dtype=jnp.float64)

    analytic_map = make_periodic_circle_material_map(
        x,
        y,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=40.0,
        epsilon=epsilon,
    )
    phi0 = circle_sdf_periodic(x, y, theta, lx=grid.lx, ly=grid.ly)
    sampled_map = make_sampled_sdf_material_map(
        phi0,
        alpha_max=40.0,
        epsilon=epsilon,
    )

    analytic_program = DifferentiableProgram(analytic_map, solver, _objective)
    sampled_program = DifferentiableProgram(sampled_map, solver, _objective)

    value_a = analytic_program.value(theta)
    value_s, grad_offset = sampled_program.value_and_grad(jnp.array(0.0, dtype=jnp.float64))

    assert np.isclose(float(value_a), float(value_s), rtol=2e-13, atol=2e-14)
    # Positive offset expands fluid, so -mean(u) decreases.
    assert float(grad_offset) < 0.0


def test_density_route_passes_end_to_end_directional_taylor_test():
    grid = create_periodic_grid(8, 6, 2.0, 1.0)
    params = MACParams(viscosity=0.12, dt=0.004, force_x=1.0, include_advection=False)
    solver = _shared_solver(grid, params, n_steps=10)
    kernel = build_cone_filter_kernel(
        1.8 * min(grid.dx, grid.dy),
        dx=grid.dx,
        dy=grid.dy,
        dtype=jnp.float64,
    )
    density_map = make_density_material_map(
        kernel,
        beta=2.0,
        eta=0.5,
        alpha_max=25.0,
        ramp_q=0.2,
    )
    program = DifferentiableProgram(density_map, solver, _objective)

    x = jnp.linspace(-0.3, 0.4, grid.nx * grid.ny, dtype=jnp.float64).reshape(
        grid.nx, grid.ny
    )
    direction = jnp.cos(jnp.arange(x.size, dtype=jnp.float64)).reshape(x.shape)
    result = directional_taylor_test(
        program.value,
        x,
        direction,
        epsilons=(8e-3, 4e-3, 2e-3, 1e-3),
    )

    assert float(jnp.min(result.first_order_rate[1:])) > 1.85
