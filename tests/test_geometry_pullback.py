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
from jax_cad.unified import (
    DifferentiableProgram,
    directional_taylor_test,
    fourier_mode_sobolev_energy,
    fourier_shape_area,
    make_periodic_fourier_material_map,
)


def _program():
    grid = create_periodic_grid(12, 8, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.1,
        dt=0.003,
        force_x=1.0,
        include_advection=False,
    )
    state0 = create_initial_state(grid, dtype=jnp.float64)
    modes = (2, 3, 4)
    base_radius = 0.18
    design_map = make_periodic_fourier_material_map(
        x,
        y,
        center=(0.83, 0.46),
        base_radius=base_radius,
        modes=modes,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=60.0,
        epsilon=0.75 * min(grid.dx, grid.dy),
    )

    def solver(fields):
        return simulate(
            state0,
            fields.momentum_resistance,
            grid,
            params,
            n_steps=14,
        )

    def objective(state, fields, coefficients):
        del fields
        return mean_flow_loss(state) + 2.0e-4 * fourier_mode_sobolev_energy(
            coefficients,
            modes,
            order=2,
        )

    return DifferentiableProgram(design_map, solver, objective), modes, base_radius


def test_fourier_parameterization_preserves_area():
    _, modes, base_radius = _program()
    coefficients = jnp.array(
        [0.12, -0.03, 0.04, 0.02, -0.05, 0.01],
        dtype=jnp.float64,
    )
    area = fourier_shape_area(
        coefficients,
        base_radius=base_radius,
        modes=modes,
        n_quadrature=4096,
    )
    expected = jnp.pi * base_radius**2
    np.testing.assert_allclose(
        np.asarray(area),
        np.asarray(expected),
        rtol=2e-14,
        atol=2e-14,
    )


def test_result_to_geometry_pullback_passes_taylor_test():
    program, modes, _ = _program()
    coefficients = jnp.zeros(2 * len(modes), dtype=jnp.float64)
    direction = jnp.array(
        [0.3, -0.4, 0.2, 0.1, -0.25, 0.15],
        dtype=jnp.float64,
    )
    result = directional_taylor_test(
        program.value,
        coefficients,
        direction,
        epsilons=(8e-3, 4e-3, 2e-3, 1e-3),
    )
    assert float(jnp.min(result.first_order_rate[1:])) > 1.9


def test_one_reverse_gradient_step_modifies_geometry_and_reduces_objective():
    program, modes, base_radius = _program()
    coefficients = jnp.zeros(2 * len(modes), dtype=jnp.float64)
    value, gradient = program.value_and_grad(coefficients)
    gradient_norm = jnp.linalg.norm(gradient)
    assert float(gradient_norm) > 1e-8

    # A normalized geometry-space step demonstrates the intended closed loop:
    # result -> reverse pullback -> native geometry update -> improved result.
    updated = coefficients - 1e-3 * gradient / gradient_norm
    updated_value = program.value(updated)
    assert float(updated_value) < float(value)

    area_before = fourier_shape_area(
        coefficients,
        base_radius=base_radius,
        modes=modes,
    )
    area_after = fourier_shape_area(
        updated,
        base_radius=base_radius,
        modes=modes,
    )
    np.testing.assert_allclose(
        np.asarray(area_after),
        np.asarray(area_before),
        rtol=2e-14,
        atol=2e-14,
    )
