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
    step,
)
from jax_cad.unified import DifferentiableProgram, make_periodic_circle_brinkman_map
from jax_cad.unified.implicit import make_implicit_fixed_point_solver


def test_scalar_fixed_point_implicit_gradient_matches_exact_result():
    # x = a*x + b, so x = b/(1-a).
    b = jnp.array(0.7, dtype=jnp.float64)

    def update(x, a):
        return a * x + b

    solve, solve_with_diagnostics = make_implicit_fixed_point_solver(
        update,
        max_iterations=500,
        atol=1e-13,
        rtol=1e-13,
        linear_tol=1e-12,
        linear_atol=1e-13,
    )
    a = jnp.array(0.35, dtype=jnp.float64)
    x0 = jnp.array(0.0, dtype=jnp.float64)

    objective = lambda coefficient: solve(coefficient, x0) ** 2
    value, gradient = jax.value_and_grad(objective)(a)

    exact_x = b / (1.0 - a)
    exact_value = exact_x**2
    exact_gradient = 2.0 * b**2 / (1.0 - a) ** 3
    _, diagnostics = solve_with_diagnostics(a, x0)

    assert bool(diagnostics.converged)
    assert np.isclose(float(value), float(exact_value), rtol=1e-10, atol=1e-12)
    assert np.isclose(float(gradient), float(exact_gradient), rtol=2e-9, atol=1e-11)


def test_mac_implicit_gradient_matches_long_unrolled_gradient():
    grid = create_periodic_grid(6, 6, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.2,
        dt=0.03,
        force_x=1.0,
        include_advection=False,
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
        max_iterations=5000,
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
        solver=lambda alpha: simulate(state0, alpha, grid, params, n_steps=2000),
        objective=lambda state, alpha, theta: mean_flow_loss(state, alpha),
    )

    theta = jnp.array([0.91, 0.47, 0.16], dtype=jnp.float64)
    alpha = design_map(theta)
    _, diagnostics = solve_with_diagnostics(alpha, state0)
    value_implicit, grad_implicit = implicit_program.value_and_grad(theta)
    value_unrolled, grad_unrolled = unrolled_program.value_and_grad(theta)

    assert bool(diagnostics.converged)
    assert np.isclose(float(value_implicit), float(value_unrolled), rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(
        np.asarray(grad_implicit),
        np.asarray(grad_unrolled),
        rtol=2e-5,
        atol=2e-7,
    )
