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
    divergence_norms,
    mean_flow_loss,
    project_velocity,
    simulate,
)
from jax_cad.unified import (
    DifferentiableProgram,
    directional_taylor_test,
    make_periodic_circle_brinkman_map,
)


def _make_problem(n_steps: int = 24):
    grid = create_periodic_grid(16, 12, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.08,
        dt=0.002,
        force_x=1.0,
        force_y=0.0,
        include_advection=False,
    )
    design_map = make_periodic_circle_brinkman_map(
        x,
        y,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=80.0,
        epsilon=0.8 * min(grid.dx, grid.dy),
    )
    state0 = create_initial_state(grid, dtype=jnp.float64)

    def solver(alpha):
        return simulate(state0, alpha, grid, params, n_steps=n_steps)

    def objective(state, alpha, theta):
        del theta
        return mean_flow_loss(state, alpha)

    return grid, DifferentiableProgram(design_map, solver, objective)


def test_periodic_projection_is_discretely_solenoidal():
    grid = create_periodic_grid(18, 14, 2.0, 1.0)
    key_u, key_v = jax.random.split(jax.random.PRNGKey(4))
    u_star = jax.random.normal(key_u, (grid.nx, grid.ny), dtype=jnp.float64)
    v_star = jax.random.normal(key_v, (grid.nx, grid.ny), dtype=jnp.float64)

    projected = project_velocity(u_star, v_star, grid, dt=0.03)
    rms, maximum = divergence_norms(projected, grid)

    assert float(rms) < 2e-11
    assert float(maximum) < 8e-11


def test_reverse_mode_reaches_geometry_parameters():
    _, problem = _make_problem()
    theta = jnp.array([1.0, 0.5, 0.18], dtype=jnp.float64)

    value, grad = problem.value_and_grad(theta)

    assert np.isfinite(float(value))
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    # A larger obstacle reduces mean flow, so the loss -mean(u) increases.
    assert float(grad[2]) > 0.0


def test_geometry_gradient_matches_central_difference():
    _, problem = _make_problem(n_steps=20)
    theta = jnp.array([0.93, 0.47, 0.17], dtype=jnp.float64)
    grad = jax.grad(problem.value)(theta)

    h = 2e-4
    direction = jnp.array([0.2, -0.3, 0.7], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    fd = (problem.value(theta + h * direction) - problem.value(theta - h * direction)) / (2.0 * h)
    ad = jnp.vdot(grad, direction)
    rel = jnp.abs(ad - fd) / jnp.maximum(jnp.abs(fd), 1e-12)

    assert float(rel) < 2e-4


def test_directional_taylor_remainder_is_second_order():
    _, problem = _make_problem(n_steps=16)
    theta = jnp.array([0.91, 0.44, 0.16], dtype=jnp.float64)
    direction = jnp.array([0.15, -0.21, 0.32], dtype=jnp.float64)

    result = directional_taylor_test(
        problem.value,
        theta,
        direction,
        epsilons=(4e-3, 2e-3, 1e-3, 5e-4),
    )

    # Ignore the first pre-asymptotic interval and require near-quadratic decay.
    assert float(jnp.min(result.first_order_rate[1:])) > 1.85
