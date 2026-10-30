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
    project_velocity_weighted,
    simulate,
    steady_residual,
    steady_residual_norms,
    step,
)
from jax_cad.unified import (
    DifferentiableProgram,
    directional_taylor_test,
    fourier_mode_sobolev_energy,
    make_implicit_fixed_point_solver,
    make_periodic_circle_brinkman_map,
    make_periodic_fourier_material_map,
)


def _params(dt: float, weighted: bool) -> MACParams:
    return MACParams(
        viscosity=0.1,
        dt=dt,
        force_x=1.0,
        include_advection=False,
        mobility_weighted=weighted,
        pressure_tol=1e-13,
        pressure_maxiter=400,
    )


def _circle_problem():
    grid = create_periodic_grid(12, 8, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    design_map = make_periodic_circle_brinkman_map(
        x,
        y,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=60.0,
        epsilon=0.75 * min(grid.dx, grid.dy),
    )
    theta = jnp.array([0.83, 0.46, 0.18], dtype=jnp.float64)
    return grid, design_map(theta), design_map, theta


def _fixed_point(grid, alpha, params):
    _, primal = make_implicit_fixed_point_solver(
        lambda state, a: step(state, a, grid, params),
        max_iterations=200_000,
        atol=1e-12,
        rtol=1e-12,
    )
    state0 = create_initial_state(grid, dtype=jnp.float64)
    state, diagnostics = primal(alpha, state0)
    assert bool(diagnostics.converged)
    return state


def test_weighted_projection_is_discretely_solenoidal():
    grid = create_periodic_grid(18, 14, 2.0, 1.0)
    key_u, key_v, key_m = jax.random.split(jax.random.PRNGKey(7), 3)
    u_tilde = jax.random.normal(key_u, (grid.nx, grid.ny), dtype=jnp.float64)
    v_tilde = jax.random.normal(key_v, (grid.nx, grid.ny), dtype=jnp.float64)
    mobility = jax.random.uniform(
        key_m, (2, grid.nx, grid.ny), dtype=jnp.float64, minval=0.05, maxval=1.0
    )
    params = _params(0.03, weighted=True)

    projected = project_velocity_weighted(
        u_tilde, v_tilde, mobility[0], mobility[1], grid, params
    )
    rms, maximum = divergence_norms(projected, grid)

    assert float(rms) < 1e-10
    assert float(maximum) < 1e-9
    assert abs(float(jnp.mean(projected.p))) < 1e-14


def test_weighted_fixed_point_satisfies_steady_residual_and_is_dt_independent():
    grid, alpha, _, _ = _circle_problem()

    state_a = _fixed_point(grid, alpha, _params(0.003, weighted=True))
    state_b = _fixed_point(grid, alpha, _params(0.0015, weighted=True))
    rms_a, max_a = steady_residual_norms(
        steady_residual(state_a, alpha, grid, _params(0.003, weighted=True))
    )
    rms_b, _ = steady_residual_norms(
        steady_residual(state_b, alpha, grid, _params(0.0015, weighted=True))
    )

    assert float(rms_a) < 1e-9 and float(max_a) < 1e-8
    assert float(rms_b) < 1e-9
    np.testing.assert_allclose(
        float(mean_flow_loss(state_a)), float(mean_flow_loss(state_b)), rtol=1e-9
    )
    np.testing.assert_allclose(np.asarray(state_a.u), np.asarray(state_b.u), atol=1e-8)


def test_constant_mobility_fixed_point_carries_first_order_splitting_error():
    grid, alpha, _, _ = _circle_problem()

    residuals = []
    means = []
    for dt in (0.003, 0.0015):
        params = _params(dt, weighted=False)
        state = _fixed_point(grid, alpha, params)
        rms, _ = steady_residual_norms(steady_residual(state, alpha, grid, params))
        residuals.append(float(rms))
        means.append(float(mean_flow_loss(state)))

    # The un-weighted projection converges to a dt-dependent balance: the true
    # steady residual is O(dt) and halves when dt halves.
    assert residuals[0] > 1e-3
    assert 1.8 < residuals[0] / residuals[1] < 2.2
    assert abs(means[0] - means[1]) > 1e-6


def test_weighted_kernel_geometry_pullback_passes_taylor_test():
    grid = create_periodic_grid(12, 8, 2.0, 1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = _params(0.003, weighted=True)
    state0 = create_initial_state(grid, dtype=jnp.float64)
    modes = (2, 3, 4)
    design_map = make_periodic_fourier_material_map(
        x,
        y,
        center=(0.83, 0.46),
        base_radius=0.18,
        modes=modes,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=60.0,
        epsilon=0.75 * min(grid.dx, grid.dy),
    )

    def solver(fields):
        return simulate(state0, fields.momentum_resistance, grid, params, n_steps=14)

    def objective(state, fields, coefficients):
        del fields
        return mean_flow_loss(state) + 2.0e-4 * fourier_mode_sobolev_energy(
            coefficients, modes, order=2
        )

    program = DifferentiableProgram(design_map, solver, objective)
    coefficients = jnp.zeros(2 * len(modes), dtype=jnp.float64)
    direction = jnp.array([0.3, -0.4, 0.2, 0.1, -0.25, 0.15], dtype=jnp.float64)
    result = directional_taylor_test(
        program.value, coefficients, direction, epsilons=(8e-3, 4e-3, 2e-3, 1e-3)
    )
    assert float(jnp.min(result.first_order_rate[1:])) > 1.9


def test_weighted_kernel_implicit_gradient_matches_unrolled_gradient():
    grid, _, design_map, theta = _circle_problem()
    params = _params(0.003, weighted=True)
    state0 = create_initial_state(grid, dtype=jnp.float64)

    implicit_solve, solve_with_diagnostics = make_implicit_fixed_point_solver(
        lambda state, a: step(state, a, grid, params),
        max_iterations=50_000,
        atol=1e-12,
        rtol=1e-12,
        linear_tol=1e-10,
        linear_atol=1e-12,
        linear_maxiter=5_000,
        linear_restart=50,
    )
    objective = lambda state, alpha, theta_: mean_flow_loss(state, alpha)
    implicit_program = DifferentiableProgram(
        design_map, lambda alpha: implicit_solve(alpha, state0), objective
    )
    unrolled_program = DifferentiableProgram(
        design_map,
        # The compact profile leaves the fluid exactly resistance-free, so the
        # transient approaches the fixed point more slowly than the legacy
        # logistic map; 24000 steps (t = 72) reach it to ~1e-12 relative.
        lambda alpha: simulate(state0, alpha, grid, params, n_steps=24000),
        objective,
    )

    _, diagnostics = solve_with_diagnostics(design_map(theta), state0)
    value_i, grad_i = implicit_program.value_and_grad(theta)
    value_u, grad_u = unrolled_program.value_and_grad(theta)

    assert bool(diagnostics.converged)
    np.testing.assert_allclose(float(value_i), float(value_u), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_i), np.asarray(grad_u), rtol=1e-6, atol=1e-9)


def test_direct_steady_solve_matches_weighted_fixed_point_and_adjoint_is_residual_based():
    from jax_cad.differentiable.mac_periodic import (
        steady_adjoint_residual,
        steady_stokes_brinkman,
    )

    grid, alpha, design_map, theta = _circle_problem()
    params = _params(0.003, weighted=True)

    direct = steady_stokes_brinkman(alpha, grid, params, tol=1e-13, atol=0.0, maxiter=2000)
    picard = _fixed_point(grid, alpha, params)
    np.testing.assert_allclose(np.asarray(direct.u), np.asarray(picard.u), atol=1e-9)
    np.testing.assert_allclose(np.asarray(direct.p), np.asarray(picard.p), atol=1e-8)
    rms, maximum = steady_residual_norms(steady_residual(direct, alpha, grid, params))
    assert float(rms) < 1e-10 and float(maximum) < 1e-9

    # The custom linear-solve rule differentiates the converged state implicitly:
    # compare with reverse mode through the long unrolled transient.
    objective = lambda state: -jnp.mean(state.u)
    direct_grad = jax.grad(lambda th: objective(steady_stokes_brinkman(design_map(th), grid, params, tol=1e-13, maxiter=2000)))(theta)
    state0 = create_initial_state(grid, dtype=jnp.float64)
    unrolled_grad = jax.grad(lambda th: objective(simulate(state0, design_map(th), grid, params, n_steps=24000)))(theta)
    np.testing.assert_allclose(np.asarray(direct_grad), np.asarray(unrolled_grad), rtol=1e-6, atol=1e-9)

    cotangent = jax.grad(objective)(direct)
    _, residual = steady_adjoint_residual(direct, alpha, grid, params, cotangent, tol=1e-12)
    assert float(residual) < 1e-9
