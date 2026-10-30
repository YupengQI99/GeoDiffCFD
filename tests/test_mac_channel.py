from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    ChannelState,
    apply_velocity_boundaries,
    cell_centres,
    conservative_advection,
    conservative_momentum_fluxes,
    create_channel_grid,
    create_initial_state,
    flux_diagnostics,
    interpolate_alpha_to_faces,
    parabolic_inlet,
    project_velocity,
    residual_rms,
    simulate,
    solve_steady,
    step,
    steady_residual,
    weighted_pressure_operator,
)


def test_steady_one_step_matches_direct_step() -> None:
    """Guard against stale inner-CG state in accelerator loop lowering."""

    grid = create_channel_grid(9, 6, 1.8, 1.0)
    inlet = parabolic_inlet(grid, 0.25, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    alpha = jnp.linspace(
        0.0, 80.0, grid.nx * grid.ny, dtype=jnp.float64
    ).reshape(grid.nx, grid.ny)
    params = ChannelParams(
        viscosity=0.08,
        dt=0.002,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=0.0,
        nonlinear_atol=0.0,
    )

    direct_state, direct_diagnostics = step(
        state0, alpha, inlet, grid, params
    )
    steady_state, steady_diagnostics = solve_steady(
        state0, alpha, inlet, grid, params, max_steps=1
    )

    for direct, nested in zip(direct_state, steady_state, strict=True):
        np.testing.assert_allclose(
            np.asarray(nested), np.asarray(direct), rtol=1.0e-13, atol=1.0e-13
        )
    assert float(direct_diagnostics.pressure_residual_rel) < 2.0e-12
    assert (
        float(steady_diagnostics.pressure_residual_rel_history[0]) < 2.0e-12
    )
    assert int(steady_diagnostics.iterations) == 1
    assert bool(steady_diagnostics.active_history[0])


def test_weighted_operator_is_symmetric_positive_definite() -> None:
    grid = create_channel_grid(7, 5, 2.0, 1.0)
    alpha = jnp.linspace(
        0.0, 20.0, grid.nx * grid.ny, dtype=jnp.float64
    ).reshape(grid.nx, grid.ny)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    mobility_u = 1.0 / (1.0 + 0.01 * alpha_u)
    mobility_v = 1.0 / (1.0 + 0.01 * alpha_v)
    x = jnp.sin(jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)).reshape(
        grid.nx, grid.ny
    )
    y = jnp.cos(
        0.7 * jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)
    ).reshape(grid.nx, grid.ny)
    ax = weighted_pressure_operator(x, mobility_u, mobility_v, grid)
    ay = weighted_pressure_operator(y, mobility_u, mobility_v, grid)
    np.testing.assert_allclose(
        float(jnp.vdot(x, ay)),
        float(jnp.vdot(ax, y)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(jnp.vdot(x, ax)) > 0.0
    constant = jnp.ones_like(x)
    a_constant = weighted_pressure_operator(
        constant, mobility_u, mobility_v, grid
    )
    # The pressure-controlled outlet removes the constant-pressure null mode.
    assert float(jnp.vdot(constant, a_constant)) > 0.0


def test_projection_enforces_divergence_and_mass_balance_on_rectangular_grid() -> None:
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
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    params = ChannelParams(
        0.1,
        0.01,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
    )
    state, diagnostics = project_velocity(
        u,
        v,
        jnp.ones_like(alpha_u),
        jnp.ones_like(alpha_v),
        inlet,
        grid,
        params,
    )
    assert float(diagnostics.divergence_rms) < 2.0e-10
    assert float(diagnostics.divergence_max) < 2.0e-9
    assert float(diagnostics.mass_flux_mismatch) < 2.0e-10
    np.testing.assert_allclose(
        np.asarray(state.u[0]), np.asarray(inlet), rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        np.asarray(state.v[:, [0, -1]]), 0.0, rtol=0.0, atol=1.0e-14
    )


@pytest.mark.parametrize("scheme", ["donor-cell", "central"])
def test_geometry_to_nonlinear_pressure_drop_has_second_order_taylor_remainder(scheme) -> None:
    grid = create_channel_grid(8, 5, 1.6, 1.0)
    inlet = parabolic_inlet(grid, 0.2, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = ChannelParams(
        0.15,
        0.005,
        advection_scheme=scheme,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
    )
    epsilon = 0.7 * min(grid.dx, grid.dy)

    def resistance(theta):
        cx, cy, radius = theta
        phi = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1.0e-30) - radius
        return 50.0 * jax.nn.sigmoid(-phi / epsilon)

    def objective(theta):
        state, _ = simulate(
            state0, resistance(theta), inlet, grid, params, n_steps=400
        )
        # Outlet pressure is zero; mean inlet pressure is the pressure drop.
        return jnp.mean(state.p[0, :])

    theta = jnp.array([0.75, 0.51, 0.14], dtype=jnp.float64)
    direction = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    value, gradient = jax.value_and_grad(objective)(theta)
    steps = jnp.array([4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4])
    derivative = jnp.vdot(gradient, direction)
    remainders = jnp.asarray(
        [
            jnp.abs(objective(theta + step * direction) - value - step * derivative)
            for step in steps
        ]
    )
    rates = jnp.log(remainders[:-1] / remainders[1:]) / jnp.log(
        steps[:-1] / steps[1:]
    )
    final_state, _ = simulate(
        state0, resistance(theta), inlet, grid, params, n_steps=400
    )
    final_residual = steady_residual(
        final_state, resistance(theta), inlet, grid, params
    )

    assert float(residual_rms(final_residual)) < 1.0e-8
    assert float(jnp.min(rates[1:])) > 1.8


def test_unobstructed_channel_preserves_imposed_flux() -> None:
    grid = create_channel_grid(20, 12, 2.0, 1.0)
    inlet = parabolic_inlet(grid, 0.5, dtype=jnp.float64)
    state = create_initial_state(grid, inlet, dtype=jnp.float64)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    params = ChannelParams(
        0.05,
        0.001,
        pressure_tolerance=1.0e-11,
        pressure_atol=1.0e-13,
        pressure_maxiter=1000,
    )
    _, diagnostics = simulate(state, alpha, inlet, grid, params, n_steps=15)
    assert float(diagnostics.mass_flux_mismatch) < 1.0e-9
    assert float(diagnostics.divergence_rms) < 1.0e-9


def test_tighter_pressure_tolerance_reduces_conservation_error() -> None:
    grid = create_channel_grid(14, 9, 2.0, 1.0)
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
        0.0, 25.0, grid.nx * grid.ny, dtype=jnp.float64
    ).reshape(grid.nx, grid.ny)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)

    def run(tolerance):
        params = ChannelParams(
            0.1,
            0.01,
            pressure_tolerance=tolerance,
            pressure_atol=0.0,
            pressure_maxiter=2000,
        )
        mobility_u = 1.0 / (1.0 + params.dt * alpha_u)
        mobility_v = 1.0 / (1.0 + params.dt * alpha_v)
        _, diagnostics = project_velocity(
            u, v, mobility_u, mobility_v, inlet, grid, params
        )
        return diagnostics

    loose = run(1.0e-5)
    tight = run(1.0e-11)
    assert float(tight.pressure_residual_rel) < float(loose.pressure_residual_rel) * 1.0e-4
    assert float(tight.divergence_rms) < float(loose.divergence_rms) * 1.0e-4
    assert float(tight.mass_flux_mismatch) < float(loose.mass_flux_mismatch) * 1.0e-4


def test_steady_residual_supports_jvp_and_vjp() -> None:
    grid = create_channel_grid(8, 5, 2.0, 1.0)
    inlet = parabolic_inlet(grid, 0.5, dtype=jnp.float64)
    state = create_initial_state(grid, inlet, dtype=jnp.float64)
    alpha = jnp.ones((grid.nx, grid.ny), dtype=jnp.float64)
    params = ChannelParams(0.1, 0.002)

    def residual_from_alpha(candidate):
        residual = steady_residual(state, candidate, inlet, grid, params)
        return jnp.concatenate(
            [
                residual.momentum_u.ravel(),
                residual.momentum_v.ravel(),
                residual.continuity.ravel(),
            ]
        )

    direction = jnp.cos(jnp.arange(alpha.size, dtype=jnp.float64)).reshape(
        alpha.shape
    )
    value, tangent = jax.jvp(residual_from_alpha, (alpha,), (direction,))
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    _, pullback = jax.vjp(residual_from_alpha, alpha)
    cotangent = pullback(jnp.ones_like(value))[0]
    assert cotangent.shape == alpha.shape
    assert bool(jnp.all(jnp.isfinite(cotangent)))


@pytest.mark.parametrize("scheme", ["donor-cell", "central"])
def test_conservative_u_momentum_fluxes_close_on_rectangular_grid(scheme) -> None:
    grid = create_channel_grid(9, 6, 2.3, 1.0)
    inlet = parabolic_inlet(grid, 0.4, dtype=jnp.float64)
    ii = jnp.arange((grid.nx + 1) * grid.ny, dtype=jnp.float64).reshape(
        grid.nx + 1, grid.ny
    )
    jj = jnp.arange(grid.nx * (grid.ny + 1), dtype=jnp.float64).reshape(
        grid.nx, grid.ny + 1
    )
    u, v = apply_velocity_boundaries(
        0.35 + 0.08 * jnp.sin(0.31 * ii),
        0.06 * jnp.cos(0.23 * jj),
        inlet,
        grid,
    )
    fluxes = conservative_momentum_fluxes(u, v, grid, scheme=scheme)
    adv_u, adv_v = conservative_advection(u, v, grid, scheme=scheme)

    x_weights = jnp.ones(grid.nx, dtype=jnp.float64).at[-1].set(0.5)
    volume_integral = (
        jnp.sum(adv_u[1:, :] * x_weights[:, None]) * grid.dx * grid.dy
    )
    boundary_flux = jnp.sum(fluxes.uu[-1, :] - fluxes.uu[0, :]) * grid.dy
    np.testing.assert_allclose(
        float(volume_integral),
        float(boundary_flux),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    transverse_volume_integral = (
        jnp.sum(adv_v[:, 1 : grid.ny]) * grid.dx * grid.dy
    )
    transverse_boundary_flux = (
        jnp.sum(
            fluxes.uv_v[-1, 1 : grid.ny]
            - fluxes.uv_v[0, 1 : grid.ny]
        )
        * grid.dy
        + jnp.sum(fluxes.vv[:, grid.ny - 1] - fluxes.vv[:, 0])
        * grid.dx
    )
    np.testing.assert_allclose(
        float(transverse_volume_integral),
        float(transverse_boundary_flux),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    profile = jnp.broadcast_to(inlet, (grid.nx + 1, grid.ny))
    zero_v = jnp.zeros((grid.nx, grid.ny + 1), dtype=jnp.float64)
    uniform_adv_u, uniform_adv_v = conservative_advection(profile, zero_v, grid, scheme=scheme)
    np.testing.assert_allclose(np.asarray(uniform_adv_u), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(uniform_adv_v), 0.0, atol=1.0e-14)


@pytest.mark.parametrize("scheme", ["donor-cell", "central"])
def test_nonlinear_steady_residual_supports_state_jvp_and_vjp(scheme) -> None:
    grid = create_channel_grid(8, 5, 2.0, 1.0)
    inlet = parabolic_inlet(grid, 0.35, dtype=jnp.float64)
    ii = jnp.arange((grid.nx + 1) * grid.ny, dtype=jnp.float64).reshape(
        grid.nx + 1, grid.ny
    )
    jj = jnp.arange(grid.nx * (grid.ny + 1), dtype=jnp.float64).reshape(
        grid.nx, grid.ny + 1
    )
    u, v = apply_velocity_boundaries(
        0.3 + 0.04 * jnp.sin(0.17 * ii),
        0.03 * jnp.cos(0.13 * jj),
        inlet,
        grid,
    )
    p = 0.02 * jnp.cos(
        0.19 * jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)
    ).reshape(grid.nx, grid.ny)
    state = ChannelState(u, v, p)
    alpha = 1.0 + 0.2 * jnp.sin(
        0.11 * jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)
    ).reshape(grid.nx, grid.ny)
    params = ChannelParams(0.1, 0.002, include_advection=True, advection_scheme=scheme)

    def residual_from_inputs(candidate_state, candidate_alpha):
        residual = steady_residual(
            candidate_state, candidate_alpha, inlet, grid, params
        )
        return jnp.concatenate(
            [
                residual.momentum_u.ravel(),
                residual.momentum_v.ravel(),
                residual.continuity.ravel(),
            ]
        )

    state_direction = ChannelState(
        0.01 * jnp.cos(0.07 * ii),
        0.01 * jnp.sin(0.09 * jj),
        0.01 * jnp.cos(
            0.05 * jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)
        ).reshape(grid.nx, grid.ny),
    )
    alpha_direction = jnp.cos(
        jnp.arange(alpha.size, dtype=jnp.float64)
    ).reshape(alpha.shape)
    value, tangent = jax.jvp(
        residual_from_inputs,
        (state, alpha),
        (state_direction, alpha_direction),
    )
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0

    _, pullback = jax.vjp(residual_from_inputs, state, alpha)
    state_bar, alpha_bar = pullback(jnp.ones_like(value))
    for leaf in jax.tree_util.tree_leaves(state_bar):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert alpha_bar.shape == alpha.shape
    assert bool(jnp.all(jnp.isfinite(alpha_bar)))


def test_steady_navier_stokes_solver_reaches_residual_and_flux_tolerances() -> None:
    grid = create_channel_grid(12, 8, 1.5, 1.0)
    inlet = parabolic_inlet(grid, 0.2, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    params = ChannelParams(
        0.2,
        0.005,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=1.0e-6,
        nonlinear_atol=1.0e-8,
    )
    state, diagnostics = solve_steady(
        state0, alpha, inlet, grid, params, max_steps=500
    )
    residual = steady_residual(state, alpha, inlet, grid, params)
    div_rms, div_max, _, _, mismatch = flux_diagnostics(state, grid)
    threshold = max(
        params.nonlinear_atol,
        params.nonlinear_rtol * float(diagnostics.initial_residual_rms),
    )

    assert bool(diagnostics.converged)
    assert int(diagnostics.iterations) < 500
    assert float(residual_rms(residual)) <= threshold * (1.0 + 1.0e-10)
    assert float(diagnostics.final_residual_max) < 5.0e-6
    assert float(div_rms) < 1.0e-10
    assert float(div_max) < 1.0e-9
    assert float(mismatch) < 1.0e-10


def test_near_blockage_routes_flux_through_resolved_gap() -> None:
    grid = create_channel_grid(18, 12, 2.0, 1.0)
    inlet = parabolic_inlet(grid, 0.12, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    x, y = cell_centres(grid, dtype=jnp.float64)
    slab = jnp.abs(x - 1.0) < 1.1 * grid.dx
    gap = jnp.abs(y - 0.5) < 1.1 * grid.dy
    alpha = 2.0e4 * (slab & ~gap)
    params = ChannelParams(
        0.12,
        0.003,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=2.0e-6,
        nonlinear_atol=1.0e-9,
    )
    state, diagnostics = solve_steady(
        state0, alpha, inlet, grid, params, max_steps=700
    )
    div_rms, div_max, inlet_flux, _, mismatch = flux_diagnostics(state, grid)

    barrier_face = grid.nx // 2
    y_faces = (jnp.arange(grid.ny, dtype=jnp.float64) + 0.5) * grid.dy
    solid_rows = jnp.abs(y_faces - 0.5) >= 1.1 * grid.dy
    leakage_fraction = (
        jnp.sum(jnp.abs(state.u[barrier_face, solid_rows])) * grid.dy
        / jnp.abs(inlet_flux)
    )
    resolved_gap_fraction = (
        jnp.sum(jnp.abs(state.u[barrier_face, ~solid_rows])) * grid.dy
        / jnp.abs(inlet_flux)
    )

    assert bool(diagnostics.converged)
    assert float(diagnostics.final_residual_rms) < 5.0e-7
    assert float(div_rms) < 1.0e-10
    assert float(div_max) < 1.0e-9
    assert float(mismatch) < 1.0e-10
    assert float(leakage_fraction) < 0.02
    assert float(resolved_gap_fraction) > 0.95
    assert float(jnp.mean(state.p[0, :])) > 1.0
