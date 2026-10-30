"""Closed-domain 2-D MAC Navier--Stokes solver for lid-driven cavities.

This module complements :mod:`jax_cad.differentiable.mac_channel`.  It keeps
the same staggered state layout and conservative donor-cell momentum fluxes,
but uses impermeable velocity boundaries on all four sides and a zero-mean
pressure gauge.  The pressure operator is

``A p = -D M G p + mean(p)``.

The rank-one mean term removes the constant-pressure null mode without
changing the projected velocity.  This makes the matrix-free operator
symmetric positive definite while retaining homogeneous normal pressure
corrections at every wall.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mac_channel import (
    Array,
    ChannelGrid2D,
    ChannelParams,
    ChannelResidual,
    ChannelState,
    _forward_conjugate_gradient,
    conservative_advection,
    divergence,
    interpolate_alpha_to_faces,
    residual_max,
    residual_rms,
)


class CavityWalls(NamedTuple):
    """Tangential wall velocities for a closed rectangular domain."""

    bottom_u: float = 0.0
    top_u: float = 1.0
    left_v: float = 0.0
    right_v: float = 0.0


class CavityProjectionDiagnostics(NamedTuple):
    pressure_residual_abs: Array
    pressure_residual_rel: Array
    divergence_rms: Array
    divergence_max: Array
    pressure_mean: Array
    boundary_normal_flux_l1: Array
    net_boundary_flux: Array


class CavitySteadyDiagnostics(NamedTuple):
    nonlinear_residual_rms_history: Array
    nonlinear_residual_max_history: Array
    pressure_residual_abs_history: Array
    pressure_residual_rel_history: Array
    divergence_rms_history: Array
    divergence_max_history: Array
    pressure_mean_history: Array
    boundary_normal_flux_l1_history: Array
    net_boundary_flux_history: Array
    active_history: Array
    initial_residual_rms: Array
    final_residual_rms: Array
    final_residual_max: Array
    converged: Array
    iterations: Array


def create_cavity_initial_state(
    grid: ChannelGrid2D, *, dtype=jnp.float32
) -> ChannelState:
    """Return a quiescent state with the closed-domain normal BCs applied."""

    u = jnp.zeros((grid.nx + 1, grid.ny), dtype=dtype)
    v = jnp.zeros((grid.nx, grid.ny + 1), dtype=dtype)
    p = jnp.zeros((grid.nx, grid.ny), dtype=dtype)
    u, v = apply_cavity_normal_boundaries(u, v, grid)
    return ChannelState(u, v, p)


def apply_cavity_normal_boundaries(
    u: Array, v: Array, grid: ChannelGrid2D
) -> tuple[Array, Array]:
    """Set zero normal velocity on every wall of the closed cavity."""

    if u.shape != (grid.nx + 1, grid.ny):
        raise ValueError(
            f"u must have shape {(grid.nx + 1, grid.ny)}, got {u.shape}"
        )
    if v.shape != (grid.nx, grid.ny + 1):
        raise ValueError(
            f"v must have shape {(grid.nx, grid.ny + 1)}, got {v.shape}"
        )
    u = u.at[0, :].set(0.0)
    u = u.at[-1, :].set(0.0)
    v = v.at[:, 0].set(0.0)
    v = v.at[:, -1].set(0.0)
    return u, v


def cavity_pressure_gradient(
    p: Array, grid: ChannelGrid2D
) -> tuple[Array, Array]:
    """Return face pressure gradients with Neumann data on all walls."""

    if p.shape != (grid.nx, grid.ny):
        raise ValueError(f"p must have shape {(grid.nx, grid.ny)}, got {p.shape}")
    grad_u = jnp.zeros((grid.nx + 1, grid.ny), dtype=p.dtype)
    grad_v = jnp.zeros((grid.nx, grid.ny + 1), dtype=p.dtype)
    grad_u = grad_u.at[1 : grid.nx, :].set(
        (p[1:, :] - p[:-1, :]) / grid.dx
    )
    grad_v = grad_v.at[:, 1 : grid.ny].set(
        (p[:, 1:] - p[:, :-1]) / grid.dy
    )
    return grad_u, grad_v


def cavity_weighted_pressure_operator(
    p: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: ChannelGrid2D,
) -> Array:
    """Apply the SPD zero-mean-gauge operator ``-D M G + P_const``."""

    grad_u, grad_v = cavity_pressure_gradient(p, grid)
    compatible = -divergence(
        mobility_u * grad_u, mobility_v * grad_v, grid
    )
    return compatible + jnp.mean(p)


def solve_cavity_pressure(
    rhs: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> tuple[Array, Array, Array]:
    """Solve the closed-domain pressure correction with a zero-mean gauge."""

    compatible_rhs = rhs - jnp.mean(rhs)
    operator = lambda p: cavity_weighted_pressure_operator(
        p, mobility_u, mobility_v, grid
    )
    p = _forward_conjugate_gradient(
        operator,
        compatible_rhs,
        tolerance=params.pressure_tolerance,
        atol=params.pressure_atol,
        maxiter=params.pressure_maxiter,
    )
    p = p - jnp.mean(p)
    residual = operator(p) - compatible_rhs
    residual_abs = jnp.linalg.norm(residual.reshape(-1))
    rhs_norm = jnp.linalg.norm(compatible_rhs.reshape(-1))
    residual_rel = residual_abs / jnp.maximum(
        rhs_norm, jnp.finfo(rhs.dtype).tiny
    )
    return p, residual_abs, residual_rel


def cavity_laplacian_u(
    u: Array, grid: ChannelGrid2D, walls: CavityWalls
) -> Array:
    """Laplacian for x velocity with wall values imposed by odd reflection."""

    lap_x = jnp.zeros_like(u)
    lap_x = lap_x.at[1 : grid.nx, :].set(
        (u[2:, :] - 2.0 * u[1 : grid.nx, :] + u[: grid.nx - 1, :])
        / (grid.dx * grid.dx)
    )
    lap_y = jnp.zeros_like(u)
    lap_y = lap_y.at[:, 1 : grid.ny - 1].set(
        (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2])
        / (grid.dy * grid.dy)
    )
    lap_y = lap_y.at[:, 0].set(
        (u[:, 1] - 3.0 * u[:, 0] + 2.0 * walls.bottom_u)
        / (grid.dy * grid.dy)
    )
    lap_y = lap_y.at[:, -1].set(
        (u[:, -2] - 3.0 * u[:, -1] + 2.0 * walls.top_u)
        / (grid.dy * grid.dy)
    )
    lap = lap_x + lap_y
    lap = lap.at[0, :].set(0.0)
    return lap.at[-1, :].set(0.0)


def cavity_laplacian_v(
    v: Array, grid: ChannelGrid2D, walls: CavityWalls
) -> Array:
    """Laplacian for y velocity with tangential side-wall Dirichlet data."""

    lap_x = jnp.zeros_like(v)
    lap_x = lap_x.at[1 : grid.nx - 1, :].set(
        (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :])
        / (grid.dx * grid.dx)
    )
    lap_x = lap_x.at[0, :].set(
        (v[1, :] - 3.0 * v[0, :] + 2.0 * walls.left_v)
        / (grid.dx * grid.dx)
    )
    lap_x = lap_x.at[-1, :].set(
        (v[-2, :] - 3.0 * v[-1, :] + 2.0 * walls.right_v)
        / (grid.dx * grid.dx)
    )
    lap_y = jnp.zeros_like(v)
    lap_y = lap_y.at[:, 1 : grid.ny].set(
        (v[:, 2:] - 2.0 * v[:, 1 : grid.ny] + v[:, : grid.ny - 1])
        / (grid.dy * grid.dy)
    )
    lap = lap_x + lap_y
    lap = lap.at[:, 0].set(0.0)
    return lap.at[:, -1].set(0.0)


def cavity_conservative_advection(
    u: Array, v: Array, grid: ChannelGrid2D, *, scheme: str = "donor-cell"
) -> tuple[Array, Array]:
    """Conservative advection with all normal wall fluxes zero."""

    u, v = apply_cavity_normal_boundaries(u, v, grid)
    adv_u, adv_v = conservative_advection(u, v, grid, scheme=scheme)
    adv_u = adv_u.at[0, :].set(0.0)
    adv_u = adv_u.at[-1, :].set(0.0)
    adv_v = adv_v.at[:, 0].set(0.0)
    adv_v = adv_v.at[:, -1].set(0.0)
    return adv_u, adv_v


def cavity_flux_diagnostics(
    state: ChannelState, grid: ChannelGrid2D
) -> tuple[Array, Array, Array, Array, Array]:
    """Return divergence, pressure gauge, and closed-boundary flux metrics."""

    div = divergence(state.u, state.v, grid)
    div_rms = jnp.sqrt(jnp.mean(div * div))
    div_max = jnp.max(jnp.abs(div))
    left = jnp.sum(state.u[0, :]) * grid.dy
    right = jnp.sum(state.u[-1, :]) * grid.dy
    bottom = jnp.sum(state.v[:, 0]) * grid.dx
    top = jnp.sum(state.v[:, -1]) * grid.dx
    net_flux = right - left + top - bottom
    normal_flux_l1 = (
        jnp.sum(jnp.abs(state.u[0, :])) * grid.dy
        + jnp.sum(jnp.abs(state.u[-1, :])) * grid.dy
        + jnp.sum(jnp.abs(state.v[:, 0])) * grid.dx
        + jnp.sum(jnp.abs(state.v[:, -1])) * grid.dx
    )
    return div_rms, div_max, jnp.mean(state.p), normal_flux_l1, net_flux


def project_cavity_velocity(
    u_tilde: Array,
    v_tilde: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> tuple[ChannelState, CavityProjectionDiagnostics]:
    """Project a tentative field in a fully closed rectangular domain."""

    if params.dt <= 0.0:
        raise ValueError("dt must be positive")
    u_tilde, v_tilde = apply_cavity_normal_boundaries(
        u_tilde, v_tilde, grid
    )
    rhs = -divergence(u_tilde, v_tilde, grid) / params.dt
    p, pressure_abs, pressure_rel = solve_cavity_pressure(
        rhs, mobility_u, mobility_v, grid, params
    )
    grad_u, grad_v = cavity_pressure_gradient(p, grid)
    u = u_tilde - params.dt * mobility_u * grad_u
    v = v_tilde - params.dt * mobility_v * grad_v
    u, v = apply_cavity_normal_boundaries(u, v, grid)
    state = ChannelState(u, v, p)
    div_rms, div_max, p_mean, flux_l1, net_flux = cavity_flux_diagnostics(
        state, grid
    )
    return state, CavityProjectionDiagnostics(
        pressure_abs,
        pressure_rel,
        div_rms,
        div_max,
        p_mean,
        flux_l1,
        net_flux,
    )


def cavity_step(
    state: ChannelState,
    alpha: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    walls: CavityWalls,
) -> tuple[ChannelState, CavityProjectionDiagnostics]:
    """Advance one explicit pseudo-time step for the closed cavity."""

    if params.viscosity < 0.0 or params.dt <= 0.0:
        raise ValueError("viscosity must be nonnegative and dt positive")
    u, v = apply_cavity_normal_boundaries(state.u, state.v, grid)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    mobility_u = 1.0 / (1.0 + params.dt * alpha_u)
    mobility_v = 1.0 / (1.0 + params.dt * alpha_v)
    if params.include_advection:
        adv_u, adv_v = cavity_conservative_advection(
            u, v, grid, scheme=params.advection_scheme
        )
    else:
        adv_u = jnp.zeros_like(u)
        adv_v = jnp.zeros_like(v)
    rhs_u = u + params.dt * (
        -adv_u
        + params.viscosity * cavity_laplacian_u(u, grid, walls)
        + params.force_x
    )
    rhs_v = v + params.dt * (
        -adv_v
        + params.viscosity * cavity_laplacian_v(v, grid, walls)
        + params.force_y
    )
    return project_cavity_velocity(
        mobility_u * rhs_u,
        mobility_v * rhs_v,
        mobility_u,
        mobility_v,
        grid,
        params,
    )


def cavity_steady_residual(
    state: ChannelState,
    alpha: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    walls: CavityWalls,
) -> ChannelResidual:
    """Return the complete steady closed-cavity discrete residual."""

    u, v = apply_cavity_normal_boundaries(state.u, state.v, grid)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    grad_u, grad_v = cavity_pressure_gradient(state.p, grid)
    if params.include_advection:
        adv_u, adv_v = cavity_conservative_advection(
            u, v, grid, scheme=params.advection_scheme
        )
    else:
        adv_u = jnp.zeros_like(u)
        adv_v = jnp.zeros_like(v)
    residual_u = (
        adv_u
        - params.viscosity * cavity_laplacian_u(u, grid, walls)
        + alpha_u * u
        + grad_u
        - params.force_x
    )
    residual_v = (
        adv_v
        - params.viscosity * cavity_laplacian_v(v, grid, walls)
        + alpha_v * v
        + grad_v
        - params.force_y
    )
    residual_u = residual_u.at[0, :].set(state.u[0, :])
    residual_u = residual_u.at[-1, :].set(state.u[-1, :])
    residual_v = residual_v.at[:, 0].set(state.v[:, 0])
    residual_v = residual_v.at[:, -1].set(state.v[:, -1])
    return ChannelResidual(residual_u, residual_v, divergence(u, v, grid))


def solve_cavity_steady(
    state0: ChannelState,
    alpha: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    walls: CavityWalls,
    max_steps: int,
) -> tuple[ChannelState, CavitySteadyDiagnostics]:
    """Solve the steady cavity by CUDA-safe residual-controlled marching."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if params.nonlinear_rtol < 0.0 or params.nonlinear_atol < 0.0:
        raise ValueError("nonlinear tolerances must be nonnegative")
    if params.nonlinear_min_steps < 1:
        raise ValueError("nonlinear_min_steps must be at least one")

    state0 = ChannelState(
        *apply_cavity_normal_boundaries(state0.u, state0.v, grid),
        state0.p - jnp.mean(state0.p),
    )
    initial_residual = cavity_steady_residual(
        state0, alpha, grid, params, walls
    )
    initial_rms = residual_rms(initial_residual)
    threshold = jnp.maximum(
        jnp.asarray(params.nonlinear_atol, dtype=initial_rms.dtype),
        jnp.asarray(params.nonlinear_rtol, dtype=initial_rms.dtype) * initial_rms,
    )
    history_float0 = jnp.zeros((max_steps,), dtype=initial_rms.dtype)
    history_bool0 = jnp.zeros((max_steps,), dtype=jnp.bool_)

    def condition(carry):
        _, iteration, done, *_ = carry
        return (iteration < max_steps) & jnp.logical_not(done)

    def body(carry):
        (
            state,
            iteration,
            _,
            nonlinear_rms_history,
            nonlinear_max_history,
            pressure_abs_history,
            pressure_rel_history,
            divergence_rms_history,
            divergence_max_history,
            pressure_mean_history,
            boundary_flux_history,
            net_flux_history,
            active_history,
        ) = carry
        candidate, projection = cavity_step(
            state, alpha, grid, params, walls
        )
        residual = cavity_steady_residual(
            candidate, alpha, grid, params, walls
        )
        nonlinear_rms = residual_rms(residual)
        nonlinear_max = residual_max(residual)
        eligible = iteration + 1 >= params.nonlinear_min_steps
        next_done = eligible & (nonlinear_rms <= threshold)
        return (
            candidate,
            iteration + 1,
            next_done,
            nonlinear_rms_history.at[iteration].set(nonlinear_rms),
            nonlinear_max_history.at[iteration].set(nonlinear_max),
            pressure_abs_history.at[iteration].set(
                projection.pressure_residual_abs
            ),
            pressure_rel_history.at[iteration].set(
                projection.pressure_residual_rel
            ),
            divergence_rms_history.at[iteration].set(
                projection.divergence_rms
            ),
            divergence_max_history.at[iteration].set(
                projection.divergence_max
            ),
            pressure_mean_history.at[iteration].set(
                projection.pressure_mean
            ),
            boundary_flux_history.at[iteration].set(
                projection.boundary_normal_flux_l1
            ),
            net_flux_history.at[iteration].set(projection.net_boundary_flux),
            active_history.at[iteration].set(True),
        )

    (
        final_state,
        iteration_count,
        converged,
        nonlinear_rms_history,
        nonlinear_max_history,
        pressure_abs_history,
        pressure_rel_history,
        divergence_rms_history,
        divergence_max_history,
        pressure_mean_history,
        boundary_flux_history,
        net_flux_history,
        active_history,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            state0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_float0,
            history_bool0,
        ),
    )
    final_residual = cavity_steady_residual(
        final_state, alpha, grid, params, walls
    )
    diagnostics = CavitySteadyDiagnostics(
        nonlinear_rms_history,
        nonlinear_max_history,
        pressure_abs_history,
        pressure_rel_history,
        divergence_rms_history,
        divergence_max_history,
        pressure_mean_history,
        boundary_flux_history,
        net_flux_history,
        active_history,
        initial_rms,
        residual_rms(final_residual),
        residual_max(final_residual),
        converged,
        iteration_count,
    )
    return final_state, diagnostics
