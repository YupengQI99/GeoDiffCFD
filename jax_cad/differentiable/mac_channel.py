"""Compatible non-periodic 2-D MAC Navier--Stokes--Brinkman kernel.

Pressure is cell centred, streamwise velocity is stored on vertical faces, and
transverse velocity on horizontal faces. The pressure correction uses the same
discrete divergence and gradient as continuity, with homogeneous correction at
prescribed-velocity boundaries and a zero-pressure part of the outlet flux.
Together with the zero viscous-flux part, the combined steady finite-volume
residual has the DFG do-nothing condition ``nu du/dn - p n = 0`` at the outlet.
With positive Brinkman mobility, the operator A = -D M G is symmetric positive
definite and is solved matrix-free by conjugate gradients. Nonlinear transport
uses conservative donor-cell fluxes by default; an opt-in central scheme is
available for spatial-error diagnostics on the staggered momentum volumes.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


class ChannelGrid2D(NamedTuple):
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float
    dy: float


class ChannelState(NamedTuple):
    u: Array  # vertical faces, shape (nx + 1, ny)
    v: Array  # horizontal faces, shape (nx, ny + 1)
    p: Array  # cell centres, shape (nx, ny)


class FaceBrinkmanCoefficient(NamedTuple):
    """Explicit native coefficients shared by stepping, residuals and forces.

    This opt-in representation removes the implicit cell-to-face convolution.
    It is a JAX pytree, so geometry derivatives and argument-passing compilation
    work exactly as for the legacy cell array. Callers must supply nonnegative
    coefficients; the cylinder driver constructs them from the bounded mask.
    """

    u: Array  # (nx + 1, ny)
    v: Array  # (nx, ny + 1)


def sample_native_face_coefficient(coefficient, grid: ChannelGrid2D, *, dtype=jnp.float64):
    """Sample one continuous scalar material field at the two MAC lattices.

    This is point sampling, not an exact volume-average or sharp-wall treatment.
    No geometric shift, smoothing width, penalty strength or force estimator is
    changed here. In particular, it does not remove finite-penalty model error.
    """
    xu, yu = jnp.meshgrid(jnp.arange(grid.nx + 1, dtype=dtype)*grid.dx,
                         (jnp.arange(grid.ny, dtype=dtype)+.5)*grid.dy, indexing="ij")
    xv, yv = jnp.meshgrid((jnp.arange(grid.nx, dtype=dtype)+.5)*grid.dx,
                         jnp.arange(grid.ny + 1, dtype=dtype)*grid.dy, indexing="ij")
    return FaceBrinkmanCoefficient(coefficient(xu, yu), coefficient(xv, yv))


class ChannelParams(NamedTuple):
    viscosity: float
    dt: float
    force_x: float = 0.0
    force_y: float = 0.0
    pressure_tolerance: float = 1.0e-10
    pressure_atol: float = 0.0
    pressure_maxiter: int = 500
    include_advection: bool = False
    nonlinear_rtol: float = 1.0e-8
    nonlinear_atol: float = 1.0e-10
    nonlinear_min_steps: int = 1
    advection_scheme: str = "donor-cell"
    pressure_preconditioner: str = "none"


class ChannelDiagnostics(NamedTuple):
    pressure_residual_abs: Array
    pressure_residual_rel: Array
    divergence_rms: Array
    divergence_max: Array
    inlet_flux: Array
    outlet_flux: Array
    mass_flux_mismatch: Array


class ChannelResidual(NamedTuple):
    momentum_u: Array
    momentum_v: Array
    continuity: Array


class MomentumFluxes(NamedTuple):
    uu: Array  # u-momentum x-flux, shape (nx + 1, ny)
    uv_u: Array  # u-momentum y-flux, shape (nx + 1, ny + 1)
    uv_v: Array  # v-momentum x-flux, shape (nx + 1, ny + 1)
    vv: Array  # v-momentum y-flux, shape (nx, ny + 1)


class SteadySolveDiagnostics(NamedTuple):
    nonlinear_residual_rms_history: Array
    nonlinear_residual_max_history: Array
    pressure_residual_abs_history: Array
    pressure_residual_rel_history: Array
    divergence_rms_history: Array
    divergence_max_history: Array
    inlet_flux_history: Array
    outlet_flux_history: Array
    mass_flux_mismatch_history: Array
    active_history: Array
    initial_residual_rms: Array
    final_residual_rms: Array
    final_residual_max: Array
    converged: Array
    iterations: Array


def create_channel_grid(nx: int, ny: int, lx: float, ly: float) -> ChannelGrid2D:
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must both be at least 3")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("lx and ly must be positive")
    return ChannelGrid2D(nx, ny, lx, ly, lx / nx, ly / ny)


def cell_centres(grid: ChannelGrid2D, *, dtype=jnp.float32) -> tuple[Array, Array]:
    x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
    y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
    return jnp.meshgrid(x, y, indexing="ij")


def parabolic_inlet(
    grid: ChannelGrid2D, mean_velocity: float, *, dtype=jnp.float32
) -> Array:
    """Return a parabolic profile normalized to the requested discrete flux."""

    y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
    raw = y * (grid.ly - y)
    target_flux = mean_velocity * grid.ly
    return raw * target_flux / (jnp.sum(raw) * grid.dy)


def apply_velocity_boundaries(
    u: Array, v: Array, inlet_u: Array, grid: ChannelGrid2D
) -> tuple[Array, Array]:
    if inlet_u.shape != (grid.ny,):
        raise ValueError(f"inlet_u must have shape {(grid.ny,)}, got {inlet_u.shape}")
    u = u.at[0, :].set(inlet_u)
    v = v.at[:, 0].set(0.0)
    v = v.at[:, -1].set(0.0)
    return u, v


def conservative_momentum_fluxes(
    u: Array, v: Array, grid: ChannelGrid2D, *, scheme: str = "donor-cell"
) -> MomentumFluxes:
    """Conservative donor-cell (default) or centred momentum fluxes.

    Central interpolation removes the first-order upwind dissipation and is
    second-order in the smooth interior. It is not a bounded/TVD scheme and
    requires its own viscous/time-step stability and boundary-error studies.
    The scheme is a static configuration choice, not a design parameter.
    """

    if scheme not in ("donor-cell", "central"):
        raise ValueError(f"unsupported advection scheme: {scheme!r}")

    def transported(speed, left, right):
        if scheme == "central":
            return 0.5 * (left + right)
        return jnp.where(speed >= 0.0, left, right)

    if u.shape != (grid.nx + 1, grid.ny):
        raise ValueError(
            f"u must have shape {(grid.nx + 1, grid.ny)}, got {u.shape}"
        )
    if v.shape != (grid.nx, grid.ny + 1):
        raise ValueError(
            f"v must have shape {(grid.nx, grid.ny + 1)}, got {v.shape}"
        )

    # Streamwise transport of u through pressure-cell centres and the outlet.
    mass_u = 0.5 * (u[:-1, :] + u[1:, :])
    upwind_u = transported(mass_u, u[:-1, :], u[1:, :])
    uu = jnp.zeros((grid.nx + 1, grid.ny), dtype=u.dtype)
    uu = uu.at[: grid.nx, :].set(mass_u * upwind_u)
    uu = uu.at[grid.nx, :].set(u[grid.nx, :] * u[grid.nx, :])

    # Transverse transport of u through MAC corners. Tangential velocity is
    # zero at the inlet and v is extrapolated with zero gradient at the outlet.
    mass_v_corner = jnp.zeros(
        (grid.nx + 1, grid.ny + 1), dtype=v.dtype
    )
    mass_v_corner = mass_v_corner.at[1 : grid.nx, 1 : grid.ny].set(
        0.5 * (v[:-1, 1 : grid.ny] + v[1:, 1 : grid.ny])
    )
    mass_v_corner = mass_v_corner.at[grid.nx, 1 : grid.ny].set(
        v[-1, 1 : grid.ny]
    )
    u_below = jnp.zeros_like(mass_v_corner)
    u_above = jnp.zeros_like(mass_v_corner)
    u_below = u_below.at[:, 1 : grid.ny].set(u[:, : grid.ny - 1])
    u_above = u_above.at[:, 1 : grid.ny].set(u[:, 1:])
    uv_u = mass_v_corner * transported(mass_v_corner, u_below, u_above)

    # Streamwise transport of v through the same geometric corner locations.
    mass_u_corner = jnp.zeros(
        (grid.nx + 1, grid.ny + 1), dtype=u.dtype
    )
    mass_u_corner = mass_u_corner.at[:, 1 : grid.ny].set(
        0.5 * (u[:, : grid.ny - 1] + u[:, 1:])
    )
    v_left = jnp.zeros_like(mass_u_corner)
    v_right = jnp.zeros_like(mass_u_corner)
    v_right = v_right.at[0, :].set(v[0, :])
    v_left = v_left.at[1 : grid.nx, :].set(v[:-1, :])
    v_right = v_right.at[1 : grid.nx, :].set(v[1:, :])
    v_left = v_left.at[grid.nx, :].set(v[-1, :])
    v_right = v_right.at[grid.nx, :].set(v[-1, :])
    uv_v = mass_u_corner * transported(mass_u_corner, v_left, v_right)
    if scheme == "central":
        # v=0 is prescribed AT the inlet, not at a ghost cell centre.
        # Averaging zero with v[0] would introduce a spurious transverse
        # momentum flux through this boundary. Equivalently use an odd ghost.
        uv_v = uv_v.at[0, :].set(0.0)

    # Transverse transport of v through pressure-cell centres.
    mass_v = 0.5 * (v[:, :-1] + v[:, 1:])
    upwind_v = transported(mass_v, v[:, :-1], v[:, 1:])
    vv = jnp.zeros((grid.nx, grid.ny + 1), dtype=v.dtype)
    vv = vv.at[:, : grid.ny].set(mass_v * upwind_v)
    return MomentumFluxes(uu, uv_u, uv_v, vv)


def conservative_advection(
    u: Array, v: Array, grid: ChannelGrid2D, *, scheme: str = "donor-cell"
) -> tuple[Array, Array]:
    """Return the divergence of the selected conservative momentum fluxes."""

    fluxes = conservative_momentum_fluxes(u, v, grid, scheme=scheme)

    adv_u = jnp.zeros_like(u)
    streamwise_u = (fluxes.uu[1:, :] - fluxes.uu[:-1, :]) / grid.dx
    # The outlet u control volume has half the interior streamwise width.
    streamwise_u = streamwise_u.at[-1, :].set(2.0 * streamwise_u[-1, :])
    adv_u = adv_u.at[1:, :].set(streamwise_u)
    adv_u = adv_u + (
        fluxes.uv_u[:, 1:] - fluxes.uv_u[:, :-1]
    ) / grid.dy
    adv_u = adv_u.at[0, :].set(0.0)

    adv_v = (fluxes.uv_v[1:, :] - fluxes.uv_v[:-1, :]) / grid.dx
    transverse_v = jnp.zeros_like(v)
    transverse_v = transverse_v.at[:, 1 : grid.ny].set(
        (fluxes.vv[:, 1 : grid.ny] - fluxes.vv[:, : grid.ny - 1])
        / grid.dy
    )
    adv_v = adv_v + transverse_v
    adv_v = adv_v.at[:, 0].set(0.0)
    adv_v = adv_v.at[:, -1].set(0.0)
    return adv_u, adv_v


def create_initial_state(
    grid: ChannelGrid2D, inlet_u: Array, *, dtype=jnp.float32
) -> ChannelState:
    u = jnp.zeros((grid.nx + 1, grid.ny), dtype=dtype)
    v = jnp.zeros((grid.nx, grid.ny + 1), dtype=dtype)
    u, v = apply_velocity_boundaries(u, v, inlet_u.astype(dtype), grid)
    p = jnp.zeros((grid.nx, grid.ny), dtype=dtype)
    return ChannelState(u, v, p)


def divergence(u: Array, v: Array, grid: ChannelGrid2D) -> Array:
    return (u[1:, :] - u[:-1, :]) / grid.dx + (
        v[:, 1:] - v[:, :-1]
    ) / grid.dy


def pressure_gradient(p: Array, grid: ChannelGrid2D) -> tuple[Array, Array]:
    """Face gradients for Neumann inlet/walls and zero outlet pressure flux.

    The outlet pressure contribution must be interpreted together with the
    zero outlet viscous flux in :func:`laplacian_u`; their sum is the zero-total-
    traction natural boundary flux in the steady momentum residual.
    """

    if p.shape != (grid.nx, grid.ny):
        raise ValueError(f"p must have shape {(grid.nx, grid.ny)}, got {p.shape}")
    grad_u = jnp.zeros((grid.nx + 1, grid.ny), dtype=p.dtype)
    grad_v = jnp.zeros((grid.nx, grid.ny + 1), dtype=p.dtype)
    grad_u = grad_u.at[1:grid.nx, :].set((p[1:, :] - p[:-1, :]) / grid.dx)
    # The distance from the final cell centre to the outlet face is dx/2.
    grad_u = grad_u.at[grid.nx, :].set(-2.0 * p[-1, :] / grid.dx)
    grad_v = grad_v.at[:, 1:grid.ny].set((p[:, 1:] - p[:, :-1]) / grid.dy)
    return grad_u, grad_v


def interpolate_alpha_to_faces(
    alpha: Array | FaceBrinkmanCoefficient, grid: ChannelGrid2D
) -> tuple[Array, Array]:
    if isinstance(alpha, FaceBrinkmanCoefficient):
        if alpha.u.shape != (grid.nx + 1, grid.ny) or alpha.v.shape != (grid.nx, grid.ny + 1):
            raise ValueError("native Brinkman coefficients have incorrect MAC shapes")
        return alpha.u, alpha.v
    if alpha.shape != (grid.nx, grid.ny):
        raise ValueError(f"alpha must have shape {(grid.nx, grid.ny)}, got {alpha.shape}")
    alpha_u = jnp.zeros((grid.nx + 1, grid.ny), dtype=alpha.dtype)
    alpha_v = jnp.zeros((grid.nx, grid.ny + 1), dtype=alpha.dtype)
    alpha_u = alpha_u.at[0, :].set(alpha[0, :])
    alpha_u = alpha_u.at[1:grid.nx, :].set(
        0.5 * (alpha[:-1, :] + alpha[1:, :])
    )
    alpha_u = alpha_u.at[grid.nx, :].set(alpha[-1, :])
    alpha_v = alpha_v.at[:, 0].set(alpha[:, 0])
    alpha_v = alpha_v.at[:, 1:grid.ny].set(
        0.5 * (alpha[:, :-1] + alpha[:, 1:])
    )
    alpha_v = alpha_v.at[:, grid.ny].set(alpha[:, -1])
    return alpha_u, alpha_v


def weighted_pressure_operator(
    p: Array, mobility_u: Array, mobility_v: Array, grid: ChannelGrid2D
) -> Array:
    """Apply ``-D M G`` without assembling a pressure matrix."""

    grad_u, grad_v = pressure_gradient(p, grid)
    return -divergence(mobility_u * grad_u, mobility_v * grad_v, grid)


def _forward_conjugate_gradient(
    operator,
    rhs: Array,
    *,
    tolerance: float,
    atol: float,
    maxiter: int,
) -> Array:
    """Solve an SPD system without a ``custom_linear_solve`` wrapper.

    JAX 0.7's CUDA lowering can return a stale primal from
    ``custom_linear_solve`` when its CG solve is nested in ``lax.scan`` even
    though the accompanying recursive-residual diagnostic is small.  The
    explicit forward loop avoids that aliasing path.  Derivatives are supplied
    separately by :func:`_channel_pressure_solution` below.
    """

    x0 = jnp.zeros_like(rhs)
    residual0 = rhs - operator(x0)
    direction0 = residual0
    squared_residual0 = jnp.real(jnp.vdot(residual0, residual0))
    squared_rhs = jnp.real(jnp.vdot(rhs, rhs))
    threshold_squared = jnp.maximum(
        jnp.asarray(tolerance, dtype=rhs.dtype) ** 2 * squared_rhs,
        jnp.asarray(atol, dtype=rhs.dtype) ** 2,
    )

    def condition(value):
        _, _, _, squared_residual, iteration = value
        return (squared_residual > threshold_squared) & (iteration < maxiter)

    def body(value):
        solution, residual, direction, squared_residual, iteration = value
        operator_direction = operator(direction)
        step_size = squared_residual / jnp.real(
            jnp.vdot(direction, operator_direction)
        )
        next_solution = solution + step_size * direction
        next_residual = residual - step_size * operator_direction
        next_squared_residual = jnp.real(
            jnp.vdot(next_residual, next_residual)
        )
        beta = next_squared_residual / squared_residual
        next_direction = next_residual + beta * direction
        return (
            next_solution,
            next_residual,
            next_direction,
            next_squared_residual,
            iteration + 1,
        )

    solution, _, _, _, _ = jax.lax.while_loop(
        condition,
        body,
        (
            x0,
            residual0,
            direction0,
            squared_residual0,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return solution


def _pressure_linear_solution(operator, rhs, grid, tolerance, atol, maxiter, preconditioner):
    if preconditioner == "none":
        return _forward_conjugate_gradient(operator, rhs, tolerance=tolerance, atol=atol, maxiter=maxiter)
    if preconditioner == "constant-poisson-dct":
        from .mac_pressure_preconditioner import preconditioned_cg
        return preconditioned_cg(operator, rhs, grid, tolerance=tolerance, atol=atol, maxiter=maxiter)[0]
    raise ValueError("unknown pressure preconditioner")


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _channel_pressure_solution(
    rhs: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: ChannelGrid2D,
    tolerance: float,
    atol: float,
    maxiter: int,
    preconditioner: str = "none",
) -> Array:
    operator = lambda p: weighted_pressure_operator(
        p, mobility_u, mobility_v, grid
    )
    return _pressure_linear_solution(operator, rhs, grid, tolerance, atol, maxiter, preconditioner)


def _channel_pressure_solution_fwd(
    rhs,
    mobility_u,
    mobility_v,
    grid,
    tolerance,
    atol,
    maxiter,
    preconditioner,
):
    pressure = _channel_pressure_solution(
        rhs,
        mobility_u,
        mobility_v,
        grid,
        tolerance,
        atol,
        maxiter,
        preconditioner,
    )
    return pressure, (pressure, mobility_u, mobility_v)


def _channel_pressure_solution_bwd(
    grid,
    tolerance,
    atol,
    maxiter,
    preconditioner,
    residuals,
    pressure_cotangent,
):
    pressure, mobility_u, mobility_v = residuals
    operator = lambda p: weighted_pressure_operator(
        p, mobility_u, mobility_v, grid
    )
    rhs_cotangent = _pressure_linear_solution(
        operator, pressure_cotangent, grid, tolerance, atol, maxiter, preconditioner)

    def operator_from_mobility(candidate_u, candidate_v):
        return weighted_pressure_operator(
            pressure, candidate_u, candidate_v, grid
        )

    _, mobility_pullback = jax.vjp(
        operator_from_mobility, mobility_u, mobility_v
    )
    mobility_u_cotangent, mobility_v_cotangent = mobility_pullback(
        -rhs_cotangent
    )
    return rhs_cotangent, mobility_u_cotangent, mobility_v_cotangent


_channel_pressure_solution.defvjp(
    _channel_pressure_solution_fwd,
    _channel_pressure_solution_bwd,
)


def solve_pressure(
    rhs: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> tuple[Array, Array, Array]:
    operator = lambda p: weighted_pressure_operator(p, mobility_u, mobility_v, grid)
    p = _channel_pressure_solution(
        rhs,
        mobility_u,
        mobility_v,
        grid,
        params.pressure_tolerance,
        params.pressure_atol,
        params.pressure_maxiter,
        params.pressure_preconditioner,
    )
    residual = operator(p) - rhs
    residual_abs = jnp.linalg.norm(residual.reshape(-1))
    rhs_norm = jnp.linalg.norm(rhs.reshape(-1))
    residual_rel = residual_abs / jnp.maximum(
        rhs_norm, jnp.finfo(rhs.dtype).tiny
    )
    return p, residual_abs, residual_rel


def flux_diagnostics(
    state: ChannelState, grid: ChannelGrid2D
) -> tuple[Array, Array, Array, Array, Array]:
    div = divergence(state.u, state.v, grid)
    divergence_rms = jnp.sqrt(jnp.mean(div * div))
    divergence_max = jnp.max(jnp.abs(div))
    inlet_flux = jnp.sum(state.u[0, :]) * grid.dy
    outlet_flux = jnp.sum(state.u[-1, :]) * grid.dy
    scale = jnp.maximum(jnp.abs(inlet_flux), jnp.finfo(state.u.dtype).tiny)
    mismatch = jnp.abs(outlet_flux - inlet_flux) / scale
    return divergence_rms, divergence_max, inlet_flux, outlet_flux, mismatch


def project_velocity(
    u_tilde: Array,
    v_tilde: Array,
    mobility_u: Array,
    mobility_v: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> tuple[ChannelState, ChannelDiagnostics]:
    """Project with prescribed inlet/walls and pressure-controlled outlet."""

    if params.dt <= 0.0:
        raise ValueError("dt must be positive")
    u_tilde, v_tilde = apply_velocity_boundaries(
        u_tilde, v_tilde, inlet_u, grid
    )
    rhs = -divergence(u_tilde, v_tilde, grid) / params.dt
    p, pressure_abs, pressure_rel = solve_pressure(
        rhs, mobility_u, mobility_v, grid, params
    )
    grad_u, grad_v = pressure_gradient(p, grid)
    u = u_tilde - params.dt * mobility_u * grad_u
    v = v_tilde - params.dt * mobility_v * grad_v
    u, v = apply_velocity_boundaries(u, v, inlet_u, grid)
    state = ChannelState(u, v, p)
    div_rms, div_max, inlet_flux, outlet_flux, mismatch = flux_diagnostics(
        state, grid
    )
    return state, ChannelDiagnostics(
        pressure_abs,
        pressure_rel,
        div_rms,
        div_max,
        inlet_flux,
        outlet_flux,
        mismatch,
    )


def laplacian_u(u: Array, grid: ChannelGrid2D) -> Array:
    """Viscous Laplacian with walls and zero outlet viscous flux.

    At the outlet, this term and :func:`pressure_gradient` are two parts of a
    single zero-total-traction finite-volume flux.  They are not independent
    physical boundary constraints on the converged steady equations.
    """

    lap_x = jnp.zeros_like(u)
    lap_x = lap_x.at[1:grid.nx, :].set(
        (u[2:, :] - 2.0 * u[1:grid.nx, :] + u[: grid.nx - 1, :])
        / (grid.dx * grid.dx)
    )
    lap_x = lap_x.at[grid.nx, :].set(
        2.0 * (u[grid.nx - 1, :] - u[grid.nx, :])
        / (grid.dx * grid.dx)
    )
    lap_y = jnp.zeros_like(u)
    lap_y = lap_y.at[:, 1 : grid.ny - 1].set(
        (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2])
        / (grid.dy * grid.dy)
    )
    # Odd ghost reflection places u=0 at the horizontal wall.
    lap_y = lap_y.at[:, 0].set(
        (u[:, 1] - 3.0 * u[:, 0]) / (grid.dy * grid.dy)
    )
    lap_y = lap_y.at[:, -1].set(
        (u[:, -2] - 3.0 * u[:, -1]) / (grid.dy * grid.dy)
    )
    return (lap_x + lap_y).at[0, :].set(0.0)


def laplacian_v(v: Array, grid: ChannelGrid2D) -> Array:
    """Viscous Laplacian with inlet tangential no-slip and wall values fixed."""

    lap_x = jnp.zeros_like(v)
    lap_x = lap_x.at[1 : grid.nx - 1, :].set(
        (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :])
        / (grid.dx * grid.dx)
    )
    lap_x = lap_x.at[0, :].set(
        (v[1, :] - 3.0 * v[0, :]) / (grid.dx * grid.dx)
    )
    lap_x = lap_x.at[-1, :].set(
        (v[-2, :] - v[-1, :]) / (grid.dx * grid.dx)
    )
    lap_y = jnp.zeros_like(v)
    lap_y = lap_y.at[:, 1:grid.ny].set(
        (v[:, 2:] - 2.0 * v[:, 1:grid.ny] + v[:, : grid.ny - 1])
        / (grid.dy * grid.dy)
    )
    lap = (lap_x + lap_y).at[:, 0].set(0.0)
    return lap.at[:, -1].set(0.0)


def step(
    state: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> tuple[ChannelState, ChannelDiagnostics]:
    """Advance one variable-mobility projection step."""

    if params.viscosity < 0.0 or params.dt <= 0.0:
        raise ValueError("viscosity must be nonnegative and dt positive")
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    mobility_u = 1.0 / (1.0 + params.dt * alpha_u)
    mobility_v = 1.0 / (1.0 + params.dt * alpha_v)
    if params.include_advection:
        adv_u, adv_v = conservative_advection(
            state.u, state.v, grid, scheme=params.advection_scheme
        )
    else:
        adv_u = jnp.zeros_like(state.u)
        adv_v = jnp.zeros_like(state.v)
    rhs_u = state.u + params.dt * (
        -adv_u
        + params.viscosity * laplacian_u(state.u, grid)
        + params.force_x
    )
    rhs_v = state.v + params.dt * (
        -adv_v
        + params.viscosity * laplacian_v(state.v, grid)
        + params.force_y
    )
    return project_velocity(
        mobility_u * rhs_u,
        mobility_v * rhs_v,
        mobility_u,
        mobility_v,
        inlet_u,
        grid,
        params,
    )


def simulate(
    state0: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    n_steps: int,
) -> tuple[ChannelState, ChannelDiagnostics]:
    if n_steps < 1:
        raise ValueError("n_steps must be at least one")

    def body(state, _):
        next_state, diagnostics = step(state, alpha, inlet_u, grid, params)
        return next_state, diagnostics

    final_state, history = jax.lax.scan(body, state0, xs=None, length=n_steps)
    final_diagnostics = jax.tree_util.tree_map(lambda values: values[-1], history)
    return final_state, final_diagnostics


def steady_residual(
    state: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
) -> ChannelResidual:
    """Discrete steady Navier--Stokes--Brinkman residual."""

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    grad_u, grad_v = pressure_gradient(state.p, grid)
    if params.include_advection:
        adv_u, adv_v = conservative_advection(
            state.u, state.v, grid, scheme=params.advection_scheme
        )
    else:
        adv_u = jnp.zeros_like(state.u)
        adv_v = jnp.zeros_like(state.v)
    residual_u = (
        adv_u
        - params.viscosity * laplacian_u(state.u, grid)
        + alpha_u * state.u
        + grad_u
        - params.force_x
    )
    residual_v = (
        adv_v
        - params.viscosity * laplacian_v(state.v, grid)
        + alpha_v * state.v
        + grad_v
        - params.force_y
    )
    residual_u = residual_u.at[0, :].set(state.u[0, :] - inlet_u)
    residual_v = residual_v.at[:, 0].set(state.v[:, 0])
    residual_v = residual_v.at[:, -1].set(state.v[:, -1])
    return ChannelResidual(
        residual_u,
        residual_v,
        divergence(state.u, state.v, grid),
    )


def residual_rms(residual: ChannelResidual) -> Array:
    total = (
        jnp.sum(residual.momentum_u * residual.momentum_u)
        + jnp.sum(residual.momentum_v * residual.momentum_v)
        + jnp.sum(residual.continuity * residual.continuity)
    )
    count = (
        residual.momentum_u.size
        + residual.momentum_v.size
        + residual.continuity.size
    )
    return jnp.sqrt(total / count)


def residual_max(residual: ChannelResidual) -> Array:
    return jnp.maximum(
        jnp.max(jnp.abs(residual.momentum_u)),
        jnp.maximum(
            jnp.max(jnp.abs(residual.momentum_v)),
            jnp.max(jnp.abs(residual.continuity)),
        ),
    )


def solve_steady(
    state0: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    max_steps: int,
    *,
    threshold: Array | float | None = None,
) -> tuple[ChannelState, SteadySolveDiagnostics]:
    """Pseudo-time solve with residual stopping and fixed-size histories.

    A data-dependent outer ``while_loop`` is required for reliable CUDA
    execution: JAX 0.7 can return a stale inner-CG state when a pressure solve
    is nested in ``scan``/``fori_loop``.  This routine is therefore the
    forward-verification path.  Use :func:`simulate` for fixed-step unrolled
    reverse-mode work; residual-based implicit differentiation is handled by a
    separate work package.

    ``threshold`` overrides the stopping threshold ``max(atol, rtol * rms0)``
    computed from the residual of ``state0``; :func:`solve_steady_chunked` uses
    it to continue a solve from an intermediate state without moving the goal.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if params.nonlinear_rtol < 0.0 or params.nonlinear_atol < 0.0:
        raise ValueError("nonlinear tolerances must be nonnegative")
    if params.nonlinear_min_steps < 1:
        raise ValueError("nonlinear_min_steps must be at least one")

    initial_residual = steady_residual(state0, alpha, inlet_u, grid, params)
    initial_rms = residual_rms(initial_residual)
    if threshold is None:
        threshold = jnp.maximum(
            jnp.asarray(params.nonlinear_atol, dtype=initial_rms.dtype),
            jnp.asarray(params.nonlinear_rtol, dtype=initial_rms.dtype) * initial_rms,
        )
    else:
        threshold = jnp.asarray(threshold, dtype=initial_rms.dtype)
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
            inlet_flux_history,
            outlet_flux_history,
            mismatch_history,
            active_history,
        ) = carry
        candidate, projection = step(state, alpha, inlet_u, grid, params)
        residual = steady_residual(candidate, alpha, inlet_u, grid, params)
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
            inlet_flux_history.at[iteration].set(projection.inlet_flux),
            outlet_flux_history.at[iteration].set(projection.outlet_flux),
            mismatch_history.at[iteration].set(
                projection.mass_flux_mismatch
            ),
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
        inlet_flux_history,
        outlet_flux_history,
        mismatch_history,
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
    final_residual = steady_residual(
        final_state, alpha, inlet_u, grid, params
    )
    diagnostics = SteadySolveDiagnostics(
        nonlinear_rms_history,
        nonlinear_max_history,
        pressure_abs_history,
        pressure_rel_history,
        divergence_rms_history,
        divergence_max_history,
        inlet_flux_history,
        outlet_flux_history,
        mismatch_history,
        active_history,
        initial_rms,
        residual_rms(final_residual),
        residual_max(final_residual),
        converged,
        iteration_count,
    )
    return final_state, diagnostics


def solve_steady_chunked(
    state0: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    max_steps: int,
    chunk_steps: int,
    *,
    max_restarts: int = 3,
    freeze_tolerance: float = 1.0e-9,
) -> tuple[ChannelState, SteadySolveDiagnostics, dict]:
    """Run :func:`solve_steady` in chunks of ``chunk_steps`` iterations.

    Motivation (archived run ``cff6489/development-joint-q1-fine``, RTX 4090,
    JAX 0.7.0): inside one long compiled ``while_loop`` the CUDA execution
    silently stopped updating the state after about 2000 iterations; the
    steady residual stayed constant to ten digits for 34 000 further
    iterations, while a fresh call from the archived state converged normally.
    Splitting the solve into separately dispatched chunks bounds the damage of
    such a freeze, and a frozen chunk (residual unchanged over the whole chunk
    although above the threshold) triggers a host round-trip of the state and a
    restart.  The stopping threshold is fixed from the residual of ``state0``
    exactly as in :func:`solve_steady`, so a chunked solve converges to the
    same criterion.  The returned report records every chunk and restart.
    """

    if chunk_steps < 2:
        raise ValueError("chunk_steps must be at least two")
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    import time

    initial_rms = residual_rms(steady_residual(state0, alpha, inlet_u, grid, params))
    threshold = jnp.maximum(
        jnp.asarray(params.nonlinear_atol, dtype=initial_rms.dtype),
        jnp.asarray(params.nonlinear_rtol, dtype=initial_rms.dtype) * initial_rms,
    )
    # Every array is an argument of the compiled program; closing over alpha,
    # the inlet or the state as constants is miscompiled on CUDA for large
    # grids (see run_artifacts.compile_and_sample_args).
    solve = jax.jit(
        lambda state, a, inlet, thr, steps: solve_steady(
            state, a, inlet, grid, params, steps, threshold=thr
        ),
        static_argnums=(4,),
    )
    history_names = [
        "nonlinear_residual_rms_history",
        "nonlinear_residual_max_history",
        "pressure_residual_abs_history",
        "pressure_residual_rel_history",
        "divergence_rms_history",
        "divergence_max_history",
        "inlet_flux_history",
        "outlet_flux_history",
        "mass_flux_mismatch_history",
    ]
    histories: dict[str, list] = {name: [] for name in history_names}
    chunks: list[dict] = []
    state = state0
    iterations = 0
    restarts = 0
    converged = False
    started = time.perf_counter()
    while iterations < max_steps:
        steps = min(chunk_steps, max_steps - iterations)
        if steps < 2:
            steps = 2
        chunk_started = time.perf_counter()
        candidate, diagnostics = solve(state, alpha, inlet_u, threshold, steps)
        candidate = jax.block_until_ready(candidate)
        done = int(diagnostics.iterations)
        rms = np.asarray(diagnostics.nonlinear_residual_rms_history)[:done]
        chunk_converged = bool(diagnostics.converged)
        frozen = (
            not chunk_converged
            and done >= 2
            and abs(float(rms[-1]) - float(rms[0])) <= freeze_tolerance * abs(float(rms[0]))
        )
        chunks.append(
            {
                "first_iteration": iterations + 1,
                "iterations": done,
                "residual_rms_start": float(rms[0]) if done else None,
                "residual_rms_end": float(rms[-1]) if done else None,
                "converged": chunk_converged,
                "frozen": frozen,
                "seconds": time.perf_counter() - chunk_started,
            }
        )
        for name in history_names:
            histories[name].append(np.asarray(getattr(diagnostics, name))[:done])
        iterations += done
        if frozen:
            restarts += 1
            if restarts > max_restarts:
                state = candidate
                break
            # Re-materialise the state on the host before the next dispatch.
            state = jax.tree_util.tree_map(
                lambda x: jax.device_put(np.asarray(jax.device_get(x))), candidate
            )
            continue
        state = candidate
        if chunk_converged:
            converged = True
            break

    def padded(name: str) -> Array:
        values = np.concatenate(histories[name]) if histories[name] else np.zeros((0,))
        out = np.zeros((max_steps,), dtype=np.asarray(initial_rms).dtype)
        n = min(len(values), max_steps)
        out[:n] = values[:n]
        return jnp.asarray(out)

    active = np.zeros((max_steps,), dtype=bool)
    active[: min(iterations, max_steps)] = True
    final_residual = steady_residual(state, alpha, inlet_u, grid, params)
    combined = SteadySolveDiagnostics(
        *(padded(name) for name in history_names),
        jnp.asarray(active),
        initial_rms,
        residual_rms(final_residual),
        residual_max(final_residual),
        jnp.asarray(converged),
        jnp.asarray(iterations, dtype=jnp.int32),
    )
    report = {
        "chunk_steps": int(chunk_steps),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "frozen_chunk_restarts": restarts,
        "max_restarts": int(max_restarts),
        "freeze_tolerance": float(freeze_tolerance),
        "wall_seconds": time.perf_counter() - started,
        "threshold": float(threshold),
    }
    return state, combined, report
