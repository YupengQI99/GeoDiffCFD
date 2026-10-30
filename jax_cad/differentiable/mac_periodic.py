"""Differentiable periodic 2-D MAC Navier--Stokes--Brinkman kernel.

This module is a deliberately small, verification-oriented baseline.  Pressure
is cell centred, while ``u`` and ``v`` are stored on the east and north faces of
periodic cells.  The discrete divergence and pressure gradient are exact
adjoints up to sign, and the FFT Poisson operator is their algebraic
composition.  Consequently the projection removes discrete divergence to
floating-point accuracy instead of relying on a mismatched collocated stencil.

Two projection variants are provided.  The original constant-mobility variant
projects with ``L_h = D_h G_h`` after the implicit Brinkman update; its fixed
point satisfies ``alpha_f u - nu L u + (1 + dt alpha_f) G_h p = f`` and is
therefore ``O(dt)``-dependent inside the penalized region.  The
mobility-weighted variant (``MACParams.mobility_weighted=True``) projects with
``A_h = -D_h M G_h``, ``M = diag(1/(1 + dt alpha_f))``, so that the fixed point
of the split step is exactly the discrete steady Stokes/Navier--Stokes--Brinkman
system ``alpha_f u - nu L u (+ advection) + G_h p = f``, ``D_h u = 0``,
independently of ``dt``.  The weighted operator is symmetric positive
semidefinite with the constant null mode and is solved matrix-free by
FFT-preconditioned conjugate gradients inside ``jax.lax.custom_linear_solve``.

The solver is intended as a trustworthy scientific kernel on which the unified
design API can be tested.  It is not yet a replacement for production CFD.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, gmres

Array = jax.Array


class PeriodicGrid2D(NamedTuple):
    """Uniform periodic Cartesian grid."""

    nx: int
    ny: int
    lx: float
    ly: float
    dx: float
    dy: float


class MACState(NamedTuple):
    """Periodic staggered velocity and cell-centred pressure."""

    u: Array  # east face of each pressure cell, shape (nx, ny)
    v: Array  # north face of each pressure cell, shape (nx, ny)
    p: Array  # cell centres, shape (nx, ny)


class MACParams(NamedTuple):
    """Fixed numerical and physical parameters.

    ``mobility_weighted`` selects the projection with the Brinkman mobility
    inside the pressure operator (``A_h = -D_h M G_h``).  ``pressure_tol`` and
    ``pressure_maxiter`` control the preconditioned conjugate-gradient solve of
    that variable-coefficient system; they are ignored by the constant-mobility
    FFT projection.
    """

    viscosity: float
    dt: float
    force_x: float = 0.0
    force_y: float = 0.0
    include_advection: bool = True
    mobility_weighted: bool = False
    pressure_tol: float = 1.0e-12
    pressure_maxiter: int = 500


def create_periodic_grid(nx: int, ny: int, lx: float, ly: float) -> PeriodicGrid2D:
    """Create a rectangular periodic grid with independent ``dx`` and ``dy``."""

    if nx < 4 or ny < 4:
        raise ValueError("nx and ny must both be at least 4")
    if lx <= 0 or ly <= 0:
        raise ValueError("lx and ly must be positive")
    return PeriodicGrid2D(nx=nx, ny=ny, lx=lx, ly=ly, dx=lx / nx, dy=ly / ny)


def cell_centres(grid: PeriodicGrid2D, *, dtype=jnp.float32) -> tuple[Array, Array]:
    """Return cell-centre coordinate arrays with shape ``(nx, ny)``."""

    x = (jnp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
    y = (jnp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
    return jnp.meshgrid(x, y, indexing="ij")


def create_initial_state(grid: PeriodicGrid2D, *, dtype=jnp.float32) -> MACState:
    """Create a quiescent periodic state."""

    shape = (grid.nx, grid.ny)
    zeros = jnp.zeros(shape, dtype=dtype)
    return MACState(u=zeros, v=zeros, p=zeros)


def divergence(u: Array, v: Array, grid: PeriodicGrid2D) -> Array:
    """Cell-centred conservative divergence of staggered face velocities."""

    return (
        (u - jnp.roll(u, 1, axis=0)) / grid.dx
        + (v - jnp.roll(v, 1, axis=1)) / grid.dy
    )


def pressure_gradient(p: Array, grid: PeriodicGrid2D) -> tuple[Array, Array]:
    """Pressure gradients at east and north faces."""

    grad_x = (jnp.roll(p, -1, axis=0) - p) / grid.dx
    grad_y = (jnp.roll(p, -1, axis=1) - p) / grid.dy
    return grad_x, grad_y


def laplacian(field: Array, grid: PeriodicGrid2D) -> Array:
    """Periodic second-order Laplacian at the field's native locations."""

    return (
        (jnp.roll(field, -1, axis=0) - 2.0 * field + jnp.roll(field, 1, axis=0))
        / (grid.dx * grid.dx)
        + (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1))
        / (grid.dy * grid.dy)
    )


def _poisson_eigenvalues(grid: PeriodicGrid2D, dtype) -> Array:
    theta_x = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.nx).astype(dtype)
    theta_y = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.ny).astype(dtype)
    lam_x = (2.0 * jnp.cos(theta_x) - 2.0) / (grid.dx * grid.dx)
    lam_y = (2.0 * jnp.cos(theta_y) - 2.0) / (grid.dy * grid.dy)
    return lam_x[:, None] + lam_y[None, :]


def solve_periodic_poisson(rhs: Array, grid: PeriodicGrid2D) -> Array:
    """Solve ``laplacian(p) = rhs`` with a zero-mean pressure gauge."""

    rhs = rhs - jnp.mean(rhs)
    rhs_hat = jnp.fft.fft2(rhs)
    eigenvalues = _poisson_eigenvalues(grid, rhs.real.dtype)
    nonzero = jnp.abs(eigenvalues) > 10.0 * jnp.finfo(rhs.real.dtype).eps
    safe_eigenvalues = jnp.where(nonzero, eigenvalues, 1.0)
    p_hat = jnp.where(nonzero, rhs_hat / safe_eigenvalues, 0.0 + 0.0j)
    p = jnp.fft.ifft2(p_hat).real
    return p - jnp.mean(p)


def project_velocity(
    u_star: Array,
    v_star: Array,
    grid: PeriodicGrid2D,
    dt: float,
) -> MACState:
    """Project a periodic face-velocity field onto the discrete null space of D."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    rhs = divergence(u_star, v_star, grid) / dt
    p = solve_periodic_poisson(rhs, grid)
    grad_x, grad_y = pressure_gradient(p, grid)
    u = u_star - dt * grad_x
    v = v_star - dt * grad_y
    return MACState(u=u, v=v, p=p)


def weighted_pressure_operator(
    p: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: PeriodicGrid2D,
) -> Array:
    """Apply ``A_h p = -D_h M G_h p`` without assembling a matrix."""

    grad_x, grad_y = pressure_gradient(p, grid)
    return -divergence(mobility_u * grad_x, mobility_v * grad_y, grid)


def solve_periodic_weighted_poisson(
    rhs: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: PeriodicGrid2D,
    *,
    tol: float,
    maxiter: int,
) -> Array:
    """Solve ``-D_h M G_h p = rhs`` on the zero-mean subspace.

    The operator is symmetric positive semidefinite with the constant null mode.
    Conjugate gradients are preconditioned by the exact FFT inverse of the
    constant-coefficient operator ``-L_h`` and wrapped in
    ``jax.lax.custom_linear_solve`` so that reverse mode differentiates the
    converged solution implicitly with respect to both the right-hand side and
    the mobility.
    """

    rhs = rhs - jnp.mean(rhs)

    def matvec(p: Array) -> Array:
        return weighted_pressure_operator(p, mobility_u, mobility_v, grid)

    def preconditioner(residual: Array) -> Array:
        return -solve_periodic_poisson(residual, grid)

    def solve(operator, b: Array) -> Array:
        p, _ = cg(operator, b, tol=tol, atol=0.0, maxiter=maxiter, M=preconditioner)
        return p - jnp.mean(p)

    return jax.lax.custom_linear_solve(matvec, rhs, solve, symmetric=True)


def project_velocity_weighted(
    u_tilde: Array,
    v_tilde: Array,
    mobility_u: Array,
    mobility_v: Array,
    grid: PeriodicGrid2D,
    params: "MACParams",
) -> MACState:
    """Mobility-weighted projection ``u = u_tilde - dt M G_h p`` with ``D_h u = 0``."""

    if params.dt <= 0:
        raise ValueError("dt must be positive")
    rhs = -divergence(u_tilde, v_tilde, grid) / params.dt
    p = solve_periodic_weighted_poisson(
        rhs,
        mobility_u,
        mobility_v,
        grid,
        tol=params.pressure_tol,
        maxiter=params.pressure_maxiter,
    )
    grad_x, grad_y = pressure_gradient(p, grid)
    u = u_tilde - params.dt * mobility_u * grad_x
    v = v_tilde - params.dt * mobility_v * grad_y
    return MACState(u=u, v=v, p=p)


def _central_derivative_x(field: Array, grid: PeriodicGrid2D) -> Array:
    return (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / (2.0 * grid.dx)


def _central_derivative_y(field: Array, grid: PeriodicGrid2D) -> Array:
    return (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2.0 * grid.dy)


def _v_at_u(v: Array) -> Array:
    """Bilinearly interpolate north-face ``v`` to east-face ``u`` locations."""

    return 0.25 * (
        v
        + jnp.roll(v, 1, axis=1)
        + jnp.roll(v, -1, axis=0)
        + jnp.roll(jnp.roll(v, -1, axis=0), 1, axis=1)
    )


def _u_at_v(u: Array) -> Array:
    """Bilinearly interpolate east-face ``u`` to north-face ``v`` locations."""

    return 0.25 * (
        u
        + jnp.roll(u, 1, axis=0)
        + jnp.roll(u, -1, axis=1)
        + jnp.roll(jnp.roll(u, 1, axis=0), -1, axis=1)
    )


def advective_terms(u: Array, v: Array, grid: PeriodicGrid2D) -> tuple[Array, Array]:
    """Centred advective-form momentum terms at staggered locations."""

    adv_u = u * _central_derivative_x(u, grid) + _v_at_u(v) * _central_derivative_y(u, grid)
    adv_v = _u_at_v(u) * _central_derivative_x(v, grid) + v * _central_derivative_y(v, grid)
    return adv_u, adv_v


def interpolate_alpha_to_faces(alpha: Array) -> tuple[Array, Array]:
    """Arithmetic interpolation of cell-centred Brinkman resistance to faces."""

    alpha_u = 0.5 * (alpha + jnp.roll(alpha, -1, axis=0))
    alpha_v = 0.5 * (alpha + jnp.roll(alpha, -1, axis=1))
    return alpha_u, alpha_v


def step(
    state: MACState,
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
) -> MACState:
    """Advance one explicit transport / implicit Brinkman / projection step.

    With ``params.mobility_weighted`` the projection uses ``-D_h M G_h`` and the
    step's fixed point is the discrete steady momentum balance
    ``alpha_f u - nu L u (+ advection) + G_h p = f``; otherwise the original
    constant-mobility FFT projection is applied.
    """

    if alpha.shape != (grid.nx, grid.ny):
        raise ValueError(
            f"alpha must have shape {(grid.nx, grid.ny)}, received {alpha.shape}"
        )
    if params.dt <= 0 or params.viscosity < 0:
        raise ValueError("dt must be positive and viscosity non-negative")

    if params.include_advection:
        adv_u, adv_v = advective_terms(state.u, state.v, grid)
    else:
        adv_u = jnp.zeros_like(state.u)
        adv_v = jnp.zeros_like(state.v)

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha)
    rhs_u = state.u + params.dt * (
        -adv_u + params.viscosity * laplacian(state.u, grid) + params.force_x
    )
    rhs_v = state.v + params.dt * (
        -adv_v + params.viscosity * laplacian(state.v, grid) + params.force_y
    )

    if params.mobility_weighted:
        mobility_u = 1.0 / (1.0 + params.dt * alpha_u)
        mobility_v = 1.0 / (1.0 + params.dt * alpha_v)
        return project_velocity_weighted(
            mobility_u * rhs_u,
            mobility_v * rhs_v,
            mobility_u,
            mobility_v,
            grid,
            params,
        )

    u_star = rhs_u / (1.0 + params.dt * alpha_u)
    v_star = rhs_v / (1.0 + params.dt * alpha_v)
    return project_velocity(u_star, v_star, grid, params.dt)


def simulate(
    state0: MACState,
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
    n_steps: int,
) -> MACState:
    """Run a fixed-horizon differentiable simulation with ``jax.lax.scan``."""

    if n_steps < 1:
        raise ValueError("n_steps must be at least one")

    def body(state, _):
        next_state = step(state, alpha, grid, params)
        return next_state, None

    final_state, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return final_state


def make_fixed_point_preconditioner(
    grid: PeriodicGrid2D,
    params: MACParams,
    alpha_mean: float,
):
    """Fourier-space inverse of the constant-coefficient part of ``I - F_s``.

    For the Stokes--Brinkman step the fixed-point residual operator acting on a
    velocity perturbation is approximately
    ``dt * (alpha_mean - nu * Laplacian)`` on the divergence-free subspace, up
    to the compact perturbation ``dt * (alpha_f - alpha_mean)`` inside the
    obstacle.  Inverting the constant-coefficient part exactly with FFTs gives a
    symmetric preconditioner under which restarted GMRES converges in a number
    of iterations that is insensitive to the grid spacing.  The pressure
    component of the state is passed through unchanged.
    """

    if alpha_mean <= 0:
        raise ValueError("alpha_mean must be positive")
    eigenvalues = None

    def apply(state: MACState) -> MACState:
        nonlocal eigenvalues
        dtype = state.u.dtype
        if eigenvalues is None or eigenvalues.dtype != dtype:
            eigenvalues = _poisson_eigenvalues(grid, dtype)
        symbol = params.dt * (alpha_mean - params.viscosity * eigenvalues)

        def solve(field):
            return jnp.fft.ifft2(jnp.fft.fft2(field) / symbol).real

        return MACState(u=solve(state.u), v=solve(state.v), p=state.p)

    return apply


class SteadySolveDiagnostics(NamedTuple):
    """Residuals of the direct steady saddle-point solve."""

    momentum_residual_rms: Array
    momentum_residual_max: Array
    divergence_max: Array
    converged: Array


def _saddle_matvec(u, v, p, alpha_u, alpha_v, grid, params):
    """Apply the symmetric steady Stokes--Brinkman saddle-point operator."""

    grad_x, grad_y = pressure_gradient(p, grid)
    ru = alpha_u * u - params.viscosity * laplacian(u, grid) + grad_x
    rv = alpha_v * v - params.viscosity * laplacian(v, grid) + grad_y
    # Continuity row written as G^T u = -D u so that the block system is symmetric.
    rp = -divergence(u, v, grid)
    return ru, rv, rp


def steady_stokes_brinkman(
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
    *,
    tol: float = 1e-12,
    atol: float = 0.0,
    maxiter: int = 2000,
    restart: int = 50,
    alpha_mean: float | None = None,
) -> MACState:
    """Direct solve of the discrete steady Stokes--Brinkman system.

    Solves the symmetric indefinite saddle-point problem
    ``[alpha_f - nu Lap, G; G^T, 0] (u, p) = (f, 0)`` with restarted GMRES and
    a block-diagonal preconditioner built from FFT inverses of the
    constant-coefficient operators: the velocity block uses
    ``(alpha_mean - nu Lap)^{-1}`` and the pressure block the Cahouet--Chabard
    approximation ``nu I + alpha_mean (-L_h)^{-1}``.  The solve is wrapped in
    ``jax.lax.custom_linear_solve`` so that reverse mode obtains the exact
    residual-based adjoint ``A^T lambda = g`` (``A`` is symmetric) instead of
    differentiating through the Krylov iteration.  The pressure gauge is zero
    mean.  This is the steady counterpart of the fixed point of :func:`step`
    with the mobility-weighted projection (Proposition~\ref{prop:fixed_point}
    of the manuscript) and is independent of ``params.dt``.
    """

    if alpha.shape != (grid.nx, grid.ny):
        raise ValueError(f"alpha must have shape {(grid.nx, grid.ny)}, received {alpha.shape}")
    if params.include_advection:
        raise ValueError("the direct steady solve is restricted to the affine Stokes--Brinkman map")
    dtype = alpha.dtype
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha)
    if alpha_mean is None:
        alpha_mean = jnp.maximum(jnp.mean(alpha), 1e-3 * jnp.max(alpha) + 1e-12)
    eigenvalues = _poisson_eigenvalues(grid, dtype)
    velocity_symbol = alpha_mean - params.viscosity * eigenvalues
    nonzero = jnp.abs(eigenvalues) > 10.0 * jnp.finfo(dtype).eps
    safe = jnp.where(nonzero, eigenvalues, 1.0)
    laplace_inverse_symbol = jnp.where(nonzero, 1.0 / (-safe), 0.0)

    shape = (grid.nx, grid.ny)
    n = grid.nx * grid.ny
    zeros = jnp.zeros(shape, dtype=dtype)
    rhs = jnp.concatenate(
        [(zeros + params.force_x).ravel(), (zeros + params.force_y).ravel(), zeros.ravel()]
    )

    def split(flat):
        return flat[:n].reshape(shape), flat[n : 2 * n].reshape(shape), flat[2 * n :].reshape(shape)

    def matvec(flat):
        u, v, p = split(flat)
        ru, rv, rp = _saddle_matvec(u, v, p, alpha_u, alpha_v, grid, params)
        return jnp.concatenate([ru.ravel(), rv.ravel(), (rp - jnp.mean(rp)).ravel()])

    def preconditioner(flat):
        u, v, p = split(flat)

        def velocity(field):
            return jnp.fft.ifft2(jnp.fft.fft2(field) / velocity_symbol).real

        p0 = p - jnp.mean(p)
        pressure = params.viscosity * p0 + alpha_mean * jnp.fft.ifft2(jnp.fft.fft2(p0) * laplace_inverse_symbol).real
        return jnp.concatenate([velocity(u).ravel(), velocity(v).ravel(), (pressure - jnp.mean(pressure)).ravel()])

    def solve(operator, b):
        x, _ = gmres(
            operator,
            b,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            M=preconditioner,
            solve_method="batched",
        )
        u, v, p = split(x)
        return jnp.concatenate([u.ravel(), v.ravel(), (p - jnp.mean(p)).ravel()])

    flat = jax.lax.custom_linear_solve(matvec, rhs, solve, symmetric=True)
    u, v, p = split(flat)
    return MACState(u=u, v=v, p=p)


def steady_adjoint_residual(
    state: MACState,
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
    cotangent: MACState,
    *,
    tol: float = 1e-12,
    atol: float = 0.0,
    maxiter: int = 4000,
    restart: int = 50,
) -> tuple[MACState, Array]:
    """Solve the steady adjoint ``A^T lambda = g`` and return ``(lambda, rel. residual)``.

    ``A`` is the symmetric saddle-point operator of :func:`steady_stokes_brinkman`,
    so the adjoint system is the same operator applied to the state cotangent
    ``g = dJ/d(u, v, p)``.  This diagnostic mirrors the transposed solve that
    ``jax.lax.custom_linear_solve`` performs inside reverse mode and lets an
    experiment archive the adjoint residual next to the primal residual.
    """

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha)
    dtype = alpha.dtype
    alpha_mean = jnp.maximum(jnp.mean(alpha), 1e-3 * jnp.max(alpha) + 1e-12)
    eigenvalues = _poisson_eigenvalues(grid, dtype)
    velocity_symbol = alpha_mean - params.viscosity * eigenvalues
    nonzero = jnp.abs(eigenvalues) > 10.0 * jnp.finfo(dtype).eps
    laplace_inverse_symbol = jnp.where(nonzero, 1.0 / (-jnp.where(nonzero, eigenvalues, 1.0)), 0.0)
    shape = (grid.nx, grid.ny)
    n = grid.nx * grid.ny

    def split(flat):
        return flat[:n].reshape(shape), flat[n : 2 * n].reshape(shape), flat[2 * n :].reshape(shape)

    def matvec(flat):
        u, v, p = split(flat)
        ru, rv, rp = _saddle_matvec(u, v, p, alpha_u, alpha_v, grid, params)
        return jnp.concatenate([ru.ravel(), rv.ravel(), (rp - jnp.mean(rp)).ravel()])

    def preconditioner(flat):
        u, v, p = split(flat)
        velocity = lambda f: jnp.fft.ifft2(jnp.fft.fft2(f) / velocity_symbol).real
        p0 = p - jnp.mean(p)
        pressure = params.viscosity * p0 + alpha_mean * jnp.fft.ifft2(jnp.fft.fft2(p0) * laplace_inverse_symbol).real
        return jnp.concatenate([velocity(u).ravel(), velocity(v).ravel(), (pressure - jnp.mean(pressure)).ravel()])

    g = jnp.concatenate([cotangent.u.ravel(), cotangent.v.ravel(), (cotangent.p - jnp.mean(cotangent.p)).ravel()])
    lam, _ = gmres(matvec, g, tol=tol, atol=atol, maxiter=maxiter, restart=restart, M=preconditioner, solve_method="batched")
    residual = jnp.linalg.norm(matvec(lam) - g) / jnp.maximum(jnp.linalg.norm(g), 1e-300)
    lu, lv, lp = split(lam)
    return MACState(u=lu, v=lv, p=lp), residual


def steady_solve_diagnostics(
    state: MACState,
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
    *,
    tolerance: float = 1e-9,
) -> SteadySolveDiagnostics:
    """Residuals of the steady system at ``state`` (for archiving)."""

    residual = steady_residual(state, alpha, grid, params)
    stacked = jnp.concatenate([residual.momentum_u.ravel(), residual.momentum_v.ravel()])
    rms = jnp.sqrt(jnp.mean(stacked * stacked))
    maximum = jnp.max(jnp.abs(stacked))
    div_max = jnp.max(jnp.abs(residual.continuity))
    return SteadySolveDiagnostics(rms, maximum, div_max, (rms <= tolerance) & (div_max <= tolerance))


class SteadyResidual(NamedTuple):
    """Discrete steady momentum and continuity residuals at staggered locations."""

    momentum_u: Array
    momentum_v: Array
    continuity: Array


def steady_residual(
    state: MACState,
    alpha: Array,
    grid: PeriodicGrid2D,
    params: MACParams,
) -> SteadyResidual:
    """Residual of ``alpha_f u - nu L u (+ advection) + G_h p - f`` and ``D_h u``.

    This is the steady system a converged pseudo-time iteration is expected to
    satisfy.  The constant-mobility projection converges to a different
    ``dt``-dependent balance; the mobility-weighted projection satisfies this
    residual to the pressure-solve tolerance.
    """

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha)
    grad_x, grad_y = pressure_gradient(state.p, grid)
    if params.include_advection:
        adv_u, adv_v = advective_terms(state.u, state.v, grid)
    else:
        adv_u = jnp.zeros_like(state.u)
        adv_v = jnp.zeros_like(state.v)
    momentum_u = (
        adv_u
        + alpha_u * state.u
        - params.viscosity * laplacian(state.u, grid)
        + grad_x
        - params.force_x
    )
    momentum_v = (
        adv_v
        + alpha_v * state.v
        - params.viscosity * laplacian(state.v, grid)
        + grad_y
        - params.force_y
    )
    return SteadyResidual(momentum_u, momentum_v, divergence(state.u, state.v, grid))


def steady_residual_norms(residual: SteadyResidual) -> tuple[Array, Array]:
    """RMS and maximum absolute value over all residual components."""

    stacked = jnp.concatenate(
        [
            residual.momentum_u.ravel(),
            residual.momentum_v.ravel(),
            residual.continuity.ravel(),
        ]
    )
    return jnp.sqrt(jnp.mean(stacked * stacked)), jnp.max(jnp.abs(stacked))


def mean_flow_loss(state: MACState, alpha: Array | None = None) -> Array:
    """Negative mean streamwise flow, suitable for minimization at fixed forcing."""

    del alpha
    return -jnp.mean(state.u)


def kinetic_energy(state: MACState) -> Array:
    """Domain-mean kinetic energy."""

    return 0.5 * jnp.mean(state.u * state.u + state.v * state.v)


def divergence_norms(state: MACState, grid: PeriodicGrid2D) -> tuple[Array, Array]:
    """Return RMS and maximum absolute discrete divergence."""

    div = divergence(state.u, state.v, grid)
    return jnp.sqrt(jnp.mean(div * div)), jnp.max(jnp.abs(div))
