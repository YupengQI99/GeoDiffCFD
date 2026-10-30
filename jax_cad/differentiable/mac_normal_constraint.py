"""Opt-in, CPU-sized MAC normal-constraint prototype (NOT cylinder qualified).

Solve incompressibility and marker no-penetration *jointly*. With physical
velocity quadrature W and Brinkman mobility M, minimise
    1/2 (q-q*)^T W M^-1 (q-q*) subject to D q = 0, B q = 0.
Fixed outer velocities are eliminated, not penalised. The outlet u volume is
half-sized: W G = -V D^T on free faces, V=dx*dy. Pressure and normal reaction
are recovered from the two multipliers. Spreading is W^-1 B^T S, the exact
work adjoint of interpolation, S being marker arclength quadrature.

Dense solves are deliberately limited to 256 pressure cells / 64 markers.
Redundant/ill-conditioned constraints FAIL CLOSED; no jitter, pseudoinverse,
pressure pinning or silent row deletion. Differentiation is only meaningful
in the admissible full-rank region. The normal constraint alone is not no-slip.
No production solver, cylinder driver or historical force estimator is changed.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .mac_channel import (
    ChannelGrid2D, ChannelParams, ChannelResidual, ChannelState,
    apply_velocity_boundaries, conservative_advection,
    interpolate_alpha_to_faces, laplacian_u, laplacian_v, steady_residual,
)

FORCE_PROTOCOL = "mac-brinkman-plus-normal-reaction-prototype-v1"
MIN_RELATIVE_EIGENVALUE = 1.e-10


class NormalMarkers(NamedTuple):
    points: jax.Array
    normals: jax.Array  # body-outward unit normals
    weights: jax.Array  # positive arclength weights


class DenseMACLayout(NamedTuple):
    divergence_matrix: jax.Array
    volume_weights: jax.Array
    free: jax.Array
    face_points_u: jax.Array
    face_points_v: jax.Array


class NormalProjection(NamedTuple):
    state: ChannelState
    normal_traction: jax.Array  # reaction ON BODY, along body-outward normal
    reaction_u: jax.Array  # force ON BODY per Eulerian volume; fluid sees minus
    reaction_v: jax.Array
    normal_velocity: jax.Array
    valid: jax.Array
    relative_minimum_eigenvalue: jax.Array
    backward_error: jax.Array
    stationarity_max: jax.Array


def dense_mac_layout(grid: ChannelGrid2D) -> DenseMACLayout:
    """Static bounded operators; array ordering matches ChannelState.ravel()."""
    nx, ny = grid.nx, grid.ny
    if nx*ny > 256:
        raise ValueError("normal prototype is limited to 256 pressure cells")
    nu, nv = (nx+1)*ny, nx*(ny+1)
    d = np.zeros((nx*ny, nu+nv))
    for i in range(nx):
        for j in range(ny):
            row = i*ny+j
            d[row, (i+1)*ny+j], d[row, i*ny+j] = 1/grid.dx, -1/grid.dx
            d[row, nu+i*(ny+1)+j+1], d[row, nu+i*(ny+1)+j] = 1/grid.dy, -1/grid.dy
    wu, wv = np.ones((nx+1, ny)), np.ones((nx, ny+1))
    wu[[0, -1], :] = .5
    wv[:, [0, -1]] = .5
    free_u, free_v = np.ones_like(wu, dtype=bool), np.ones_like(wv, dtype=bool)
    free_u[0, :], free_v[:, 0], free_v[:, -1] = False, False, False
    xu, yu = np.meshgrid(np.arange(nx+1)*grid.dx, (np.arange(ny)+.5)*grid.dy, indexing="ij")
    xv, yv = np.meshgrid((np.arange(nx)+.5)*grid.dx, np.arange(ny+1)*grid.dy, indexing="ij")
    return DenseMACLayout(
        jnp.asarray(d), jnp.asarray(np.r_[wu.ravel(), wv.ravel()]*grid.dx*grid.dy),
        jnp.asarray(np.flatnonzero(np.r_[free_u.ravel(), free_v.ravel()])),
        jnp.asarray(np.c_[xu.ravel(), yu.ravel()]), jnp.asarray(np.c_[xv.ravel(), yv.ravel()]),
    )


def ellipse_markers(center, semiaxes, angle, count: int, *, allow_fine=False) -> NormalMarkers:
    """Differentiable positions, unit normals AND arclength; uniform angle rule.

    Ellipse quadrature is not claimed exact. Equal semiaxes give a circle with
    exact constant arclength weights. No reference CFD coefficient is used.
    """
    if not 1 <= count <= (128 if allow_fine else 64):
        raise ValueError("normal markers exceed the explicit bounded scope")
    theta = 2*jnp.pi*jnp.arange(count, dtype=jnp.float64)/count
    c, s = jnp.cos(theta), jnp.sin(theta)
    axes = jnp.asarray(semiaxes)
    rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                          [jnp.sin(angle), jnp.cos(angle)]])
    local_normal = jnp.stack((c/axes[0], s/axes[1]), axis=1)
    normals = local_normal/jnp.linalg.norm(local_normal, axis=1, keepdims=True)
    points = jnp.stack((axes[0]*c, axes[1]*s), axis=1) @ rotation.T+jnp.asarray(center)
    weights = (2*jnp.pi/count)*jnp.sqrt((axes[0]*s)**2+(axes[1]*c)**2)
    # Invalid axes must not masquerade as a reflected valid body.
    weights = jnp.where(jnp.all(axes > 0), weights, jnp.nan)
    return NormalMarkers(points, normals @ rotation.T, weights)


def _cubic_kernel(x):
    a = jnp.abs(x)
    return jnp.where(a < 1, 2/3-a*a+.5*a*a*a,
                     jnp.where(a < 2, (2-a)**3/6, 0.))


def normal_interpolation(markers: NormalMarkers, layout: DenseMACLayout,
                         grid: ChannelGrid2D):
    """C2 tensor cubic B-spline interpolation on each native MAC lattice.

    Require complete support strictly away from prescribed/exterior faces;
    do not renormalise truncated kernels. The flag is also checked inside JIT.
    """
    count = markers.points.shape[0]
    if not 1 <= count <= 64 or markers.points.shape != (count, 2) or markers.normals.shape != (count, 2) or markers.weights.shape != (count,):
        raise ValueError("invalid normal marker shapes/count")
    spacing = jnp.array([grid.dx, grid.dy])
    def sample(points):
        delta = (markers.points[:, None, :]-points[None, :, :])/spacing
        return jnp.prod(_cubic_kernel(delta), axis=-1)
    bu, bv = sample(layout.face_points_u), sample(layout.face_points_v)
    valid = (jnp.all(jnp.isfinite(markers.points)) & jnp.all(jnp.isfinite(markers.normals))
             & jnp.all(jnp.isfinite(markers.weights)) & jnp.all(markers.weights > 0)
             & jnp.all(jnp.abs(jnp.linalg.norm(markers.normals, axis=1)-1) < 1.e-12)
             & jnp.all(markers.points >= 2.5*spacing)
             & jnp.all(markers.points <= jnp.array([grid.lx, grid.ly])-2.5*spacing)
             & jnp.all(jnp.abs(jnp.sum(bu, axis=1)-1) < 1.e-12)
             & jnp.all(jnp.abs(jnp.sum(bv, axis=1)-1) < 1.e-12))
    return jnp.concatenate((bu*markers.normals[:, 0, None],
                            bv*markers.normals[:, 1, None]), axis=1), valid


def _join(u, v):
    return jnp.concatenate((u.ravel(), v.ravel()))


def _split(q, grid):
    n = (grid.nx+1)*grid.ny
    return q[:n].reshape((grid.nx+1, grid.ny)), q[n:].reshape((grid.nx, grid.ny+1))


def spread_normal_reaction(traction, markers, layout, grid):
    """Work-conjugate body reaction density (negative of force on the fluid)."""
    b, _ = normal_interpolation(markers, layout, grid)
    return _split((b.T @ (markers.weights*traction))/layout.volume_weights, grid)


def project_velocity_normal(u_tilde, v_tilde, mobility_u, mobility_v, inlet_u,
                            grid, params, layout, markers) -> NormalProjection:
    """Joint weighted projection; check ``valid`` before consuming ANY result.

    Pressure uses the existing open-outlet gauge, not an extra pinned cell.
    Rank diagnostic is for the diagonally equilibrated joint Schur matrix.
    Redundant markers are rejected rather than assigned nonunique tractions.
    Supports JVP/VJP for admissible fixed marker count/layout in float64.
    """
    if grid.nx*grid.ny > 256:
        raise ValueError("normal prototype is limited to 256 pressure cells")
    if u_tilde.dtype != jnp.float64 or v_tilde.dtype != jnp.float64:
        raise ValueError("normal prototype requires float64")
    u, v = apply_velocity_boundaries(u_tilde, v_tilde, inlet_u, grid)
    qstar, mobility = _join(u, v), _join(mobility_u, mobility_v)
    b, geometry_ok = normal_interpolation(markers, layout, grid)
    h, npres = min(grid.dx, grid.dy), grid.nx*grid.ny
    c = jnp.concatenate((h*layout.divergence_matrix, b), axis=0)
    cf = c[:, layout.free]
    inverse_hessian = mobility[layout.free]/layout.volume_weights[layout.free]
    schur = (cf*inverse_hessian) @ cf.T
    scale = 1/jnp.sqrt(jnp.diag(schur))
    equilibrated = scale[:, None]*schur*scale[None, :]
    eigenvalues = jnp.linalg.eigvalsh(equilibrated)
    relative_minimum = eigenvalues[0]/eigenvalues[-1]
    valid = (geometry_ok & (relative_minimum > MIN_RELATIVE_EIGENVALUE)
             & jnp.all(jnp.isfinite(equilibrated)) & jnp.all(jnp.isfinite(qstar))
             & jnp.all(jnp.isfinite(mobility)) & jnp.all(mobility > 0)
             & jnp.isfinite(params.dt) & (params.dt > 0))
    rhs = c @ qstar
    z = jnp.linalg.solve(equilibrated, scale*rhs)
    multiplier = scale*z
    # Singular/redundant systems must never silently return plausible forces.
    multiplier = jnp.where(valid, multiplier, jnp.nan)
    corrected = qstar.at[layout.free].add(-inverse_hessian*(cf.T @ multiplier))
    pressure = -h*multiplier[:npres]/(params.dt*grid.dx*grid.dy)
    traction = multiplier[npres:]/(params.dt*markers.weights)
    ru, rv = spread_normal_reaction(traction, markers, layout, grid)
    cu, cv = _split(corrected, grid)
    defect = equilibrated @ z-scale*rhs
    backward = jnp.max(jnp.abs(defect))/jnp.maximum(
        jnp.linalg.norm(equilibrated, ord=jnp.inf)*jnp.max(jnp.abs(z))
        +jnp.max(jnp.abs(scale*rhs)), jnp.finfo(qstar.dtype).tiny)
    stationarity = ((corrected-qstar)[layout.free]/inverse_hessian+cf.T @ multiplier)
    return NormalProjection(ChannelState(cu, cv, pressure.reshape((grid.nx, grid.ny))),
                            traction, ru, rv, b @ corrected, valid, relative_minimum,
                            backward, jnp.max(jnp.abs(stationarity)))


def step_normal(state, alpha, inlet_u, grid, params, layout, markers):
    """Experimental counterpart of MAC step, with identical explicit transport."""
    au, av = interpolate_alpha_to_faces(alpha, grid)
    mu, mv = 1/(1+params.dt*au), 1/(1+params.dt*av)
    if params.include_advection:
        advu, advv = conservative_advection(state.u, state.v, grid, scheme=params.advection_scheme)
    else:
        advu, advv = jnp.zeros_like(state.u), jnp.zeros_like(state.v)
    rhsu = state.u+params.dt*(-advu+params.viscosity*laplacian_u(state.u, grid)+params.force_x)
    rhsv = state.v+params.dt*(-advv+params.viscosity*laplacian_v(state.v, grid)+params.force_y)
    return project_velocity_normal(mu*rhsu, mv*rhsv, mu, mv, inlet_u, grid, params, layout, markers)


def steady_residual_normal(result, alpha, inlet_u, grid, params):
    """Momentum residual includes BOTH Brinkman and constraint reactions.

    Return normal constraints separately. This is a residual evaluator, not a
    new steady solver or an assertion that a projected timestep is converged.
    """
    base = steady_residual(result.state, alpha, inlet_u, grid, params)
    ru, rv = base.momentum_u+result.reaction_u, base.momentum_v+result.reaction_v
    ru = ru.at[0, :].set(base.momentum_u[0, :])
    rv = rv.at[:, 0].set(base.momentum_v[:, 0]).at[:, -1].set(base.momentum_v[:, -1])
    return ChannelResidual(ru, rv, base.continuity), result.normal_velocity


def total_body_reaction(result, alpha, layout, grid):
    """Versioned new-model total; never substitute this into old paper tables."""
    au, av = interpolate_alpha_to_faces(alpha, grid)
    q = _join(au*result.state.u+result.reaction_u, av*result.state.v+result.reaction_v)
    weighted_u, weighted_v = _split(layout.volume_weights*q, grid)
    return jnp.array([jnp.sum(weighted_u), jnp.sum(weighted_v)])


def require_valid(result: NormalProjection):
    """Host-side fail-fast companion; do not call from JIT/gradient traces."""
    if not bool(result.valid):
        raise ValueError("inadmissible geometry/mobility or rank-deficient normal constraint system")
    return result
