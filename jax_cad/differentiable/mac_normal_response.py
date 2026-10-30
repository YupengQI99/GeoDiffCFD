"""Bounded scalable normal constraint using a pressure-projected response basis.

This is block elimination of the SAME joint projection as mac_normal_constraint,
not sequential normal and pressure projections. Only a marker-sized Schur matrix
is dense; native interpolation uses 16-face stencils by default. An explicit
ratio-two physical kernel uses 64-face stencils for fixed-model refinement.
Geometry/mobility changes
require rebuilding the basis (and differentiation must include that rebuild).
The default MAC solver and historical Brinkman force convention are unchanged.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mac_channel import (
    ChannelState, apply_velocity_boundaries, conservative_advection, divergence,
    interpolate_alpha_to_faces, laplacian_u, laplacian_v, pressure_gradient,
    project_velocity, solve_pressure, residual_rms, residual_max, flux_diagnostics,
)
from .mac_normal_constraint import NormalProjection, _cubic_kernel, MIN_RELATIVE_EIGENVALUE, steady_residual_normal


class NormalStencil(NamedTuple):
    indices_u: jax.Array
    indices_v: jax.Array
    weights_u: jax.Array
    weights_v: jax.Array
    arclength: jax.Array
    valid: jax.Array


class NormalResponse(NamedTuple):
    stencil: NormalStencil
    mobility_u: jax.Array
    mobility_v: jax.Array
    velocity_u: jax.Array  # (markers, nx+1, ny), zero prescribed velocities
    velocity_v: jax.Array
    pressure_impulse: jax.Array  # pressure * dt for each unit impulse
    schur: jax.Array
    inverse_equilibrated_schur: jax.Array
    scale: jax.Array
    valid: jax.Array
    relative_minimum_eigenvalue: jax.Array
    pressure_residual_relative_max: jax.Array
    basis_divergence_max: jax.Array
    relative_asymmetry: jax.Array


def normal_stencil(markers, grid, *, allow_fine=False, kernel_ratio=1):
    """Midpoint quadrature of a cubic kernel with H=kernel_ratio*grid_spacing.

    Integer ratios one/two preserve the cardinal partition and linear moments.
    The 1/ratio**2 factor is the physical dx*dy/(Hx*Hy) quadrature weight,
    not row normalisation or a fitted reaction correction. Kernel width is a
    fixed model parameter, not a geometry-dependent or differentiated argument.
    """
    if type(kernel_ratio) is not int or kernel_ratio not in (1, 2):
        raise ValueError("physical kernel ratio must be the bounded integer one or two")
    limit_x,limit_y,limit_markers = (880,164,128) if allow_fine else (440,82,64)
    if grid.nx > limit_x or grid.ny > limit_y:
        raise ValueError("bounded normal response grid exceeds explicit scope")
    count = markers.points.shape[0]
    if not 1 <= count <= limit_markers or markers.points.shape != (count, 2) or markers.normals.shape != (count, 2) or markers.weights.shape != (count,):
        raise ValueError("normal response marker rows exceed explicit scope or are invalid")
    spacing = jnp.array([grid.dx, grid.dy])
    width = 4*kernel_ratio
    offsets = jnp.stack(jnp.meshgrid(jnp.arange(width), jnp.arange(width), indexing="ij"), axis=-1).reshape((width*width, 2))
    def component(shift, shape):
        coordinate = markers.points/spacing-jnp.array(shift)
        start = jnp.floor(coordinate).astype(jnp.int32)-(2*kernel_ratio-1)
        indices = start[:, None, :]+offsets[None, :, :]
        weights = jnp.prod(_cubic_kernel((coordinate[:, None, :]-indices)/kernel_ratio), axis=-1)/kernel_ratio**2
        inside = jnp.all(indices >= 0) & jnp.all(indices < jnp.array(shape))
        linear = indices[..., 0]*shape[1]+indices[..., 1]
        return jnp.clip(linear, 0, shape[0]*shape[1]-1), weights, inside
    iu, wu, vu = component((0., .5), (grid.nx+1, grid.ny))
    iv, wv, vv = component((.5, 0.), (grid.nx, grid.ny+1))
    valid = (vu & vv & jnp.all(jnp.isfinite(markers.points)) & jnp.all(jnp.isfinite(markers.normals))
             & jnp.all(jnp.isfinite(markers.weights)) & jnp.all(markers.weights > 0)
             & jnp.all(jnp.abs(jnp.linalg.norm(markers.normals, axis=1)-1) < 1.e-12)
             & jnp.all(markers.points >= (2*kernel_ratio+.5)*spacing)
             & jnp.all(markers.points <= jnp.array([grid.lx, grid.ly])-(2*kernel_ratio+.5)*spacing)
             & jnp.all(jnp.abs(jnp.sum(wu, axis=1)-1) < 1.e-12)
             & jnp.all(jnp.abs(jnp.sum(wv, axis=1)-1) < 1.e-12))
    return NormalStencil(iu, iv, wu*markers.normals[:, 0, None], wv*markers.normals[:, 1, None], markers.weights, valid)


def normal_sample(u, v, stencil):
    return (jnp.sum(u.ravel()[stencil.indices_u]*stencil.weights_u, axis=1)
            +jnp.sum(v.ravel()[stencil.indices_v]*stencil.weights_v, axis=1))


def spread_impulse(amplitude, stencil, grid):
    """W_relative^-1 B^T amplitude, where W_relative=W/(dx*dy)."""
    u = jnp.zeros((grid.nx+1)*grid.ny, dtype=amplitude.dtype)
    v = jnp.zeros(grid.nx*(grid.ny+1), dtype=amplitude.dtype)
    u = u.at[stencil.indices_u.ravel()].add((stencil.weights_u*amplitude[:, None]).ravel()).reshape((grid.nx+1, grid.ny))
    v = v.at[stencil.indices_v.ravel()].add((stencil.weights_v*amplitude[:, None]).ravel()).reshape((grid.nx, grid.ny+1))
    u = u.at[0, :].multiply(2).at[-1, :].multiply(2)
    v = v.at[:, 0].multiply(2).at[:, -1].multiply(2)
    return u, v


def build_normal_response(markers, mobility_u, mobility_v, grid, params, *, column_batching="vmap", allow_fine=False, kernel_ratio=1):
    """Differentiable preparation using the existing implicit pressure VJP.

    Forward-mode through the preparation is NOT supported by the old pressure
    custom-VJP. Reverse-mode must include this call, not treat the basis as a
    geometry-independent constant. No marker-rank regularisation is applied.
    ``map`` uses sequential columns with identical pressure equations, allowing
    separate diagnosis of batched CUDA compilation for 24x220x41. Fresh
    pressure/divergence gates are retained for every column in either mode.
    """
    if mobility_u.dtype != jnp.float64 or mobility_v.dtype != jnp.float64:
        raise ValueError("normal response requires float64")
    if column_batching not in ("vmap", "map"):
        raise ValueError("column batching must be vmap or map")
    stencil = normal_stencil(markers, grid, allow_fine=allow_fine, kernel_ratio=kernel_ratio)
    count = len(markers.weights)
    zero_inlet = jnp.zeros(grid.ny, dtype=mobility_u.dtype)
    def column(amplitude):
        ru, rv = spread_impulse(amplitude, stencil, grid)
        tu, tv = apply_velocity_boundaries(mobility_u*ru, mobility_v*rv, zero_inlet, grid)
        phi, _, relative = solve_pressure(-divergence(tu, tv, grid), mobility_u, mobility_v, grid, params)
        gu, gv = pressure_gradient(phi, grid)
        zu, zv = apply_velocity_boundaries(tu-mobility_u*gu, tv-mobility_v*gv, zero_inlet, grid)
        return zu, zv, phi, relative, jnp.max(jnp.abs(divergence(zu, zv, grid)))
    impulses = jnp.eye(count, dtype=mobility_u.dtype)
    zu, zv, phi, residual, div = (jax.vmap(column)(impulses) if column_batching == "vmap"
                                else jax.lax.map(column, impulses))
    schur = jax.vmap(lambda u, v: normal_sample(u, v, stencil))(zu, zv).T
    scale = 1/jnp.sqrt(jnp.diag(schur))
    eq = scale[:, None]*schur*scale[None, :]
    eigenvalues = jnp.linalg.eigvalsh(.5*(eq+eq.T))
    relative_minimum = eigenvalues[0]/eigenvalues[-1]
    asymmetry = jnp.max(jnp.abs(eq-eq.T))/jnp.maximum(jnp.max(jnp.abs(eq)), 1.e-30)
    valid = (stencil.valid & jnp.all(jnp.isfinite(eq)) & jnp.all(jnp.isfinite(mobility_u))
             & jnp.all(jnp.isfinite(mobility_v)) & jnp.all(mobility_u > 0) & jnp.all(mobility_v > 0)
             & (relative_minimum > MIN_RELATIVE_EIGENVALUE) & (asymmetry < 1.e-8)
             & (jnp.max(residual) < 1.e-9) & (jnp.max(div) < 1.e-8))
    inverse = jnp.where(valid, jnp.linalg.inv(eq), jnp.nan)
    return NormalResponse(stencil, mobility_u, mobility_v, zu, zv, phi, schur, inverse,
                          scale, valid, relative_minimum, jnp.max(residual), jnp.max(div), asymmetry)


def project_with_normal_response(u_tilde, v_tilde, inlet_u, grid, params, response):
    """Apply the exact block-eliminated joint correction using one pressure solve."""
    state, diagnostics = project_velocity(u_tilde, v_tilde, response.mobility_u, response.mobility_v, inlet_u, grid, params)
    rhs = normal_sample(state.u, state.v, response.stencil)
    scaled_rhs = response.scale*rhs
    multiplier = response.scale*(response.inverse_equilibrated_schur @ scaled_rhs)
    u = state.u-jnp.einsum("m,mij->ij", multiplier, response.velocity_u)
    v = state.v-jnp.einsum("m,mij->ij", multiplier, response.velocity_v)
    p = state.p-jnp.einsum("m,mij->ij", multiplier, response.pressure_impulse)/params.dt
    ru, rv = spread_impulse(multiplier/params.dt, response.stencil, grid)
    traction = grid.dx*grid.dy*multiplier/(params.dt*response.stencil.arclength)
    normal = normal_sample(u, v, response.stencil)
    defect = response.schur @ multiplier-rhs
    backward = jnp.max(jnp.abs(defect))/jnp.maximum(jnp.linalg.norm(response.schur, ord=jnp.inf)*jnp.max(jnp.abs(multiplier))+jnp.max(jnp.abs(rhs)), 1.e-300)
    su, sv = apply_velocity_boundaries(u_tilde, v_tilde, inlet_u, grid)
    gu, gv = pressure_gradient(p, grid)
    stationarity_u = (u-su)/response.mobility_u+params.dt*(gu+ru)
    stationarity_v = (v-sv)/response.mobility_v+params.dt*(gv+rv)
    stationarity = jnp.maximum(jnp.max(jnp.abs(stationarity_u[1:])), jnp.max(jnp.abs(stationarity_v[:, 1:-1])))
    valid = (response.valid & jnp.all(jnp.isfinite(p)) & (backward < 1.e-9)
             & (diagnostics.pressure_residual_rel < 1.e-8) & jnp.isfinite(params.dt) & (params.dt > 0))
    state = ChannelState(jnp.where(valid, u, jnp.nan), jnp.where(valid, v, jnp.nan), jnp.where(valid, p, jnp.nan))
    return NormalProjection(state, jnp.where(valid, traction, jnp.nan), ru, rv, normal,
                            valid, response.relative_minimum_eigenvalue, backward, stationarity)


def step_with_normal_response(state, inlet_u, grid, params, response):
    if params.include_advection:
        advu, advv = conservative_advection(state.u, state.v, grid, scheme=params.advection_scheme)
    else:
        advu, advv = jnp.zeros_like(state.u), jnp.zeros_like(state.v)
    rhsu = state.u+params.dt*(-advu+params.viscosity*laplacian_u(state.u, grid)+params.force_x)
    rhsv = state.v+params.dt*(-advv+params.viscosity*laplacian_v(state.v, grid)+params.force_y)
    return project_with_normal_response(response.mobility_u*rhsu, response.mobility_v*rhsv,
                                         inlet_u, grid, params, response)


def body_reaction_components(result, alpha, grid):
    """Kinematic force: report old Brinkman, NEW constraint, and their sum."""
    from jax_cad.verification import brinkman_force
    brinkman = jnp.asarray(brinkman_force(result.state, alpha, grid))
    wu = jnp.ones((grid.nx+1, 1), dtype=result.state.u.dtype).at[0].set(.5).at[-1].set(.5)
    wv = jnp.ones((1, grid.ny+1), dtype=result.state.v.dtype).at[:, 0].set(.5).at[:, -1].set(.5)
    constraint = grid.dx*grid.dy*jnp.array([jnp.sum(wu*result.reaction_u), jnp.sum(wv*result.reaction_v)])
    return dict(brinkman=brinkman, constraint=constraint, total=brinkman+constraint)


def normal_control_volume_balance(result, alpha, inlet_u, grid, params, control_volume):
    """Extend native momentum balance to include the new reaction explicitly.

    The inferred flux force is unchanged, but both the captured reaction and
    momentum residual gain the constraint contribution. No volume force is
    renamed as a sharp surface traction. Control volumes must be interior, as
    checked by the existing native MAC audit.
    """
    from jax_cad.verification import mac_staggered_control_volume_momentum_balance
    old = mac_staggered_control_volume_momentum_balance(result.state, alpha, inlet_u, grid, params, control_volume)
    c = control_volume
    constraint = grid.dx*grid.dy*jnp.array([
        jnp.sum(result.reaction_u[c.u_i_start:c.u_i_stop+1, c.u_j_start:c.u_j_stop+1]),
        jnp.sum(result.reaction_v[c.v_i_start:c.v_i_stop+1, c.v_j_start:c.v_j_stop+1]),
    ])
    brinkman = jnp.array([old.captured_brinkman_force_x, old.captured_brinkman_force_y])
    inferred = jnp.array([old.inferred_brinkman_reaction_x, old.inferred_brinkman_reaction_y])
    residual = jnp.array([old.momentum_residual_integral_x, old.momentum_residual_integral_y])+constraint
    global_components = body_reaction_components(result, alpha, grid)
    captured = brinkman+constraint
    return dict(inferred_total=inferred, captured_brinkman=brinkman, captured_constraint=constraint,
                captured_total=captured, global_total=global_components["total"],
                uncaptured_total=global_components["total"]-captured,
                momentum_residual_integral=residual, algebraic_closure=inferred-captured+residual)


def solve_normal_chunk(initial, alpha, inlet_u, grid, params, response, chunk_steps,
                       threshold, completed):
    """Forward-only, bounded history; data-dependent loop avoids stale CUDA CG.

    Pass ALL field/basis arrays as compiled arguments. The fixed convergence
    threshold must be inherited across chunks, not redefined per chunk.
    """
    def diagnostics(result):
        residual, normal = steady_residual_normal(result, alpha, inlet_u, grid, params)
        div = residual.continuity
        _, _, _, _, mismatch = flux_diagnostics(result.state, grid)
        return jnp.array([residual_rms(residual), residual_max(residual),
                          jnp.sqrt(jnp.mean(div**2)), jnp.max(jnp.abs(div)),
                          jnp.max(jnp.abs(normal)), mismatch,
                          result.backward_error, result.stationarity_max])
    first = diagnostics(initial)
    history = jnp.full((chunk_steps, 8), jnp.nan, dtype=initial.state.u.dtype)
    def condition(value):
        result, _, index, metrics = value
        return ((index < chunk_steps) & result.valid & jnp.all(jnp.isfinite(metrics))
                & ((metrics[0] > threshold) | (completed+index < params.nonlinear_min_steps)))
    def body(value):
        previous, records, index, _ = value
        result = step_with_normal_response(previous.state, inlet_u, grid, params, response)
        metrics = diagnostics(result)
        return result, records.at[index].set(metrics), index+1, metrics
    return jax.lax.while_loop(condition, body, (initial, history, jnp.array(0, dtype=jnp.int32), first))
