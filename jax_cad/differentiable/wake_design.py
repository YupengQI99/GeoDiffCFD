"""Opt-in finite-horizon wake design through the existing material interface.

This is a DEVELOPMENT experiment, not sharp-wall cylinder verification. The
physics consumes only the cell material field, not a geometry-specific B matrix.
Forward and reverse chunks use dynamic while loops, avoiding the historical
CUDA scan/inner-pressure compilation failure. The reverse pass replays each
chunk and differentiates EVERY time step, including the explicit force/alpha
dependence. It is a discrete transient adjoint, not a steady-state adjoint.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .mac_channel import ChannelState, cell_centres, step
from jax_cad.unified.geometry import fourier_boundary_radius
from jax_cad.unified.materials import fields_from_sdf
from jax_cad.verification import brinkman_force


def shape_fields(theta, grid, *, center, radius, alpha_max, epsilon,
                 wall_shift=0., normalize_level_set=False):
    """Two even cosine modes; exact quadrature area, fixed centre, no B data.

    Nonzero modes give a RADIAL level set, not Euclidean signed distance.
    Epsilon is fixed in this field coordinate; no sharp-limit claim is made.
    """
    theta = jnp.asarray(theta)
    if theta.shape != (2,):
        raise ValueError("wake design uses exactly two cosine coefficients")
    x, y = cell_centres(grid, dtype=theta.dtype)
    dx, dy = x-center[0], y-center[1]
    angle = jnp.arctan2(dy, dx)
    coefficients = jnp.concatenate((theta, jnp.zeros_like(theta)))
    boundary = fourier_boundary_radius(angle, coefficients, base_radius=radius,
                                       modes=(2, 4), n_quadrature=1024)
    phi = jnp.sqrt(dx*dx+dy*dy+jnp.finfo(theta.dtype).tiny)-boundary
    from .wall_shift import corrected_level_set
    phi = corrected_level_set(phi,angle,coefficients,(2,4),wall_shift=wall_shift,
                              normalize_level_set=normalize_level_set)
    return fields_from_sdf(phi, alpha_max=alpha_max, epsilon=epsilon, profile="compact")


def initial_flow(grid, inlet):
    """Same divergence-free full-channel profile for every design (not warm-started)."""
    return ChannelState(jnp.broadcast_to(inlet, (grid.nx+1, grid.ny)),
                        jnp.zeros((grid.nx, grid.ny+1), dtype=inlet.dtype),
                        jnp.zeros((grid.nx, grid.ny), dtype=inlet.dtype))


def make_wake_kernels(grid, params, *, chunk_steps, force_scale):
    """Return jitted forward/replay/adjoint blocks; fields enter as arguments.

    history columns: Cd, Cl, pressure residual, div max, mass mismatch, CFL,
    maximum speed. ``weights`` are fixed quadrature weights on the objective
    window, including zeros during startup. Each block has static capacity but
    a dynamic active length; CPU/GPU audits compare both value and pullback.
    """
    def local(state, alpha, inlet):
        following, diagnostic = step(state, alpha, inlet, grid, params)
        forces = force_scale*jnp.asarray(brinkman_force(following, alpha, grid))
        cfl = params.dt*(jnp.max(jnp.abs(following.u))/grid.dx
                         + jnp.max(jnp.abs(following.v))/grid.dy)
        speed = jnp.maximum(jnp.max(jnp.abs(following.u)), jnp.max(jnp.abs(following.v)))
        metrics = jnp.array([forces[0], forces[1], diagnostic.pressure_residual_rel,
                             diagnostic.divergence_max, diagnostic.mass_flux_mismatch, cfl, speed])
        return following, forces[0], metrics

    def forward(state, alpha, inlet, weights, active):
        history = jnp.zeros((chunk_steps, 7), dtype=state.u.dtype)
        def body(carry):
            k, current, total, history = carry
            following, drag, metrics = local(current, alpha, inlet)
            return k+1, following, total+weights[k]*drag, history.at[k].set(metrics)
        _, final, value, history = jax.lax.while_loop(
            lambda c: c[0] < active, body,
            (jnp.array(0), state, jnp.array(0., dtype=state.u.dtype), history))
        return final, value, history

    def replay(state, alpha, inlet, active):
        tape = jax.tree.map(lambda x: jnp.zeros((chunk_steps,)+x.shape, x.dtype), state)
        def body(carry):
            k, current, tape = carry
            tape = jax.tree.map(lambda buffer, x: buffer.at[k].set(x), tape, current)
            following, _, _ = local(current, alpha, inlet)
            return k+1, following, tape
        _, final, tape = jax.lax.while_loop(lambda c: c[0] < active, body,
                                            (jnp.array(0), state, tape))
        return final, tape

    def reverse(tape, alpha, inlet, weights, active, final_cotangent):
        def differentiable_step(state, material):
            following, drag, _ = local(state, material, inlet)
            return following, drag
        def body(carry):
            k, cotangent, material_cotangent = carry
            state = jax.tree.map(lambda x: x[k], tape)
            _, pullback = jax.vjp(differentiable_step, state, alpha)
            cotangent, increment = pullback((cotangent, weights[k]))
            return k-1, cotangent, material_cotangent+increment
        _, initial_cotangent, material_cotangent = jax.lax.while_loop(
            lambda c: c[0] >= 0, body,
            (active-1, final_cotangent, jnp.zeros_like(alpha)))
        return initial_cotangent, material_cotangent

    return jax.jit(forward), jax.jit(replay), jax.jit(reverse)


def validate_history(history):
    history = np.asarray(history)
    if not np.all(np.isfinite(history)):
        raise RuntimeError("nonfinite transient field/diagnostic")
    for column, limit, name in ((2, 1e-8, "fresh pressure residual"),
                                 (3, 1e-7, "divergence"), (4, 1e-8, "mass mismatch"),
                                 (5, .8, "advective CFL"), (6, 8., "velocity")):
        if np.max(history[:, column]) > limit:
            raise RuntimeError(f"{name} exceeded {limit}: {np.max(history[:, column])}")


def objective_weights(n_steps, window_start_step):
    if not 0 <= window_start_step < n_steps:
        raise ValueError("objective requires a nonempty declared time window")
    return np.r_[np.zeros(window_start_step),
                 np.ones(n_steps-window_start_step)/(n_steps-window_start_step)]


def evaluate_trajectory(alpha, inlet, state0, kernels, *, n_steps, chunk_steps,
                        window_start_step, gradient=False, progress=None):
    """Checkpointed trajectory; no frozen warmup and no omitted geometry terms.

    Host checkpoints scale as n_steps/chunk_steps; device replay tape has only
    chunk_steps states. All state/material dependence is differentiated. Caller
    subsequently pulls the shared material cotangent back to native geometry.
    """
    forward, replay, reverse = kernels
    weights = objective_weights(n_steps, window_start_step)
    state, value, checkpoints, history, samples = state0, 0., [], [], []
    for start in range(0, n_steps, chunk_steps):
        active = min(chunk_steps, n_steps-start)
        block_weights = np.zeros(chunk_steps)
        block_weights[:active] = weights[start:start+active]
        if gradient:
            checkpoints.append((start, jax.tree.map(lambda x: np.asarray(x).copy(), state)))
        state, increment, records = forward(state, alpha, inlet, jnp.asarray(block_weights), jnp.array(active))
        records = np.asarray(records)[:active]
        validate_history(records)
        value += float(increment)
        history.append(records)
        # Global evolution plus at most 20 late frames resolves a shedding
        # cycle without accidentally sampling nearly identical phases. This
        # changes saved output only, never the objective or trajectory.
        if (start == 0 or start+active > n_steps-20*chunk_steps
                or (start+active) % max(chunk_steps, (n_steps//16//chunk_steps)*chunk_steps) == 0):
            samples.append((start+active, jax.tree.map(lambda x: np.asarray(x).copy(), state)))
        if progress:
            progress("forward", start+active, value, records[-1])
    material_gradient = None
    if gradient:
        cotangent = jax.tree.map(jnp.zeros_like, state)
        material_gradient = jnp.zeros_like(alpha)
        next_expected = jax.tree.map(np.asarray, state)
        for start, checkpoint in reversed(checkpoints):
            active = min(chunk_steps, n_steps-start)
            block_weights = np.zeros(chunk_steps)
            block_weights[:active] = weights[start:start+active]
            replayed, tape = replay(jax.tree.map(jnp.asarray, checkpoint), alpha, inlet, jnp.array(active))
            for actual, expected in zip(replayed, next_expected):
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-10, atol=1e-10,
                                           err_msg="checkpoint replay changed trajectory")
            cotangent, increment = reverse(tape, alpha, inlet, jnp.asarray(block_weights), jnp.array(active), cotangent)
            material_gradient = material_gradient+increment
            if not np.all(np.isfinite(np.asarray(material_gradient))):
                raise RuntimeError("nonfinite material adjoint")
            next_expected = checkpoint
            if progress:
                progress("reverse", start, value, np.zeros(7))
    return dict(value=value, final=state, history=np.concatenate(history), snapshots=samples,
                material_gradient=material_gradient)
