"""Implicit differentiation of converged fixed-point solvers.

The forward solver may use a convergence-based ``lax.while_loop``.  The custom
VJP does not backpropagate through those iterations.  Instead, at the converged
state ``s = F(s, q)`` it solves

    (I - dF/ds)^T lambda = cotangent

and returns ``(dF/dq)^T lambda``.  This keeps reverse storage independent of the
number of nonlinear/fixed-point iterations and provides a second derivative
backend under the same :class:`~jax_cad.unified.DifferentiableProgram` API.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import gmres


class FixedPointDiagnostics(NamedTuple):
    """Forward convergence diagnostics."""

    iterations: jax.Array
    residual_norm: jax.Array
    converged: jax.Array


def tree_l2_norm(tree: Any) -> jax.Array:
    """Euclidean norm of an array PyTree."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("tree must contain at least one array leaf")
    terms = [jnp.vdot(x, x).real for x in leaves]
    return jnp.sqrt(sum(terms[1:], terms[0]))


def tree_subtract(a: Any, b: Any) -> Any:
    """Subtract identically structured PyTrees."""

    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _zeros_like_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def make_implicit_fixed_point_solver(
    update: Callable[[Any, Any], Any],
    *,
    max_iterations: int = 10_000,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    linear_tol: float = 1e-9,
    linear_atol: float = 1e-11,
    linear_maxiter: int = 200,
    linear_restart: int = 40,
) -> tuple[Callable[[Any, Any], Any], Callable[[Any, Any], tuple[Any, FixedPointDiagnostics]]]:
    """Create an implicitly differentiated fixed-point solver.

    Parameters
    ----------
    update:
        Callable ``update(state, parameters) -> next_state``.
    max_iterations, atol, rtol:
        Convergence controls for the primal fixed-point iteration.
    linear_tol, linear_atol, linear_maxiter, linear_restart:
        Controls for the matrix-free transposed linear solve in the custom VJP.

    Returns
    -------
    solve:
        ``solve(parameters, initial_state) -> converged_state`` with a custom
        reverse rule based on the implicit function theorem.
    solve_with_diagnostics:
        Non-custom wrapper returning the same primal state plus convergence
        metadata.  It is intended for validation and logging, not for use inside
        an objective transformed by ``jax.grad``.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")

    def primal(parameters: Any, initial_state: Any) -> tuple[Any, FixedPointDiagnostics]:
        dtype = jax.tree_util.tree_leaves(initial_state)[0].dtype
        initial_error = jnp.asarray(jnp.inf, dtype=dtype)
        initial_iteration = jnp.asarray(0, dtype=jnp.int32)

        def cond(carry):
            iteration, state, error = carry
            scale = atol + rtol * jnp.maximum(tree_l2_norm(state), 1.0)
            return jnp.logical_and(iteration < max_iterations, error > scale)

        def body(carry):
            iteration, state, _ = carry
            next_state = update(state, parameters)
            error = tree_l2_norm(tree_subtract(next_state, state))
            return iteration + 1, next_state, error

        iterations, state, _ = jax.lax.while_loop(
            cond,
            body,
            (initial_iteration, initial_state, initial_error),
        )
        # Report the residual at the returned state, not the previous iterate.
        residual = tree_l2_norm(tree_subtract(update(state, parameters), state))
        threshold = atol + rtol * jnp.maximum(tree_l2_norm(state), 1.0)
        diagnostics = FixedPointDiagnostics(
            iterations=iterations,
            residual_norm=residual,
            converged=residual <= threshold,
        )
        return state, diagnostics

    @jax.custom_vjp
    def solve(parameters: Any, initial_state: Any) -> Any:
        state, _ = primal(parameters, initial_state)
        return state

    def solve_fwd(parameters: Any, initial_state: Any):
        state, _ = primal(parameters, initial_state)
        return state, (state, parameters, initial_state)

    def solve_bwd(residuals, state_cotangent):
        state, parameters, initial_state = residuals
        flat_state, unravel_state = ravel_pytree(state)
        flat_cotangent, _ = ravel_pytree(state_cotangent)

        def update_flat(flat_candidate):
            candidate = unravel_state(flat_candidate)
            updated = update(candidate, parameters)
            return ravel_pytree(updated)[0]

        _, state_pullback = jax.vjp(update_flat, flat_state)

        def transpose_residual_matvec(vector):
            return vector - state_pullback(vector)[0]

        adjoint, _ = gmres(
            transpose_residual_matvec,
            flat_cotangent,
            tol=linear_tol,
            atol=linear_atol,
            maxiter=linear_maxiter,
            restart=linear_restart,
        )
        adjoint_state = unravel_state(adjoint)

        _, parameter_pullback = jax.vjp(lambda q: update(state, q), parameters)
        parameter_cotangent = parameter_pullback(adjoint_state)[0]
        initial_cotangent = _zeros_like_tree(initial_state)
        return parameter_cotangent, initial_cotangent

    solve.defvjp(solve_fwd, solve_bwd)
    return solve, primal


def make_implicit_affine_solver(
    update: Callable[[Any, Any], Any],
    *,
    tol: float = 1e-12,
    atol: float = 1e-14,
    maxiter: int = 20_000,
    restart: int = 60,
    linear_tol: float = 1e-10,
    linear_atol: float = 1e-12,
    linear_maxiter: int = 40_000,
    linear_restart: int = 60,
    preconditioner: Callable[[Any], Any] | None = None,
) -> tuple[Callable[[Any, Any], Any], Callable[[Any, Any], tuple[Any, FixedPointDiagnostics]]]:
    """Krylov solve of the fixed point of an *affine* update map.

    For a pseudo-time map that is affine in the state, ``s = F(s, q)`` is the
    linear system ``(I - F_s) s = F(0, q)``.  Instead of Picard iteration the
    primal is obtained by matrix-free GMRES on that system, which reaches the
    same fixed point in far fewer applications of ``update``.  The reverse rule
    is the same implicit adjoint ``(I - F_s)^T lambda = cotangent`` used by
    :func:`make_implicit_fixed_point_solver`; both directions are therefore
    residual-based linear solves whose tolerances are exposed.

    ``preconditioner`` is an optional symmetric approximate inverse of
    ``I - F_s`` acting on state PyTrees; it is applied as a left preconditioner
    in both the primal and the transposed adjoint solve.  Because ``I - F_s``
    scales with the pseudo-time step, an unpreconditioned Krylov solve degrades
    as the grid is refined; a Fourier-space inverse of the constant-coefficient
    part restores grid-independent iteration counts.

    The returned ``primal`` reports the residual ``||F(s) - s||`` at the
    returned state; ``iterations`` is set to ``-1`` because JAX's GMRES does not
    expose its iteration count.
    """

    if maxiter < 1 or linear_maxiter < 1:
        raise ValueError("maxiter values must be positive")

    def _flat_preconditioner(unravel):
        if preconditioner is None:
            return None

        def apply(flat):
            return ravel_pytree(preconditioner(unravel(flat)))[0]

        return apply

    def primal(parameters: Any, initial_state: Any) -> tuple[Any, FixedPointDiagnostics]:
        flat0, unravel = ravel_pytree(initial_state)

        def update_flat(flat):
            return ravel_pytree(update(unravel(flat), parameters))[0]

        offset = update_flat(jnp.zeros_like(flat0))

        def matvec(flat):
            return flat - (update_flat(flat) - offset)

        flat_state, _ = gmres(
            matvec,
            offset,
            x0=flat0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            M=_flat_preconditioner(unravel),
            solve_method="batched",
        )
        state = unravel(flat_state)
        residual = tree_l2_norm(tree_subtract(update(state, parameters), state))
        threshold = atol + tol * jnp.maximum(tree_l2_norm(state), 1.0)
        diagnostics = FixedPointDiagnostics(
            iterations=jnp.asarray(-1, dtype=jnp.int32),
            residual_norm=residual,
            converged=residual <= 10.0 * threshold,
        )
        return state, diagnostics

    @jax.custom_vjp
    def solve(parameters: Any, initial_state: Any) -> Any:
        state, _ = primal(parameters, initial_state)
        return state

    def solve_fwd(parameters: Any, initial_state: Any):
        state, _ = primal(parameters, initial_state)
        return state, (state, parameters, initial_state)

    def solve_bwd(residuals, state_cotangent):
        state, parameters, initial_state = residuals
        flat_state, unravel_state = ravel_pytree(state)
        flat_cotangent, _ = ravel_pytree(state_cotangent)

        def update_flat(flat_candidate):
            return ravel_pytree(update(unravel_state(flat_candidate), parameters))[0]

        _, state_pullback = jax.vjp(update_flat, flat_state)

        def transpose_residual_matvec(vector):
            return vector - state_pullback(vector)[0]

        adjoint, _ = gmres(
            transpose_residual_matvec,
            flat_cotangent,
            tol=linear_tol,
            atol=linear_atol,
            maxiter=linear_maxiter,
            restart=linear_restart,
            M=_flat_preconditioner(unravel_state),
            solve_method="batched",
        )
        adjoint_state = unravel_state(adjoint)
        _, parameter_pullback = jax.vjp(lambda q: update(state, q), parameters)
        parameter_cotangent = parameter_pullback(adjoint_state)[0]
        return parameter_cotangent, _zeros_like_tree(initial_state)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve, primal


def fixed_point_adjoint(
    update: Callable[[Any, Any], Any],
    state: Any,
    parameters: Any,
    cotangent: Any,
    *,
    linear_tol: float = 1e-9,
    linear_atol: float = 1e-11,
    linear_maxiter: int = 200,
    linear_restart: int = 40,
) -> tuple[Any, jax.Array, jax.Array]:
    """Solve ``(I - F_s)^T lambda = cotangent`` and report its relative residual.

    This diagnostic mirrors the linear solve inside the custom VJP of
    :func:`make_implicit_fixed_point_solver` so that experiments can archive the
    adjoint residual next to the primal residual.  Returns
    ``(lambda, relative_residual, cotangent_norm)``.
    """

    flat_state, unravel_state = ravel_pytree(state)
    flat_cotangent, _ = ravel_pytree(cotangent)

    def update_flat(flat_candidate):
        return ravel_pytree(update(unravel_state(flat_candidate), parameters))[0]

    _, state_pullback = jax.vjp(update_flat, flat_state)

    def matvec(vector):
        return vector - state_pullback(vector)[0]

    adjoint, _ = gmres(
        matvec,
        flat_cotangent,
        tol=linear_tol,
        atol=linear_atol,
        maxiter=linear_maxiter,
        restart=linear_restart,
    )
    residual = jnp.linalg.norm(matvec(adjoint) - flat_cotangent)
    cotangent_norm = jnp.linalg.norm(flat_cotangent)
    return unravel_state(adjoint), residual / jnp.maximum(cotangent_norm, 1e-300), cotangent_norm
