"""Derivative verification utilities for differentiable physics programs."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp


class TaylorTestResult(NamedTuple):
    """Errors and observed orders from a directional Taylor test."""

    epsilons: jax.Array
    zero_order_error: jax.Array
    first_order_error: jax.Array
    zero_order_rate: jax.Array
    first_order_rate: jax.Array
    directional_derivative: jax.Array


def tree_vdot(a: Any, b: Any) -> jax.Array:
    """Euclidean inner product between two identically structured PyTrees."""

    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    if treedef_a != treedef_b:
        raise ValueError("PyTrees have different structures")
    if not leaves_a:
        raise ValueError("PyTrees must contain at least one array leaf")
    terms = [jnp.vdot(x, y).real for x, y in zip(leaves_a, leaves_b)]
    return sum(terms[1:], terms[0])


def tree_add_scaled(x: Any, direction: Any, scale: float | jax.Array) -> Any:
    """Return ``x + scale * direction`` for arbitrary array PyTrees."""

    return jax.tree_util.tree_map(lambda a, d: a + scale * d, x, direction)


def normalize_direction(direction: Any) -> Any:
    """Normalize a PyTree direction in the Euclidean norm."""

    norm = jnp.sqrt(tree_vdot(direction, direction))
    if bool(norm == 0):
        raise ValueError("direction must be non-zero")
    return jax.tree_util.tree_map(lambda d: d / norm, direction)


def _observed_rates(errors: jax.Array, epsilons: jax.Array) -> jax.Array:
    tiny = jnp.finfo(errors.dtype).tiny
    numerator = jnp.log(jnp.maximum(errors[:-1], tiny) / jnp.maximum(errors[1:], tiny))
    denominator = jnp.log(epsilons[:-1] / epsilons[1:])
    return numerator / denominator


def directional_taylor_test(
    fun: Callable[[Any], jax.Array],
    point: Any,
    direction: Any,
    epsilons: Sequence[float] = (1e-1, 5e-2, 2.5e-2, 1.25e-2),
    *,
    normalize: bool = True,
) -> TaylorTestResult:
    """Check first- and second-order directional Taylor behaviour.

    For a differentiable scalar function ``f`` this evaluates

    ``E0(h) = |f(x+h d)-f(x)|`` and
    ``E1(h) = |f(x+h d)-f(x)-h grad(f)(x).d|``.

    A correct derivative should produce asymptotic rates close to one for
    ``E0`` and two for ``E1``.  This test is substantially more informative
    than comparing one gradient component at one finite-difference step.
    """

    eps = jnp.asarray(epsilons)
    if eps.ndim != 1 or eps.size < 2:
        raise ValueError("epsilons must be a one-dimensional sequence of length >= 2")
    if bool(jnp.any(eps <= 0)):
        raise ValueError("all epsilons must be positive")
    if bool(jnp.any(eps[:-1] <= eps[1:])):
        raise ValueError("epsilons must be strictly decreasing")

    d = normalize_direction(direction) if normalize else direction
    f0, grad = jax.value_and_grad(fun)(point)
    directional = tree_vdot(grad, d)

    values = jnp.stack([fun(tree_add_scaled(point, d, h)) for h in eps])
    e0 = jnp.abs(values - f0)
    e1 = jnp.abs(values - f0 - eps * directional)

    return TaylorTestResult(
        epsilons=eps,
        zero_order_error=e0,
        first_order_error=e1,
        zero_order_rate=_observed_rates(e0, eps),
        first_order_rate=_observed_rates(e1, eps),
        directional_derivative=directional,
    )
