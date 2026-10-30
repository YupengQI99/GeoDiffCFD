"""Reusable differentiable maps from design variables to physics fields."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def periodic_delta(coordinate: Array, center: Array, period: float) -> Array:
    """Shortest signed distance on a periodic interval."""

    return jnp.mod(coordinate - center + 0.5 * period, period) - 0.5 * period


def circle_sdf_periodic(
    x: Array,
    y: Array,
    theta: Array,
    *,
    lx: float,
    ly: float,
) -> Array:
    """Exact cell-centred SDF of a periodic circle.

    ``theta = [cx, cy, radius]``.  Positive values denote fluid and negative
    values denote the circular solid.
    """

    theta = jnp.asarray(theta)
    if theta.shape != (3,):
        raise ValueError(f"theta must have shape (3,), received {theta.shape}")
    cx, cy, radius = theta
    dx = periodic_delta(x, cx, lx)
    dy = periodic_delta(y, cy, ly)
    return jnp.sqrt(dx * dx + dy * dy + jnp.finfo(theta.dtype).tiny) - radius


def sdf_to_brinkman(
    phi: Array, alpha_max: float, epsilon: float, profile: str = "compact"
) -> Array:
    """Map a signed distance field to a smooth Brinkman resistance field.

    ``profile="compact"`` uses the C^2 quintic Heaviside with support
    ``|phi| <= epsilon`` (see :mod:`jax_cad.unified.materials`);
    ``profile="sigmoid"`` reproduces the legacy logistic map.
    """

    from jax_cad.unified.materials import smoothed_heaviside

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if alpha_max < 0:
        raise ValueError("alpha_max must be non-negative")
    return alpha_max * (1.0 - smoothed_heaviside(phi, epsilon, profile))


def make_periodic_circle_brinkman_map(
    x: Array,
    y: Array,
    *,
    lx: float,
    ly: float,
    alpha_max: float,
    epsilon: float,
    profile: str = "compact",
) -> Callable[[Array], Array]:
    """Create ``theta -> alpha`` for a periodic circular obstacle."""

    def design_map(theta: Array) -> Array:
        phi = circle_sdf_periodic(x, y, theta, lx=lx, ly=ly)
        return sdf_to_brinkman(phi, alpha_max=alpha_max, epsilon=epsilon, profile=profile)

    return design_map


def make_sampled_sdf_offset_map(
    phi0: Array,
    *,
    alpha_max: float,
    epsilon: float,
    profile: str = "compact",
) -> Callable[[Array], Array]:
    """Create ``offset -> alpha`` for an offline sampled SDF.

    Positive offset increases ``phi`` and therefore expands the region treated
    as fluid under the convention ``phi > 0`` in fluid.
    """

    phi0 = jnp.asarray(phi0)

    def design_map(offset: Array) -> Array:
        return sdf_to_brinkman(phi0 + offset, alpha_max, epsilon, profile)

    return design_map
