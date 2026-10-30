"""Smooth Heaviside and delta functions for level-set methods.

All functions are JAX-compatible (jit, grad, vmap).
Convention: H(phi) ~ 1 for phi >> eps (fluid), ~ 0 for phi << -eps (solid).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def smooth_heaviside(phi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Cosine-based smooth Heaviside function.

    H_eps(phi) = 0           if phi < -eps
                 0.5*(1 + phi/eps + sin(pi*phi/eps)/pi)  if |phi| <= eps
                 1           if phi > eps

    Args:
        phi: Level-set field (any shape).
        eps: Smoothing half-width (typically 1-3 cells).

    Returns:
        Smoothed Heaviside values in [0, 1].
    """
    scaled = phi / eps
    interior = 0.5 * (1.0 + scaled + jnp.sin(jnp.pi * scaled) / jnp.pi)
    return jnp.where(phi < -eps, 0.0, jnp.where(phi > eps, 1.0, interior))


@jax.jit
def sigmoid_heaviside(phi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Sigmoid-based smooth Heaviside (better gradient flow for optimization).

    H_eps(phi) = sigmoid(phi / eps) = 1 / (1 + exp(-phi/eps))

    Args:
        phi: Level-set field.
        eps: Smoothing width parameter.

    Returns:
        Smoothed Heaviside values in (0, 1).
    """
    return jax.nn.sigmoid(phi / eps)


@jax.jit
def smooth_delta(phi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Smooth approximation to the Dirac delta function.

    delta_eps(phi) = (1 + cos(pi*phi/eps)) / (2*eps)  if |phi| <= eps
                     0                                  otherwise

    Args:
        phi: Level-set field.
        eps: Smoothing half-width.

    Returns:
        Smoothed delta values >= 0.
    """
    interior = (1.0 + jnp.cos(jnp.pi * phi / eps)) / (2.0 * eps)
    return jnp.where(jnp.abs(phi) <= eps, interior, 0.0)
