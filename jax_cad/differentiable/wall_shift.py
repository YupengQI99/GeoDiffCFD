"""Opt-in material-mask correction; nominal geometry is not resized.

The angular radial normalization has unit gradient ON the nominal boundary.
It is NOT an exact signed distance throughout a finite-width interface band.
GPU experiments must independently pass their registered CPU gates.
"""
import math
import jax.numpy as jnp


def corrected_level_set(phi, angle, coefficients, modes, *, wall_shift=0., normalize_level_set=False):
    if not math.isfinite(wall_shift) or wall_shift < 0:
        raise ValueError('wall_shift must be a finite nonnegative inward offset')
    if type(normalize_level_set) is not bool:
        raise ValueError('normalize_level_set must be boolean')
    # No added arithmetic in the legacy branch: preserve existing forward bits.
    if not normalize_level_set and wall_shift == 0:
        return phi
    if normalize_level_set:
        count=len(modes)
        m=jnp.asarray(modes,dtype=coefficients.dtype)
        phase=angle[...,None]*m
        derivative=jnp.sum(m*(-coefficients[:count]*jnp.sin(phase)
                              +coefficients[count:]*jnp.cos(phase)),axis=-1)
        phi=phi/jnp.sqrt(1+derivative*derivative)
    return phi+wall_shift if wall_shift != 0 else phi
