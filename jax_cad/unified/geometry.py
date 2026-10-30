"""Geometry-native differentiable parameterizations.

The routines in this module expose low-dimensional *geometric* variables rather
than cell-wise material variables.  They map those variables to a smooth
level-set field, after which :mod:`jax_cad.unified.materials` supplies the common
Eulerian material-field representation consumed by physics solvers.

The first parameterization is a periodic star-shaped boundary represented by
Fourier modes of its log-radius.  A differentiable normalization fixes the
cross-sectional area exactly with respect to the angular quadrature.  This
prevents a flow optimizer from improving the objective by trivially deleting or
shrinking the obstacle and provides a clean test of the complete pullback

    scalar result -> physics state -> material field -> geometry coefficients.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp

from jax_cad.unified.maps import periodic_delta
from jax_cad.unified.materials import MaterialFields, fields_from_sdf

Array = jax.Array


def _mode_array(modes: Sequence[int], dtype) -> Array:
    mode_tuple = tuple(int(mode) for mode in modes)
    if not mode_tuple:
        raise ValueError("at least one Fourier mode is required")
    if any(mode < 1 for mode in mode_tuple):
        raise ValueError("Fourier modes must be positive integers")
    if len(set(mode_tuple)) != len(mode_tuple):
        raise ValueError("Fourier modes must be unique")
    return jnp.asarray(mode_tuple, dtype=dtype)


def _split_coefficients(coefficients: Array, modes: Sequence[int]) -> tuple[Array, Array]:
    coefficients = jnp.asarray(coefficients)
    n_modes = len(tuple(modes))
    expected = (2 * n_modes,)
    if coefficients.shape != expected:
        raise ValueError(
            f"coefficients must have shape {expected}: cosine coefficients "
            "followed by sine coefficients"
        )
    return coefficients[:n_modes], coefficients[n_modes:]


def fourier_log_radius(
    angle: Array,
    coefficients: Array,
    modes: Sequence[int],
) -> Array:
    """Evaluate the unnormalised Fourier expansion of the log-radius.

    ``coefficients = [a_1, ..., a_M, b_1, ..., b_M]`` and

    .. math::

        f(\vartheta)=\sum_m a_m\cos(m\vartheta)+b_m\sin(m\vartheta).
    """

    coefficients = jnp.asarray(coefficients)
    cosine_coefficients, sine_coefficients = _split_coefficients(
        coefficients, modes
    )
    mode_values = _mode_array(modes, coefficients.dtype)
    phase = jnp.asarray(angle)[..., None] * mode_values
    return jnp.sum(
        cosine_coefficients * jnp.cos(phase)
        + sine_coefficients * jnp.sin(phase),
        axis=-1,
    )


def fourier_area_log_normalization(
    coefficients: Array,
    modes: Sequence[int],
    *,
    n_quadrature: int = 512,
) -> Array:
    """Return the log-scale that preserves the reference cross-sectional area.

    With ``r(theta) = R exp(f(theta) - c)``, choosing

    ``c = 0.5 log(mean(exp(2 f)))``

    gives ``mean(r**2) = R**2`` and hence
    ``0.5 * integral(r**2 dtheta) = pi * R**2`` under the same quadrature.
    """

    if n_quadrature < 16:
        raise ValueError("n_quadrature must be at least 16")
    coefficients = jnp.asarray(coefficients)
    angles = (
        2.0
        * jnp.pi
        * (jnp.arange(n_quadrature, dtype=coefficients.dtype) + 0.5)
        / n_quadrature
    )
    log_radius = fourier_log_radius(angles, coefficients, modes)
    return 0.5 * jnp.log(jnp.mean(jnp.exp(2.0 * log_radius)))


def fourier_boundary_radius(
    angle: Array,
    coefficients: Array,
    *,
    base_radius: float,
    modes: Sequence[int],
    n_quadrature: int = 512,
) -> Array:
    """Evaluate the positive, area-normalised radial boundary function."""

    if base_radius <= 0:
        raise ValueError("base_radius must be positive")
    coefficients = jnp.asarray(coefficients)
    normalization = fourier_area_log_normalization(
        coefficients,
        modes,
        n_quadrature=n_quadrature,
    )
    return base_radius * jnp.exp(
        fourier_log_radius(angle, coefficients, modes) - normalization
    )


def periodic_fourier_level_set(
    x: Array,
    y: Array,
    coefficients: Array,
    *,
    center: tuple[float, float] | Array,
    base_radius: float,
    modes: Sequence[int],
    lx: float,
    ly: float,
    n_quadrature: int = 512,
) -> Array:
    """Construct an area-preserving periodic star-shaped level-set field.

    Positive values denote fluid and negative values denote the immersed solid.
    The field is exact for the circular reference state and is a radial
    level-set, rather than an exact Euclidean signed distance, for nonzero modes.
    This distinction is explicit because the smoothing width must be interpreted
    with respect to the chosen level-set parameterization.
    """

    if lx <= 0 or ly <= 0:
        raise ValueError("period lengths must be positive")
    coefficients = jnp.asarray(coefficients)
    center_array = jnp.asarray(center, dtype=coefficients.dtype)
    if center_array.shape != (2,):
        raise ValueError("center must contain exactly two coordinates")

    delta_x = periodic_delta(x, center_array[0], lx)
    delta_y = periodic_delta(y, center_array[1], ly)
    angle = jnp.arctan2(delta_y, delta_x)
    radius = jnp.sqrt(
        delta_x * delta_x
        + delta_y * delta_y
        + jnp.finfo(coefficients.dtype).tiny
    )
    boundary_radius = fourier_boundary_radius(
        angle,
        coefficients,
        base_radius=base_radius,
        modes=modes,
        n_quadrature=n_quadrature,
    )
    return radius - boundary_radius


def fourier_shape_area(
    coefficients: Array,
    *,
    base_radius: float,
    modes: Sequence[int],
    n_quadrature: int = 2048,
) -> Array:
    """Evaluate the radial cross-sectional area by angular quadrature."""

    coefficients = jnp.asarray(coefficients)
    angles = (
        2.0
        * jnp.pi
        * (jnp.arange(n_quadrature, dtype=coefficients.dtype) + 0.5)
        / n_quadrature
    )
    radius = fourier_boundary_radius(
        angles,
        coefficients,
        base_radius=base_radius,
        modes=modes,
        n_quadrature=n_quadrature,
    )
    return jnp.pi * jnp.mean(radius * radius)


def fourier_mode_sobolev_energy(
    coefficients: Array,
    modes: Sequence[int],
    *,
    order: int = 2,
) -> Array:
    """Penalise high-frequency geometric modes without coupling to the solver."""

    if order < 0:
        raise ValueError("order must be non-negative")
    coefficients = jnp.asarray(coefficients)
    cosine_coefficients, sine_coefficients = _split_coefficients(
        coefficients, modes
    )
    mode_values = _mode_array(modes, coefficients.dtype)
    weights = mode_values ** (2 * order)
    return jnp.sum(
        weights
        * (cosine_coefficients * cosine_coefficients
           + sine_coefficients * sine_coefficients)
    )


def make_periodic_fourier_material_map(
    x: Array,
    y: Array,
    *,
    center: tuple[float, float] | Array,
    base_radius: float,
    modes: Sequence[int],
    lx: float,
    ly: float,
    alpha_max: float,
    epsilon: float,
    n_quadrature: int = 512,
    profile: str = "compact",
    conductivity_fluid: float | None = None,
    conductivity_solid: float | None = None,
    heat_capacity_fluid: float | None = None,
    heat_capacity_solid: float | None = None,
) -> Callable[[Array], MaterialFields]:
    """Create ``Fourier geometry coefficients -> MaterialFields``.

    The returned callable obeys the same material-field contract as analytic
    primitive SDFs, sampled SDF offsets, and density topology variables.
    """

    mode_tuple = tuple(int(mode) for mode in modes)

    def design_map(coefficients: Array) -> MaterialFields:
        phi = periodic_fourier_level_set(
            x,
            y,
            coefficients,
            center=center,
            base_radius=base_radius,
            modes=mode_tuple,
            lx=lx,
            ly=ly,
            n_quadrature=n_quadrature,
        )
        return fields_from_sdf(
            phi,
            alpha_max=alpha_max,
            epsilon=epsilon,
            profile=profile,
            conductivity_fluid=conductivity_fluid,
            conductivity_solid=conductivity_solid,
            heat_capacity_fluid=heat_capacity_fluid,
            heat_capacity_solid=heat_capacity_solid,
        )

    return design_map
