"""Unified material-field PyTrees and geometry-to-material maps.

Geometry representations should not couple directly to one solver.  They map to
material fields, and each physics kernel consumes the subset it needs.  This
module implements the first common field contract for immersed flow and leaves
optional thermal fields in the same PyTree for later multiphysics solvers.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from jax_cad.unified.maps import circle_sdf_periodic

Array = jax.Array


class MaterialFields(NamedTuple):
    """Differentiable Eulerian material fields on a fixed grid.

    Parameters
    ----------
    fluid_fraction:
        Smooth phase indicator, one in fluid and zero in solid.
    momentum_resistance:
        Brinkman/Darcy resistance consumed by immersed flow solvers.
    level_set:
        Optional signed-distance or level-set field.  Density-based routes may
        leave it as ``None``.
    thermal_conductivity, volumetric_heat_capacity:
        Optional mixture fields for future thermo-fluid kernels.
    """

    fluid_fraction: Array
    momentum_resistance: Array
    level_set: Array | None
    thermal_conductivity: Array | None
    volumetric_heat_capacity: Array | None


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


PROFILES = ("compact", "sigmoid")


def compact_heaviside(phi: Array, epsilon: float) -> Array:
    """C^2 quintic smoothed Heaviside with compact support ``|phi| <= epsilon``.

    ``H(s) = 1/2 + (15/16) s - (5/8) s^3 + (3/16) s^5`` for ``s = phi/epsilon``
    in ``[-1, 1]``, ``H = 0`` for ``phi <= -epsilon`` and ``H = 1`` for
    ``phi >= epsilon``.  Its derivative ``(15/16)(1 - s^2)^2 / epsilon`` and
    second derivative vanish at ``|s| = 1``, so the material fields are exactly
    pure fluid (``alpha = 0``) and pure solid (``alpha = alpha_max``) outside the
    band and twice continuously differentiable in the geometry.  Unlike a
    logistic profile, no exponential resistance tail leaks into the fluid.
    """

    _validate_positive("epsilon", epsilon)
    s = jnp.clip(phi / epsilon, -1.0, 1.0)
    return 0.5 + s * (15.0 / 16.0 + s * s * (-5.0 / 8.0 + s * s * (3.0 / 16.0)))


def compact_delta(phi: Array, epsilon: float) -> Array:
    """Derivative of :func:`compact_heaviside` with respect to ``phi``."""

    _validate_positive("epsilon", epsilon)
    s = phi / epsilon
    inside = jnp.abs(s) < 1.0
    return jnp.where(inside, (15.0 / 16.0) * (1.0 - s * s) ** 2 / epsilon, 0.0)


def smoothed_heaviside(phi: Array, epsilon: float, profile: str = "compact") -> Array:
    """Return the smooth phase indicator ``chi`` (one in fluid, zero in solid)."""

    if profile == "compact":
        return compact_heaviside(phi, epsilon)
    if profile == "sigmoid":
        _validate_positive("epsilon", epsilon)
        return jax.nn.sigmoid(phi / epsilon)
    raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")


def fields_from_sdf(
    phi: Array,
    *,
    alpha_max: float,
    epsilon: float,
    profile: str = "compact",
    conductivity_fluid: float | None = None,
    conductivity_solid: float | None = None,
    heat_capacity_fluid: float | None = None,
    heat_capacity_solid: float | None = None,
) -> MaterialFields:
    """Convert an SDF/level set into a common material-field PyTree.

    ``profile`` selects the phase indicator: the default compact C^2 quintic
    Heaviside (exactly zero resistance beyond ``epsilon`` from the interface) or
    the legacy logistic ``sigmoid`` used in earlier archived records.
    """

    _validate_positive("epsilon", epsilon)
    if alpha_max < 0:
        raise ValueError("alpha_max must be non-negative")

    chi = smoothed_heaviside(phi, epsilon, profile)
    alpha = alpha_max * (1.0 - chi)

    conductivity = None
    if conductivity_fluid is not None or conductivity_solid is not None:
        if conductivity_fluid is None or conductivity_solid is None:
            raise ValueError("both fluid and solid conductivity must be supplied")
        conductivity = chi * conductivity_fluid + (1.0 - chi) * conductivity_solid

    heat_capacity = None
    if heat_capacity_fluid is not None or heat_capacity_solid is not None:
        if heat_capacity_fluid is None or heat_capacity_solid is None:
            raise ValueError("both fluid and solid heat capacity must be supplied")
        heat_capacity = chi * heat_capacity_fluid + (1.0 - chi) * heat_capacity_solid

    return MaterialFields(
        fluid_fraction=chi,
        momentum_resistance=alpha,
        level_set=phi,
        thermal_conductivity=conductivity,
        volumetric_heat_capacity=heat_capacity,
    )


def make_periodic_circle_material_map(
    x: Array,
    y: Array,
    *,
    lx: float,
    ly: float,
    alpha_max: float,
    epsilon: float,
    profile: str = "compact",
    conductivity_fluid: float | None = None,
    conductivity_solid: float | None = None,
    heat_capacity_fluid: float | None = None,
    heat_capacity_solid: float | None = None,
) -> Callable[[Array], MaterialFields]:
    """Create an analytic-circle ``theta -> MaterialFields`` map."""

    def design_map(theta: Array) -> MaterialFields:
        phi = circle_sdf_periodic(x, y, theta, lx=lx, ly=ly)
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


def make_sampled_sdf_material_map(
    phi0: Array,
    *,
    alpha_max: float,
    epsilon: float,
    profile: str = "compact",
) -> Callable[[Array], MaterialFields]:
    """Create an offline-SDF ``offset -> MaterialFields`` map."""

    phi0 = jnp.asarray(phi0)

    def design_map(offset: Array) -> MaterialFields:
        return fields_from_sdf(
            phi0 + offset, alpha_max=alpha_max, epsilon=epsilon, profile=profile
        )

    return design_map


def build_cone_filter_kernel(
    radius: float,
    *,
    dx: float,
    dy: float,
    dtype=jnp.float32,
) -> Array:
    """Build a normalized anisotropic-grid cone kernel in physical units."""

    _validate_positive("radius", radius)
    _validate_positive("dx", dx)
    _validate_positive("dy", dy)
    half_x = int(jnp.ceil(radius / dx))
    half_y = int(jnp.ceil(radius / dy))
    offsets_x = jnp.arange(-half_x, half_x + 1, dtype=dtype) * dx
    offsets_y = jnp.arange(-half_y, half_y + 1, dtype=dtype) * dy
    xx, yy = jnp.meshgrid(offsets_x, offsets_y, indexing="ij")
    weights = jnp.maximum(radius - jnp.sqrt(xx * xx + yy * yy), 0.0)
    total = jnp.sum(weights)
    return weights / total


def periodic_density_filter(density: Array, kernel: Array) -> Array:
    """Apply a constant-preserving periodic density filter."""

    if density.ndim != 2 or kernel.ndim != 2:
        raise ValueError("density and kernel must both be two-dimensional")
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("kernel dimensions must be odd")

    half_x = kernel.shape[0] // 2
    half_y = kernel.shape[1] // 2
    filtered = jnp.zeros_like(density)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            shift = (i - half_x, j - half_y)
            filtered = filtered + kernel[i, j] * jnp.roll(
                density,
                shift=shift,
                axis=(0, 1),
            )
    return filtered


def heaviside_projection(density: Array, beta: float, eta: float = 0.5) -> Array:
    """Smooth density projection with values in approximately ``[0, 1]``."""

    _validate_positive("beta", beta)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    denominator = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    numerator = jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))
    return numerator / denominator


def ramp_resistance(fluid_fraction: Array, alpha_max: float, q: float) -> Array:
    """RAMP interpolation from fluid fraction to momentum resistance."""

    if alpha_max < 0:
        raise ValueError("alpha_max must be non-negative")
    _validate_positive("q", q)
    return alpha_max * q * (1.0 - fluid_fraction) / (q + fluid_fraction)


def make_density_material_map(
    kernel: Array,
    *,
    beta: float,
    eta: float,
    alpha_max: float,
    ramp_q: float,
) -> Callable[[Array], MaterialFields]:
    """Create ``density logits -> MaterialFields`` for topology optimization."""

    def design_map(logits: Array) -> MaterialFields:
        raw = jax.nn.sigmoid(logits)
        filtered = periodic_density_filter(raw, kernel)
        chi = heaviside_projection(filtered, beta=beta, eta=eta)
        alpha = ramp_resistance(chi, alpha_max=alpha_max, q=ramp_q)
        return MaterialFields(
            fluid_fraction=chi,
            momentum_resistance=alpha,
            level_set=None,
            thermal_conductivity=None,
            volumetric_heat_capacity=None,
        )

    return design_map
