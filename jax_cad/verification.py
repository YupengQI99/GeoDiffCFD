"""Shared numerical-verification utilities for canonical CFD cases."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .differentiable.mac_channel import (
    Array,
    ChannelGrid2D,
    ChannelParams,
    ChannelState,
    conservative_momentum_fluxes,
    interpolate_alpha_to_faces,
    steady_residual,
)


class ErrorNorms(NamedTuple):
    l2_absolute: Array
    linf_absolute: Array
    l2_relative: Array
    linf_relative: Array


class DFGCircleTrace(NamedTuple):
    """Pure-fluid-side traces reconstructed on the nominal DFG cylinder."""

    theta: Array
    pressure: Array
    tangential_normal_derivative: Array
    normal_normal_derivative: Array
    tangential_surface_velocity: Array
    normal_surface_velocity: Array
    pressure_fit_rms: Array
    tangential_fit_rms: Array
    normal_fit_rms: Array
    pressure_fit_condition_number: Array
    velocity_fit_condition_number: Array
    pressure_value_weight_l1: Array
    velocity_value_weight_l1: Array
    velocity_derivative_weight_l1_normalized: Array
    minimum_stencil_signed_distance: Array
    minimum_interpolation_node_signed_distance: Array
    minimum_velocity_alpha_support_signed_distance: Array


class DFGCircleTraction(NamedTuple):
    """DFG force components and their pressure/viscous decomposition."""

    force_x: Array
    force_y: Array
    pressure_force_x: Array
    pressure_force_y: Array
    viscous_force_x: Array
    viscous_force_y: Array
    drag_density: Array
    lift_density: Array


class MACStaggeredControlVolume(NamedTuple):
    """Inclusive native-MAC index bounds for an interior momentum balance.

    The u- and v-momentum equations live on different dual control volumes, so
    an exactly conservative discrete force necessarily uses two rectangles
    offset by half a cell.  All bounds are inclusive.
    """

    u_i_start: int
    u_i_stop: int
    u_j_start: int
    u_j_stop: int
    v_i_start: int
    v_i_stop: int
    v_j_start: int
    v_j_stop: int


class MACControlVolumeMomentumBalance(NamedTuple):
    """Steady finite-Brinkman reaction inferred from native MAC fluxes.

    This is a discrete momentum-balance diagnostic, not a sharp-interface
    surface-traction measurement.  Contributions have the sign of force on the
    stationary penalised body.
    """

    inferred_brinkman_reaction_x: Array
    inferred_brinkman_reaction_y: Array
    convective_contribution_x: Array
    convective_contribution_y: Array
    pressure_contribution_x: Array
    pressure_contribution_y: Array
    viscous_contribution_x: Array
    viscous_contribution_y: Array
    source_contribution_x: Array
    source_contribution_y: Array
    captured_brinkman_force_x: Array
    captured_brinkman_force_y: Array
    global_brinkman_force_x: Array
    global_brinkman_force_y: Array
    uncaptured_brinkman_force_x: Array
    uncaptured_brinkman_force_y: Array
    momentum_residual_integral_x: Array
    momentum_residual_integral_y: Array
    algebraic_closure_x: Array
    algebraic_closure_y: Array


def error_norms(
    numerical: Array, reference: Array, weights: Array | None = None
) -> ErrorNorms:
    """Return weighted absolute and relative L2/L-infinity errors."""

    numerical = jnp.asarray(numerical)
    reference = jnp.asarray(reference)
    if numerical.shape != reference.shape:
        raise ValueError(
            f"numerical and reference shapes differ: {numerical.shape} != "
            f"{reference.shape}"
        )
    if weights is None:
        weights = jnp.ones_like(reference)
    else:
        weights = jnp.broadcast_to(jnp.asarray(weights), reference.shape)
    if bool(jnp.any(weights < 0.0)):
        raise ValueError("weights must be nonnegative")
    weight_sum = jnp.sum(weights)
    if float(weight_sum) <= 0.0:
        raise ValueError("weights must have positive sum")

    difference = numerical - reference
    l2_absolute = jnp.sqrt(jnp.sum(weights * difference * difference) / weight_sum)
    linf_absolute = jnp.max(jnp.abs(difference))
    reference_l2 = jnp.sqrt(jnp.sum(weights * reference * reference) / weight_sum)
    reference_linf = jnp.max(jnp.abs(reference))
    tiny = jnp.finfo(jnp.result_type(numerical, reference)).tiny
    return ErrorNorms(
        l2_absolute,
        linf_absolute,
        l2_absolute / jnp.maximum(reference_l2, tiny),
        linf_absolute / jnp.maximum(reference_linf, tiny),
    )


def observed_orders(spacings: Array, errors: Array) -> Array:
    """Return pairwise observed orders for successively refined grids."""

    spacings = jnp.asarray(spacings)
    errors = jnp.asarray(errors)
    if spacings.ndim != 1 or errors.ndim != 1 or spacings.shape != errors.shape:
        raise ValueError("spacings and errors must be equal-length 1-D arrays")
    if spacings.size < 2:
        raise ValueError("at least two grids are required")
    if bool(jnp.any(spacings <= 0.0)) or bool(jnp.any(errors <= 0.0)):
        raise ValueError("spacings and errors must be positive")
    if bool(jnp.any(spacings[:-1] <= spacings[1:])):
        raise ValueError("spacings must be strictly decreasing")
    return jnp.log(errors[:-1] / errors[1:]) / jnp.log(
        spacings[:-1] / spacings[1:]
    )


def poiseuille_profile(y: Array, height: float, mean_velocity: float) -> Array:
    """Continuum planar-Poiseuille profile for an independently fixed flux."""

    if height <= 0.0:
        raise ValueError("height must be positive")
    y = jnp.asarray(y)
    return 6.0 * mean_velocity * y * (height - y) / (height * height)


def poiseuille_pressure_gradient(
    viscosity: float, height: float, mean_velocity: float
) -> float:
    """Return ``dp/dx`` for planar Poiseuille flow at unit density."""

    if viscosity < 0.0 or height <= 0.0:
        raise ValueError("viscosity must be nonnegative and height positive")
    return -12.0 * viscosity * mean_velocity / (height * height)


def parabolic_inlet_face_averages(
    grid: ChannelGrid2D, maximum_velocity: float, *, dtype=jnp.float32
) -> Array:
    """Return exact boundary-face averages of the DFG parabolic inlet.

    Point samples at MAC face centres overestimate the integral of the
    quadratic profile on a finite grid.  Integrating each face analytically
    makes the discrete flux exactly ``(2/3) * U_max * H`` without rescaling
    from a numerical solution.
    """

    if maximum_velocity < 0.0:
        raise ValueError("maximum_velocity must be nonnegative")
    edges = jnp.arange(grid.ny + 1, dtype=dtype) * grid.dy

    def antiderivative(y):
        return 0.5 * grid.ly * y * y - y * y * y / 3.0

    integrals = (
        4.0
        * maximum_velocity
        / (grid.ly * grid.ly)
        * (antiderivative(edges[1:]) - antiderivative(edges[:-1]))
    )
    return integrals / grid.dy


def linear_slope(x: Array, values: Array) -> Array:
    """Least-squares slope with an intercept, without assembling a matrix."""

    x = jnp.asarray(x)
    values = jnp.asarray(values)
    if x.ndim != 1 or values.ndim != 1 or x.shape != values.shape:
        raise ValueError("x and values must be equal-length 1-D arrays")
    centred_x = x - jnp.mean(x)
    denominator = jnp.sum(centred_x * centred_x)
    if float(denominator) <= 0.0:
        raise ValueError("x coordinates must not all be equal")
    return jnp.sum(centred_x * (values - jnp.mean(values))) / denominator


def cavity_streamfunction(state: ChannelState, grid: ChannelGrid2D) -> Array:
    """Integrate ``u = d psi / dy`` onto MAC grid vertices from the bottom wall."""

    if state.u.shape != (grid.nx + 1, grid.ny):
        raise ValueError("state.u is incompatible with grid")
    psi = jnp.zeros((grid.nx + 1, grid.ny + 1), dtype=state.u.dtype)
    return psi.at[:, 1:].set(jnp.cumsum(state.u * grid.dy, axis=1))


def primary_vortex_vertex(
    state: ChannelState, grid: ChannelGrid2D
) -> tuple[Array, Array, Array]:
    """Return the interior vertex of minimum streamfunction and its value."""

    psi = cavity_streamfunction(state, grid)
    interior = psi[1:-1, 1:-1]
    flat_index = jnp.argmin(interior)
    ix, iy = jnp.unravel_index(flat_index, interior.shape)
    x = (ix + 1).astype(psi.dtype) * grid.dx
    y = (iy + 1).astype(psi.dtype) * grid.dy
    return x, y, interior[ix, iy]


def bilinear_cell_value(
    field: Array, x: float, y: float, grid: ChannelGrid2D
) -> Array:
    """Interpolate a cell-centred field at a point inside the grid bounds."""

    if field.shape != (grid.nx, grid.ny):
        raise ValueError("field is incompatible with grid")
    if not (0.0 <= x <= grid.lx and 0.0 <= y <= grid.ly):
        raise ValueError("interpolation point lies outside the grid")
    xi = jnp.asarray(x / grid.dx - 0.5, dtype=field.dtype)
    yi = jnp.asarray(y / grid.dy - 0.5, dtype=field.dtype)
    i0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, grid.nx - 2)
    j0 = jnp.clip(jnp.floor(yi).astype(jnp.int32), 0, grid.ny - 2)
    tx = jnp.clip(xi - i0.astype(field.dtype), 0.0, 1.0)
    ty = jnp.clip(yi - j0.astype(field.dtype), 0.0, 1.0)
    f00 = field[i0, j0]
    f10 = field[i0 + 1, j0]
    f01 = field[i0, j0 + 1]
    f11 = field[i0 + 1, j0 + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def _bilinear_uniform_value(
    field: Array,
    x: Array,
    y: Array,
    *,
    x_origin: float,
    y_origin: float,
    dx: float,
    dy: float,
) -> Array:
    """Interpolate a two-dimensional field on a uniform Cartesian lattice."""

    if field.ndim != 2 or field.shape[0] < 2 or field.shape[1] < 2:
        raise ValueError("field must be a two-dimensional lattice of at least 2 x 2")
    xi = (jnp.asarray(x, dtype=field.dtype) - x_origin) / dx
    yi = (jnp.asarray(y, dtype=field.dtype) - y_origin) / dy
    tolerance = 32.0 * jnp.finfo(field.dtype).eps
    if bool(
        jnp.any(xi < -tolerance)
        or jnp.any(xi > field.shape[0] - 1 + tolerance)
        or jnp.any(yi < -tolerance)
        or jnp.any(yi > field.shape[1] - 1 + tolerance)
    ):
        raise ValueError("interpolation points leave the lattice support")
    i0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, field.shape[0] - 2)
    j0 = jnp.clip(jnp.floor(yi).astype(jnp.int32), 0, field.shape[1] - 2)
    tx = jnp.clip(xi - i0.astype(field.dtype), 0.0, 1.0)
    ty = jnp.clip(yi - j0.astype(field.dtype), 0.0, 1.0)
    return (
        (1.0 - tx) * (1.0 - ty) * field[i0, j0]
        + tx * (1.0 - ty) * field[i0 + 1, j0]
        + (1.0 - tx) * ty * field[i0, j0 + 1]
        + tx * ty * field[i0 + 1, j0 + 1]
    )


def _polynomial_trace(
    offsets: Array,
    samples: Array,
    degree: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Fit radial samples and return trace, derivative, and diagnostics."""

    scale = offsets[-1]
    normalized_offsets = offsets / scale
    powers = jnp.arange(0, degree + 1, dtype=offsets.dtype)
    design = normalized_offsets[:, None] ** powers[None, :]
    pseudoinverse = jnp.linalg.pinv(design)
    coefficients = pseudoinverse @ samples.T
    fitted = (design @ coefficients).T
    fit_rms = jnp.sqrt(jnp.mean((samples - fitted) ** 2, axis=1))
    value = coefficients[0, :]
    derivative = coefficients[1, :] / scale
    return (
        value,
        derivative,
        fit_rms,
        jnp.linalg.cond(design),
        jnp.sum(jnp.abs(pseudoinverse[0, :])),
        jnp.sum(jnp.abs(pseudoinverse[1, :])),
    )


def _minimum_circle_stencil_signed_distance(
    x: Array,
    y: Array,
    *,
    field_shape: tuple[int, int],
    x_origin: float,
    y_origin: float,
    dx: float,
    dy: float,
    centre_x: float,
    centre_y: float,
    radius: float,
) -> Array:
    """Return the closest circle distance among every bilinear stencil node."""

    xi = (jnp.asarray(x) - x_origin) / dx
    yi = (jnp.asarray(y) - y_origin) / dy
    i0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, field_shape[0] - 2)
    j0 = jnp.clip(jnp.floor(yi).astype(jnp.int32), 0, field_shape[1] - 2)
    minimum = jnp.asarray(jnp.inf, dtype=jnp.result_type(x, y))
    for di in (0, 1):
        for dj in (0, 1):
            node_x = x_origin + (i0 + di) * dx
            node_y = y_origin + (j0 + dj) * dy
            signed_distance = (
                jnp.sqrt((node_x - centre_x) ** 2 + (node_y - centre_y) ** 2)
                - radius
            )
            minimum = jnp.minimum(minimum, jnp.min(signed_distance))
    return minimum


def _minimum_velocity_alpha_support_signed_distance(
    x: Array,
    y: Array,
    *,
    component: str,
    grid: ChannelGrid2D,
    centre_x: float,
    centre_y: float,
    radius: float,
) -> Array:
    """Return the closest cell centre feeding a sampled staggered ``alpha``.

    Bilinear velocity interpolation touches four native staggered nodes.  Each
    node's Brinkman coefficient is itself an average of adjacent cell-centred
    coefficients, as defined by :func:`interpolate_alpha_to_faces`.  Checking
    only the geometric locations of the velocity nodes can therefore label a
    stencil pure-fluid even though one of those underlying cell centres is in
    the diffuse interface.  This routine follows that complete dependency
    stencil for either velocity component.
    """

    dtype = jnp.result_type(x, y)
    if component == "u":
        field_shape = (grid.nx + 1, grid.ny)
        x_origin = 0.0
        y_origin = 0.5 * grid.dy
    elif component == "v":
        field_shape = (grid.nx, grid.ny + 1)
        x_origin = 0.5 * grid.dx
        y_origin = 0.0
    else:
        raise ValueError("component must be 'u' or 'v'")

    xi = (jnp.asarray(x) - x_origin) / grid.dx
    yi = (jnp.asarray(y) - y_origin) / grid.dy
    i0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, field_shape[0] - 2)
    j0 = jnp.clip(jnp.floor(yi).astype(jnp.int32), 0, field_shape[1] - 2)
    minimum = jnp.asarray(jnp.inf, dtype=dtype)
    for di in (0, 1):
        for dj in (0, 1):
            face_i = i0 + di
            face_j = j0 + dj
            if component == "u":
                # alpha_u[i, j] averages alpha[i - 1, j] and alpha[i, j],
                # with the cell index clipped at the inlet and outlet.
                for cell_di in (-1, 0):
                    cell_i = jnp.clip(face_i + cell_di, 0, grid.nx - 1)
                    cell_j = face_j
                    cell_x = (cell_i.astype(dtype) + 0.5) * grid.dx
                    cell_y = (cell_j.astype(dtype) + 0.5) * grid.dy
                    signed_distance = (
                        jnp.sqrt(
                            (cell_x - centre_x) ** 2
                            + (cell_y - centre_y) ** 2
                        )
                        - radius
                    )
                    minimum = jnp.minimum(minimum, jnp.min(signed_distance))
            else:
                # alpha_v[i, j] averages alpha[i, j - 1] and alpha[i, j],
                # with the cell index clipped at the horizontal walls.
                for cell_dj in (-1, 0):
                    cell_i = face_i
                    cell_j = jnp.clip(face_j + cell_dj, 0, grid.ny - 1)
                    cell_x = (cell_i.astype(dtype) + 0.5) * grid.dx
                    cell_y = (cell_j.astype(dtype) + 0.5) * grid.dy
                    signed_distance = (
                        jnp.sqrt(
                            (cell_x - centre_x) ** 2
                            + (cell_y - centre_y) ** 2
                        )
                        - radius
                    )
                    minimum = jnp.minimum(minimum, jnp.min(signed_distance))
    return minimum


def dfg_circle_exterior_trace(
    state: ChannelState,
    grid: ChannelGrid2D,
    centre_x: float,
    centre_y: float,
    radius: float,
    h_reference: float,
    offsets_over_h: Array,
    *,
    quadrature_points: int,
    pressure_degree: int = 2,
    tangential_velocity_degree: int = 3,
    minimum_fluid_signed_distance: float = 0.0,
) -> DFGCircleTrace:
    """Eagerly reconstruct pure-fluid-side traces on the nominal DFG circle.

    Pressure and native staggered velocities are sampled on rays pointing from
    the solid into the fluid. Every bilinear interpolation node, plus every
    adjacent cell centre used to construct a sampled staggered ``alpha_face``,
    must lie at least ``minimum_fluid_signed_distance`` outside the nominal
    circle. Pressure, tangential velocity, and normal velocity are then
    independently extrapolated to the circle. No sharp no-slip value is
    imposed on the finite Brinkman solution. The routine performs eager
    validation and is intended for post-processing rather than use under
    :func:`jax.jit`.
    """

    if (
        radius <= 0.0
        or h_reference <= 0.0
        or minimum_fluid_signed_distance < 0.0
    ):
        raise ValueError(
            "radius and h_reference must be positive and fluid distance nonnegative"
        )
    expected_shapes = {
        "u": (grid.nx + 1, grid.ny),
        "v": (grid.nx, grid.ny + 1),
        "p": (grid.nx, grid.ny),
    }
    for name, expected_shape in expected_shapes.items():
        if getattr(state, name).shape != expected_shape:
            raise ValueError(f"state.{name} must have shape {expected_shape}")
    if quadrature_points < 8 or quadrature_points % 2:
        raise ValueError("quadrature_points must be an even integer of at least 8")
    offsets_over_h = jnp.asarray(offsets_over_h, dtype=state.p.dtype)
    if offsets_over_h.ndim != 1 or offsets_over_h.size < 2:
        raise ValueError("offsets_over_h must be a one-dimensional sample array")
    if bool(jnp.any(offsets_over_h <= 0.0)) or bool(
        jnp.any(offsets_over_h[:-1] >= offsets_over_h[1:])
    ):
        raise ValueError("offsets_over_h must be positive and strictly increasing")
    if pressure_degree < 1 or pressure_degree >= offsets_over_h.size - 1:
        raise ValueError("pressure_degree must leave at least one fit residual DOF")
    if (
        tangential_velocity_degree < 1
        or tangential_velocity_degree >= offsets_over_h.size - 1
    ):
        raise ValueError(
            "tangential_velocity_degree must leave at least one fit residual DOF"
        )

    theta = (
        2.0
        * jnp.pi
        * jnp.arange(quadrature_points, dtype=state.p.dtype)
        / quadrature_points
    )
    normal_x = jnp.cos(theta)
    normal_y = jnp.sin(theta)
    offsets = offsets_over_h * h_reference
    outer_radius = radius + float(offsets[-1])
    if (
        centre_x - outer_radius < 0.0
        or centre_x + outer_radius > grid.lx
        or centre_y - outer_radius < 0.0
        or centre_y + outer_radius > grid.ly
    ):
        raise ValueError("radial trace samples leave the rectangular domain")
    sample_radius = radius + offsets[None, :]
    sample_x = centre_x + normal_x[:, None] * sample_radius
    sample_y = centre_y + normal_y[:, None] * sample_radius

    lattice_definitions = (
        (state.p.shape, 0.5 * grid.dx, 0.5 * grid.dy),
        (state.u.shape, 0.0, 0.5 * grid.dy),
        (state.v.shape, 0.5 * grid.dx, 0.0),
    )
    stencil_distances = [
        _minimum_circle_stencil_signed_distance(
            sample_x,
            sample_y,
            field_shape=shape,
            x_origin=x_origin,
            y_origin=y_origin,
            dx=grid.dx,
            dy=grid.dy,
            centre_x=centre_x,
            centre_y=centre_y,
            radius=radius,
        )
        for shape, x_origin, y_origin in lattice_definitions
    ]
    minimum_interpolation_node_signed_distance = jnp.min(
        jnp.stack(stencil_distances)
    )
    alpha_support_distances = [
        _minimum_velocity_alpha_support_signed_distance(
            sample_x,
            sample_y,
            component=component,
            grid=grid,
            centre_x=centre_x,
            centre_y=centre_y,
            radius=radius,
        )
        for component in ("u", "v")
    ]
    minimum_velocity_alpha_support_signed_distance = jnp.min(
        jnp.stack(alpha_support_distances)
    )
    minimum_stencil_signed_distance = jnp.minimum(
        minimum_interpolation_node_signed_distance,
        minimum_velocity_alpha_support_signed_distance,
    )
    stencil_tolerance = 64.0 * jnp.finfo(state.p.dtype).eps * max(
        grid.lx, grid.ly, radius
    )
    if float(minimum_stencil_signed_distance) + stencil_tolerance < float(
        minimum_fluid_signed_distance
    ):
        raise ValueError(
            "a radial trace interpolation or staggered-alpha dependency stencil "
            "enters the excluded solid or diffuse-interface region"
        )

    pressure_samples = _bilinear_uniform_value(
        state.p,
        sample_x,
        sample_y,
        x_origin=0.5 * grid.dx,
        y_origin=0.5 * grid.dy,
        dx=grid.dx,
        dy=grid.dy,
    )
    u_samples = _bilinear_uniform_value(
        state.u,
        sample_x,
        sample_y,
        x_origin=0.0,
        y_origin=0.5 * grid.dy,
        dx=grid.dx,
        dy=grid.dy,
    )
    v_samples = _bilinear_uniform_value(
        state.v,
        sample_x,
        sample_y,
        x_origin=0.5 * grid.dx,
        y_origin=0.0,
        dx=grid.dx,
        dy=grid.dy,
    )
    tangential_samples = (
        u_samples * normal_y[:, None] - v_samples * normal_x[:, None]
    )
    normal_samples = u_samples * normal_x[:, None] + v_samples * normal_y[:, None]
    (
        pressure,
        _,
        pressure_fit_rms,
        pressure_fit_condition_number,
        pressure_value_weight_l1,
        _,
    ) = _polynomial_trace(
        offsets, pressure_samples, pressure_degree
    )
    (
        tangential_surface_velocity,
        tangential_normal_derivative,
        tangential_fit_rms,
        velocity_fit_condition_number,
        velocity_value_weight_l1,
        velocity_derivative_weight_l1_normalized,
    ) = _polynomial_trace(
        offsets, tangential_samples, tangential_velocity_degree
    )
    (
        normal_surface_velocity,
        normal_normal_derivative,
        normal_fit_rms,
        _,
        _,
        _,
    ) = _polynomial_trace(
        offsets, normal_samples, tangential_velocity_degree
    )
    return DFGCircleTrace(
        theta,
        pressure,
        tangential_normal_derivative,
        normal_normal_derivative,
        tangential_surface_velocity,
        normal_surface_velocity,
        pressure_fit_rms,
        tangential_fit_rms,
        normal_fit_rms,
        pressure_fit_condition_number,
        velocity_fit_condition_number,
        pressure_value_weight_l1,
        velocity_value_weight_l1,
        velocity_derivative_weight_l1_normalized,
        minimum_stencil_signed_distance,
        minimum_interpolation_node_signed_distance,
        minimum_velocity_alpha_support_signed_distance,
    )


def integrate_dfg_circle_traction(
    pressure: Array,
    tangential_normal_derivative: Array,
    theta: Array,
    viscosity: float,
    radius: float,
    *,
    normal_normal_derivative: Array | None = None,
) -> DFGCircleTraction:
    """Integrate ``(nu grad(u) - p I) n`` on the DFG circle.

    The normal points out of the cylinder and into the fluid, and the tangent
    is ``(n_y, -n_x)``, matching Schäfer--Turek/FEATFLOW. If
    ``normal_normal_derivative`` is omitted, the published sharp-no-slip
    reduced formula is evaluated. Passing it evaluates the full traction of a
    finite-slip numerical field. Samples must be uniformly spaced over
    ``[0, 2*pi)``; the periodic trapezoidal rule then has constant arc weight.
    """

    pressure = jnp.asarray(pressure)
    tangential_normal_derivative = jnp.asarray(tangential_normal_derivative)
    theta = jnp.asarray(theta)
    if (
        pressure.ndim != 1
        or pressure.shape != tangential_normal_derivative.shape
        or pressure.shape != theta.shape
        or pressure.size < 8
    ):
        raise ValueError("pressure, derivative, and theta must be equal 1-D arrays")
    if viscosity < 0.0 or radius <= 0.0:
        raise ValueError("viscosity must be nonnegative and radius positive")
    expected_theta = (
        2.0
        * jnp.pi
        * jnp.arange(theta.size, dtype=theta.dtype)
        / theta.size
    )
    if not bool(jnp.all(jnp.isfinite(theta))) or not bool(
        jnp.allclose(theta, expected_theta, rtol=1.0e-12, atol=1.0e-14)
    ):
        raise ValueError("theta must be finite and uniformly cover [0, 2*pi)")
    if normal_normal_derivative is None:
        normal_normal_derivative = jnp.zeros_like(tangential_normal_derivative)
    else:
        normal_normal_derivative = jnp.asarray(normal_normal_derivative)
        if normal_normal_derivative.shape != pressure.shape:
            raise ValueError("normal derivative must have the pressure shape")
    normal_x = jnp.cos(theta)
    normal_y = jnp.sin(theta)
    pressure_drag_density = -pressure * normal_x
    pressure_lift_density = -pressure * normal_y
    viscous_drag_density = viscosity * (
        tangential_normal_derivative * normal_y
        + normal_normal_derivative * normal_x
    )
    viscous_lift_density = viscosity * (
        -tangential_normal_derivative * normal_x
        + normal_normal_derivative * normal_y
    )
    drag_density = pressure_drag_density + viscous_drag_density
    lift_density = pressure_lift_density + viscous_lift_density
    arc_weight = 2.0 * jnp.pi * radius / pressure.size
    pressure_force_x = arc_weight * jnp.sum(pressure_drag_density)
    pressure_force_y = arc_weight * jnp.sum(pressure_lift_density)
    viscous_force_x = arc_weight * jnp.sum(viscous_drag_density)
    viscous_force_y = arc_weight * jnp.sum(viscous_lift_density)
    return DFGCircleTraction(
        pressure_force_x + viscous_force_x,
        pressure_force_y + viscous_force_y,
        pressure_force_x,
        pressure_force_y,
        viscous_force_x,
        viscous_force_y,
        drag_density,
        lift_density,
    )


def brinkman_force(
    state: ChannelState, alpha: Array, grid: ChannelGrid2D
) -> tuple[Array, Array]:
    """Integrate the stationary Brinkman reaction exerted on the solid.

    The fluid momentum equations contain ``-alpha * velocity``. Therefore the
    equal-and-opposite force on the immersed stationary body is the volume
    integral of ``alpha * velocity`` on the native staggered control volumes.
    Boundary half-control-volume weights are included explicitly.
    """

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    u_width = jnp.ones(grid.nx + 1, dtype=state.u.dtype)
    u_width = u_width.at[0].set(0.5).at[-1].set(0.5)
    v_height = jnp.ones(grid.ny + 1, dtype=state.v.dtype)
    v_height = v_height.at[0].set(0.5).at[-1].set(0.5)
    force_x = (
        jnp.sum(alpha_u * state.u * u_width[:, None]) * grid.dx * grid.dy
    )
    force_y = (
        jnp.sum(alpha_v * state.v * v_height[None, :]) * grid.dx * grid.dy
    )
    return force_x, force_y


def mac_staggered_control_volume_momentum_balance(
    state: ChannelState,
    alpha: Array,
    inlet_u: Array,
    grid: ChannelGrid2D,
    params: ChannelParams,
    control_volume: MACStaggeredControlVolume,
) -> MACControlVolumeMomentumBalance:
    """Return an exact steady discrete momentum balance around an interior box.

    The conservative momentum tensor represented by the channel kernel is
    ``u tensor u + p I - nu grad(u)``.  This routine sums its native face
    fluxes on the two staggered rectangles and infers the stationary-body force
    as source minus outward momentum flux.  It deliberately does not
    interpolate a circular surface or label the result as sharp traction.

    For each component the implementation satisfies, to roundoff,

    ``F_cv = F_alpha,captured - integral(R_momentum)``.

    Consequently ``F_cv`` equals :func:`brinkman_force` only when the box
    captures every nonzero face-centred Brinkman coefficient and the steady
    momentum residual is zero.  Boundary-adjacent momentum equations have
    special closures, so this minimal auditable form accepts interior boxes
    only.  The u and v bounds are inclusive native-array indices.
    """

    if state.u.shape != (grid.nx + 1, grid.ny):
        raise ValueError("state.u has the wrong shape for grid")
    if state.v.shape != (grid.nx, grid.ny + 1):
        raise ValueError("state.v has the wrong shape for grid")
    if state.p.shape != (grid.nx, grid.ny):
        raise ValueError("state.p has the wrong shape for grid")
    if inlet_u.shape != (grid.ny,):
        raise ValueError("inlet_u has the wrong shape for grid")
    if params.viscosity < 0.0:
        raise ValueError("viscosity must be nonnegative")

    ui0, ui1 = control_volume.u_i_start, control_volume.u_i_stop
    uj0, uj1 = control_volume.u_j_start, control_volume.u_j_stop
    vi0, vi1 = control_volume.v_i_start, control_volume.v_i_stop
    vj0, vj1 = control_volume.v_j_start, control_volume.v_j_stop
    if not (1 <= ui0 <= ui1 <= grid.nx - 1):
        raise ValueError("u i bounds must satisfy 1 <= start <= stop <= nx - 1")
    if not (1 <= uj0 <= uj1 <= grid.ny - 2):
        raise ValueError("u j bounds must satisfy 1 <= start <= stop <= ny - 2")
    if not (1 <= vi0 <= vi1 <= grid.nx - 2):
        raise ValueError("v i bounds must satisfy 1 <= start <= stop <= nx - 2")
    if not (1 <= vj0 <= vj1 <= grid.ny - 1):
        raise ValueError("v j bounds must satisfy 1 <= start <= stop <= ny - 1")

    dtype_u = state.u.dtype
    dtype_v = state.v.dtype
    zero_u = jnp.asarray(0.0, dtype=dtype_u)
    zero_v = jnp.asarray(0.0, dtype=dtype_v)
    if params.include_advection:
        fluxes = conservative_momentum_fluxes(
            state.u, state.v, grid, scheme=params.advection_scheme
        )
        outward_advection_x = (
            grid.dy
            * jnp.sum(
                fluxes.uu[ui1, uj0 : uj1 + 1]
                - fluxes.uu[ui0 - 1, uj0 : uj1 + 1]
            )
            + grid.dx
            * jnp.sum(
                fluxes.uv_u[ui0 : ui1 + 1, uj1 + 1]
                - fluxes.uv_u[ui0 : ui1 + 1, uj0]
            )
        )
        outward_advection_y = (
            grid.dy
            * jnp.sum(
                fluxes.uv_v[vi1 + 1, vj0 : vj1 + 1]
                - fluxes.uv_v[vi0, vj0 : vj1 + 1]
            )
            + grid.dx
            * jnp.sum(
                fluxes.vv[vi0 : vi1 + 1, vj1]
                - fluxes.vv[vi0 : vi1 + 1, vj0 - 1]
            )
        )
    else:
        outward_advection_x = zero_u
        outward_advection_y = zero_v

    outward_pressure_x = grid.dy * jnp.sum(
        state.p[ui1, uj0 : uj1 + 1]
        - state.p[ui0 - 1, uj0 : uj1 + 1]
    )
    outward_pressure_y = grid.dx * jnp.sum(
        state.p[vi0 : vi1 + 1, vj1]
        - state.p[vi0 : vi1 + 1, vj0 - 1]
    )

    outward_velocity_gradient_x = (
        grid.dy
        * jnp.sum(
            (state.u[ui1 + 1, uj0 : uj1 + 1]
             - state.u[ui1, uj0 : uj1 + 1])
            / grid.dx
            - (state.u[ui0, uj0 : uj1 + 1]
               - state.u[ui0 - 1, uj0 : uj1 + 1])
            / grid.dx
        )
        + grid.dx
        * jnp.sum(
            (state.u[ui0 : ui1 + 1, uj1 + 1]
             - state.u[ui0 : ui1 + 1, uj1])
            / grid.dy
            - (state.u[ui0 : ui1 + 1, uj0]
               - state.u[ui0 : ui1 + 1, uj0 - 1])
            / grid.dy
        )
    )
    outward_velocity_gradient_y = (
        grid.dy
        * jnp.sum(
            (state.v[vi1 + 1, vj0 : vj1 + 1]
             - state.v[vi1, vj0 : vj1 + 1])
            / grid.dx
            - (state.v[vi0, vj0 : vj1 + 1]
               - state.v[vi0 - 1, vj0 : vj1 + 1])
            / grid.dx
        )
        + grid.dx
        * jnp.sum(
            (state.v[vi0 : vi1 + 1, vj1 + 1]
             - state.v[vi0 : vi1 + 1, vj1])
            / grid.dy
            - (state.v[vi0 : vi1 + 1, vj0]
               - state.v[vi0 : vi1 + 1, vj0 - 1])
            / grid.dy
        )
    )

    convective_x = -outward_advection_x
    convective_y = -outward_advection_y
    pressure_x = -outward_pressure_x
    pressure_y = -outward_pressure_y
    viscous_x = params.viscosity * outward_velocity_gradient_x
    viscous_y = params.viscosity * outward_velocity_gradient_y
    cell_area = grid.dx * grid.dy
    source_x = (
        jnp.asarray(params.force_x, dtype=dtype_u)
        * (ui1 - ui0 + 1)
        * (uj1 - uj0 + 1)
        * cell_area
    )
    source_y = (
        jnp.asarray(params.force_y, dtype=dtype_v)
        * (vi1 - vi0 + 1)
        * (vj1 - vj0 + 1)
        * cell_area
    )
    force_x = source_x + convective_x + pressure_x + viscous_x
    force_y = source_y + convective_y + pressure_y + viscous_y

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    captured_x = (
        jnp.sum(
            alpha_u[ui0 : ui1 + 1, uj0 : uj1 + 1]
            * state.u[ui0 : ui1 + 1, uj0 : uj1 + 1]
        )
        * cell_area
    )
    captured_y = (
        jnp.sum(
            alpha_v[vi0 : vi1 + 1, vj0 : vj1 + 1]
            * state.v[vi0 : vi1 + 1, vj0 : vj1 + 1]
        )
        * cell_area
    )
    residual = steady_residual(state, alpha, inlet_u, grid, params)
    residual_x = (
        jnp.sum(residual.momentum_u[ui0 : ui1 + 1, uj0 : uj1 + 1])
        * cell_area
    )
    residual_y = (
        jnp.sum(residual.momentum_v[vi0 : vi1 + 1, vj0 : vj1 + 1])
        * cell_area
    )
    global_x, global_y = brinkman_force(state, alpha, grid)
    closure_x = force_x - captured_x + residual_x
    closure_y = force_y - captured_y + residual_y
    return MACControlVolumeMomentumBalance(
        inferred_brinkman_reaction_x=force_x,
        inferred_brinkman_reaction_y=force_y,
        convective_contribution_x=convective_x,
        convective_contribution_y=convective_y,
        pressure_contribution_x=pressure_x,
        pressure_contribution_y=pressure_y,
        viscous_contribution_x=viscous_x,
        viscous_contribution_y=viscous_y,
        source_contribution_x=source_x,
        source_contribution_y=source_y,
        captured_brinkman_force_x=captured_x,
        captured_brinkman_force_y=captured_y,
        global_brinkman_force_x=global_x,
        global_brinkman_force_y=global_y,
        uncaptured_brinkman_force_x=global_x - captured_x,
        uncaptured_brinkman_force_y=global_y - captured_y,
        momentum_residual_integral_x=residual_x,
        momentum_residual_integral_y=residual_y,
        algebraic_closure_x=closure_x,
        algebraic_closure_y=closure_y,
    )


def cell_center_velocity(
    state: ChannelState,
) -> tuple[Array, Array]:
    """Average staggered face velocities to pressure-cell centres."""

    return (
        0.5 * (state.u[:-1, :] + state.u[1:, :]),
        0.5 * (state.v[:, :-1] + state.v[:, 1:]),
    )
