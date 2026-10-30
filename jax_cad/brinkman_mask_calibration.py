"""Reference-free calibration of a cell-centred Brinkman wall mask.

The primary problem is one-dimensional tangential shear next to a planar,
axis-aligned wall. It supports the legacy compact-cosine mask and the design
kernel's compact-quintic mask; cosine remains the bit-preserving default.
For such a wall the tangential MAC velocity sees the
cell-centred mask without a normal-direction face convolution: the arithmetic
average in :func:`interpolate_alpha_to_faces` is along the invariant tangent.

The sign convention used throughout is

* the nominal wall is at ``normal_coordinate = 0``;
* positive coordinate points from solid into fluid; and
* a positive mask shift moves the penalised region into the fluid.

The displacement length is the intercept of the far-fluid unit-shear line,
``u = normal_coordinate + displacement_length``.  A positive value therefore
places the extrapolated no-slip plane inside the nominal solid.  Calibration
finds the positive mask shift that makes this displacement zero.  No external
flow benchmark or force coefficient enters this calculation.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np


MaskSampling = Literal[
    "tangential-cell",
    "normal-face-average-sensitivity",
]
MaskProfile = Literal["compact-cosine", "compact-quintic"]


class PlanarShearSolution(NamedTuple):
    """Discrete half-space shear solution and its far-fluid linear fit."""

    normal_coordinate: np.ndarray
    fluid_fraction: np.ndarray
    alpha: np.ndarray
    velocity: np.ndarray
    far_field_slope: float
    far_field_intercept: float
    displacement_length: float
    residual_linf: float


class MaskShiftCalibration(NamedTuple):
    """Root of planar shear displacement versus mask shift."""

    delta_b_over_h: float
    epsilon_over_h: float
    mask_sampling: str
    unshifted_displacement_over_h: float
    shift_over_h: float
    displacement_over_h: float
    bracket_lower_over_h: float
    bracket_upper_over_h: float
    iterations: int


class PlanarPoiseuilleSolution(NamedTuple):
    """Symmetric penalised-channel solution and pure-fluid parabola fit."""

    normal_coordinate: np.ndarray
    fluid_fraction: np.ndarray
    alpha: np.ndarray
    velocity: np.ndarray
    quadratic_coefficient: float
    linear_coefficient: float
    constant_coefficient: float
    effective_half_width: float
    displacement_length: float
    fit_rms: float
    residual_linf: float


class MACFaceMaskProfile(NamedTuple):
    """Local two-component MAC mask audit along a plane or circular normal."""

    normal_coordinate_over_h: np.ndarray
    cell_alpha_fraction: np.ndarray
    alpha_u_fraction: np.ndarray
    alpha_v_fraction: np.ndarray
    tangential_alpha_fraction: np.ndarray
    normal_alpha_fraction: np.ndarray
    tangent_normal_coupling_fraction: np.ndarray


def compact_cosine_fluid_fraction(
    signed_distance: np.ndarray | float,
    epsilon: float,
) -> np.ndarray:
    """Return the exact compact-cosine fluid fraction used by the DFG mask.

    ``epsilon == 0`` selects a sharp Heaviside and is useful for the analytic
    convergence audit.  Production diffuse masks require positive epsilon.
    Values exactly on a sharp interface are assigned fluid fraction one half;
    no cell centre in the default interface phase lies exactly there.
    """

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    phi = np.asarray(signed_distance, dtype=float)
    if epsilon == 0.0:
        return np.where(phi < 0.0, 0.0, np.where(phi > 0.0, 1.0, 0.5))
    scaled = phi / epsilon
    interior = 0.5 * (
        1.0 + scaled + np.sin(np.pi * scaled) / np.pi
    )
    return np.where(
        phi < -epsilon,
        0.0,
        np.where(phi > epsilon, 1.0, interior),
    )


def compact_quintic_fluid_fraction(signed_distance, epsilon):
    """NumPy equivalent of the design kernel's C^2 compact Heaviside."""
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    phi = np.asarray(signed_distance, dtype=float)
    if epsilon == 0:
        return np.where(phi < 0, 0., np.where(phi > 0, 1., .5))
    s = np.clip(phi / epsilon, -1., 1.)
    return .5 + s * (15/16 + s*s * (-5/8 + s*s * 3/16))


def planar_brinkman_alpha(
    normal_coordinate: np.ndarray,
    *,
    alpha_max: float,
    epsilon: float,
    mask_shift: float = 0.0,
    grid_spacing: float = 1.0,
    sampling: MaskSampling = "tangential-cell",
    profile: MaskProfile = "compact-cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a planar wall mask at the velocity sample coordinates.

    ``tangential-cell`` is the calibration object.  For an axis-aligned wall,
    it exactly represents the tangential MAC coefficient because averaging
    adjacent cell values along the wall leaves the mask unchanged.

    ``normal-face-average-sensitivity`` instead averages cell-mask samples a
    half cell to either side in the wall-normal direction.  It represents the
    coefficient seen by the *normal* MAC component and is deliberately kept as
    an implementation/orientation sensitivity rather than a correction to the
    planar tangential theory.
    """

    if alpha_max < 0.0:
        raise ValueError("alpha_max must be nonnegative")
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive")
    coordinate = np.asarray(normal_coordinate, dtype=float)
    if profile not in ("compact-cosine", "compact-quintic"):
        raise ValueError(f"unknown mask profile: {profile}")
    fraction_function = (compact_cosine_fluid_fraction if profile == "compact-cosine"
                         else compact_quintic_fluid_fraction)

    def cell_values(location: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fraction = fraction_function(
            location - mask_shift, epsilon
        )
        return fraction, alpha_max * (1.0 - fraction)

    if sampling == "tangential-cell":
        return cell_values(coordinate)
    if sampling == "normal-face-average-sensitivity":
        fraction_minus, alpha_minus = cell_values(
            coordinate - 0.5 * grid_spacing
        )
        fraction_plus, alpha_plus = cell_values(
            coordinate + 0.5 * grid_spacing
        )
        return (
            0.5 * (fraction_minus + fraction_plus),
            0.5 * (alpha_minus + alpha_plus),
        )
    raise ValueError(f"unknown mask sampling: {sampling}")


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a real tridiagonal system with the Thomas algorithm."""

    diagonal_work = np.asarray(diagonal, dtype=float).copy()
    rhs_work = np.asarray(rhs, dtype=float).copy()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    size = diagonal_work.size
    if size < 2:
        raise ValueError("tridiagonal system must contain at least two rows")
    if lower.shape != (size - 1,) or upper.shape != (size - 1,):
        raise ValueError("off-diagonals have the wrong shape")
    if rhs_work.shape != (size,):
        raise ValueError("rhs has the wrong shape")

    for index in range(1, size):
        if diagonal_work[index - 1] == 0.0:
            raise np.linalg.LinAlgError("zero pivot in tridiagonal solve")
        multiplier = lower[index - 1] / diagonal_work[index - 1]
        diagonal_work[index] -= multiplier * upper[index - 1]
        rhs_work[index] -= multiplier * rhs_work[index - 1]

    solution = np.empty(size, dtype=float)
    if diagonal_work[-1] == 0.0:
        raise np.linalg.LinAlgError("zero pivot in tridiagonal solve")
    solution[-1] = rhs_work[-1] / diagonal_work[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = (
            rhs_work[index] - upper[index] * solution[index + 1]
        ) / diagonal_work[index]
    return solution


def _system_residual_linf(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    residual = diagonal * solution - rhs
    residual[1:] += lower * solution[:-1]
    residual[:-1] += upper * solution[1:]
    return float(np.max(np.abs(residual)))


def solve_planar_brinkman_shear(
    *,
    alpha_max: float,
    viscosity: float,
    grid_spacing: float,
    epsilon_over_h: float,
    mask_shift_over_h: float = 0.0,
    solid_cells: int = 64,
    fluid_cells: int = 64,
    far_field_fit_cells: int = 16,
    sampling: MaskSampling = "tangential-cell",
    profile: MaskProfile = "compact-cosine",
) -> PlanarShearSolution:
    """Solve ``-nu u'' + alpha u = 0`` with unit far-fluid shear.

    The left domain face lies deep in the solid and has ``u = 0``.  The right
    face has ``du/dn = 1``.  A cell-centred finite-volume Laplacian is used,
    including the half-cell Dirichlet distance at the left boundary.
    """

    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive")
    if epsilon_over_h < 0.0:
        raise ValueError("epsilon_over_h must be nonnegative")
    if solid_cells < 2 or fluid_cells < 2:
        raise ValueError("solid_cells and fluid_cells must both be at least 2")
    if far_field_fit_cells < 2 or far_field_fit_cells > fluid_cells:
        raise ValueError(
            "far_field_fit_cells must be between 2 and fluid_cells"
        )

    size = solid_cells + fluid_cells
    coordinate = (
        np.arange(size, dtype=float) - solid_cells + 0.5
    ) * grid_spacing
    fluid_fraction, alpha = planar_brinkman_alpha(
        coordinate,
        alpha_max=alpha_max,
        epsilon=epsilon_over_h * grid_spacing,
        mask_shift=mask_shift_over_h * grid_spacing,
        grid_spacing=grid_spacing,
        sampling=sampling,
        profile=profile,
    )

    diffusion = viscosity / (grid_spacing * grid_spacing)
    lower = np.full(size - 1, -diffusion, dtype=float)
    upper = np.full(size - 1, -diffusion, dtype=float)
    diagonal = 2.0 * diffusion + alpha
    # Cell-centred finite volume with u=0 at a boundary half a cell away.
    diagonal[0] = 3.0 * diffusion + alpha[0]
    # Unit Neumann flux at the right domain face.
    diagonal[-1] = diffusion + alpha[-1]
    rhs = np.zeros(size, dtype=float)
    rhs[-1] = viscosity / grid_spacing

    velocity = _solve_tridiagonal(lower, diagonal, upper, rhs)
    fit_indices = np.arange(size - far_field_fit_cells, size)
    if np.any(alpha[fit_indices] > max(alpha_max, 1.0) * 1.0e-13):
        raise ValueError(
            "far-field fit intersects the diffuse/penalised mask; increase "
            "fluid_cells or decrease far_field_fit_cells"
        )
    design = np.column_stack(
        (coordinate[fit_indices], np.ones(far_field_fit_cells))
    )
    slope, intercept = np.linalg.lstsq(
        design, velocity[fit_indices], rcond=None
    )[0]
    if slope <= 0.0:
        raise RuntimeError("far-field shear slope must be positive")
    displacement = intercept / slope
    return PlanarShearSolution(
        coordinate,
        fluid_fraction,
        alpha,
        velocity,
        float(slope),
        float(intercept),
        float(displacement),
        _system_residual_linf(
            lower, diagonal, upper, velocity, rhs
        ),
    )


def calibrate_planar_mask_shift(
    delta_b_over_h: float,
    *,
    epsilon_over_h: float,
    bracket_over_h: tuple[float, float] = (-2.0, 3.0),
    displacement_tolerance_over_h: float = 1.0e-10,
    shift_tolerance_over_h: float = 1.0e-12,
    max_iterations: int = 80,
    solid_cells: int = 64,
    fluid_cells: int = 64,
    far_field_fit_cells: int = 16,
    sampling: MaskSampling = "tangential-cell",
    profile: MaskProfile = "compact-cosine",
) -> MaskShiftCalibration:
    """Calibrate mask shift by bisection at fixed ``delta_B / h``.

    Here ``delta_B = sqrt(viscosity / alpha_max)``.  The dimensionless solve
    sets ``h = viscosity = 1`` and therefore contains no case-specific length,
    velocity, force, or benchmark observable.
    """

    if delta_b_over_h <= 0.0:
        raise ValueError("delta_b_over_h must be positive")
    if displacement_tolerance_over_h <= 0.0:
        raise ValueError("displacement tolerance must be positive")
    if shift_tolerance_over_h <= 0.0:
        raise ValueError("shift tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    lower_shift, upper_shift = map(float, bracket_over_h)
    if lower_shift >= upper_shift:
        raise ValueError("calibration bracket must be strictly increasing")
    alpha_max = 1.0 / (delta_b_over_h * delta_b_over_h)

    def displacement(shift: float) -> float:
        return solve_planar_brinkman_shear(
            alpha_max=alpha_max,
            viscosity=1.0,
            grid_spacing=1.0,
            epsilon_over_h=epsilon_over_h,
            mask_shift_over_h=shift,
            solid_cells=solid_cells,
            fluid_cells=fluid_cells,
            far_field_fit_cells=far_field_fit_cells,
            sampling=sampling,
            profile=profile,
        ).displacement_length

    unshifted = displacement(0.0)
    lower_value = displacement(lower_shift)
    upper_value = displacement(upper_shift)
    if lower_value == 0.0:
        upper_shift = lower_shift
        upper_value = lower_value
    elif upper_value == 0.0:
        lower_shift = upper_shift
        lower_value = upper_value
    elif lower_value * upper_value > 0.0:
        raise ValueError(
            "calibration bracket does not straddle zero displacement: "
            f"ell({lower_shift})={lower_value}, "
            f"ell({upper_shift})={upper_value}"
        )

    iteration = 0
    midpoint = 0.5 * (lower_shift + upper_shift)
    midpoint_value = displacement(midpoint)
    while iteration < max_iterations:
        iteration += 1
        midpoint = 0.5 * (lower_shift + upper_shift)
        midpoint_value = displacement(midpoint)
        if (
            abs(midpoint_value) <= displacement_tolerance_over_h
            or upper_shift - lower_shift <= shift_tolerance_over_h
        ):
            break
        if lower_value * midpoint_value <= 0.0:
            upper_shift = midpoint
            upper_value = midpoint_value
        else:
            lower_shift = midpoint
            lower_value = midpoint_value
    else:
        raise RuntimeError("mask-shift bisection did not converge")

    return MaskShiftCalibration(
        float(delta_b_over_h),
        float(epsilon_over_h),
        sampling,
        float(unshifted),
        float(midpoint),
        float(midpoint_value),
        float(lower_shift),
        float(upper_shift),
        iteration,
    )


def solve_planar_brinkman_poiseuille(
    *,
    alpha_max: float,
    viscosity: float,
    grid_spacing: float,
    epsilon_over_h: float,
    mask_shift_over_h: float = 0.0,
    nominal_half_width_over_h: int = 24,
    exterior_solid_cells: int = 32,
    body_force: float = 1.0,
) -> PlanarPoiseuilleSolution:
    """Solve a symmetric penalised Poiseuille channel as a cross-check.

    The calibrated planar-shear shift is expected to remove the leading wall
    displacement.  A residual of order ``1 / half_width`` remains because a
    parabolic profile also probes the next wall-response moment; this is why
    Poiseuille is reported as a cross-check rather than used to retune shift.
    """

    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be positive")
    if viscosity <= 0.0 or grid_spacing <= 0.0:
        raise ValueError("viscosity and grid_spacing must be positive")
    if epsilon_over_h < 0.0:
        raise ValueError("epsilon_over_h must be nonnegative")
    if nominal_half_width_over_h < 4:
        raise ValueError("nominal_half_width_over_h must be at least 4")
    if exterior_solid_cells < 2:
        raise ValueError("exterior_solid_cells must be at least 2")
    if body_force <= 0.0:
        raise ValueError("body_force must be positive")

    half_width = nominal_half_width_over_h * grid_spacing
    side_cells = nominal_half_width_over_h + exterior_solid_cells
    coordinate = (
        np.arange(2 * side_cells, dtype=float) - side_cells + 0.5
    ) * grid_spacing
    # Positive signed distance denotes the nominal fluid channel.  Subtracting
    # a positive shift moves both penalised interfaces into that channel.
    signed_distance = (
        half_width
        - np.abs(coordinate)
        - mask_shift_over_h * grid_spacing
    )
    fluid_fraction = compact_cosine_fluid_fraction(
        signed_distance, epsilon_over_h * grid_spacing
    )
    alpha = alpha_max * (1.0 - fluid_fraction)

    size = coordinate.size
    diffusion = viscosity / (grid_spacing * grid_spacing)
    lower = np.full(size - 1, -diffusion, dtype=float)
    upper = np.full(size - 1, -diffusion, dtype=float)
    diagonal = 2.0 * diffusion + alpha
    diagonal[0] = 3.0 * diffusion + alpha[0]
    diagonal[-1] = 3.0 * diffusion + alpha[-1]
    rhs = np.full(size, body_force, dtype=float)
    velocity = _solve_tridiagonal(lower, diagonal, upper, rhs)

    pure_fluid = alpha <= max(alpha_max, 1.0) * 1.0e-13
    if np.count_nonzero(pure_fluid) < 5:
        raise ValueError("not enough pure-fluid cells for a quadratic fit")
    # Scale the abscissa before least squares; otherwise the Vandermonde
    # conditioning needlessly grows in the wide-channel convergence audit.
    fit_scale = float(np.max(np.abs(coordinate[pure_fluid])))
    scaled_coordinate = coordinate[pure_fluid] / fit_scale
    design = np.column_stack(
        (
            scaled_coordinate**2,
            scaled_coordinate,
            np.ones(np.count_nonzero(pure_fluid)),
        )
    )
    scaled_quadratic, scaled_linear, constant = np.linalg.lstsq(
        design, velocity[pure_fluid], rcond=None
    )[0]
    quadratic = scaled_quadratic / (fit_scale * fit_scale)
    linear = scaled_linear / fit_scale
    fitted = design @ np.array(
        (scaled_quadratic, scaled_linear, constant)
    )
    fit_rms = float(
        np.sqrt(np.mean((velocity[pure_fluid] - fitted) ** 2))
    )
    if quadratic >= 0.0 or constant <= 0.0:
        raise RuntimeError("pure-fluid Poiseuille fit has invalid signs")
    # Symmetry should make the linear coefficient roundoff-small.  The positive
    # root of the even part is the effective channel half-width.
    effective_half_width = float(np.sqrt(-constant / quadratic))
    return PlanarPoiseuilleSolution(
        coordinate,
        fluid_fraction,
        alpha,
        velocity,
        float(quadratic),
        float(linear),
        float(constant),
        effective_half_width,
        effective_half_width - half_width,
        fit_rms,
        _system_residual_linf(
            lower, diagonal, upper, velocity, rhs
        ),
    )


def sharp_step_continuous_displacement(
    delta_b: float,
    *,
    mask_shift: float = 0.0,
    solid_depth: float = np.inf,
) -> float:
    """Analytic displacement for a sharp continuous Brinkman half-space.

    With an infinitely deep solid, ``ell = delta_B - mask_shift``.  For a
    finite solid terminated by a zero-velocity boundary, the exact constant is
    ``delta_B * tanh((solid_depth + mask_shift) / delta_B) - mask_shift``.
    """

    if delta_b <= 0.0:
        raise ValueError("delta_b must be positive")
    if solid_depth <= 0.0:
        raise ValueError("solid_depth must be positive")
    if np.isinf(solid_depth):
        return float(delta_b - mask_shift)
    if solid_depth + mask_shift <= 0.0:
        raise ValueError("shift places the interface beyond the solid boundary")
    return float(
        delta_b * np.tanh((solid_depth + mask_shift) / delta_b)
        - mask_shift
    )


def mac_face_mask_profile(
    normal_coordinate_over_h: np.ndarray,
    *,
    normal_angle_radians: float,
    epsilon_over_h: float,
    mask_shift_over_h: float = 0.0,
    curvature_radius_over_h: float | None = None,
) -> MACFaceMaskProfile:
    """Audit current MAC face averaging for a plane or a circular interface.

    The returned coefficients are fractions of ``alpha_max``.  For a plane,
    adjacent cell signed distances are evaluated exactly from the local normal.
    For a circle, the two cells adjacent to each Cartesian face are evaluated
    using the exact radial signed distance.  The latter exposes both curvature
    and the difference between the u/v component masks without pretending that
    a one-dimensional profile is a complete curved-body flow solve.
    """

    if epsilon_over_h < 0.0:
        raise ValueError("epsilon_over_h must be nonnegative")
    if curvature_radius_over_h is not None and curvature_radius_over_h <= 0.5:
        raise ValueError("curvature_radius_over_h must exceed half a cell")
    q = np.asarray(normal_coordinate_over_h, dtype=float)
    if q.ndim != 1:
        raise ValueError("normal_coordinate_over_h must be one-dimensional")
    normal_x = float(np.cos(normal_angle_radians))
    normal_y = float(np.sin(normal_angle_radians))

    def solid_fraction(phi: np.ndarray) -> np.ndarray:
        return 1.0 - compact_cosine_fluid_fraction(
            phi - mask_shift_over_h, epsilon_over_h
        )

    direct = solid_fraction(q)
    if curvature_radius_over_h is None:
        # A u face averages the cells half a cell away in Cartesian x; a v
        # face does the same in y.  Project those offsets onto the plane normal.
        phi_u_minus = q - 0.5 * normal_x
        phi_u_plus = q + 0.5 * normal_x
        phi_v_minus = q - 0.5 * normal_y
        phi_v_plus = q + 0.5 * normal_y
    else:
        radius = float(curvature_radius_over_h)
        centreline_x = (radius + q) * normal_x
        centreline_y = (radius + q) * normal_y
        phi_u_minus = np.sqrt(
            (centreline_x - 0.5) ** 2 + centreline_y**2
        ) - radius
        phi_u_plus = np.sqrt(
            (centreline_x + 0.5) ** 2 + centreline_y**2
        ) - radius
        phi_v_minus = np.sqrt(
            centreline_x**2 + (centreline_y - 0.5) ** 2
        ) - radius
        phi_v_plus = np.sqrt(
            centreline_x**2 + (centreline_y + 0.5) ** 2
        ) - radius

    alpha_u = 0.5 * (
        solid_fraction(phi_u_minus) + solid_fraction(phi_u_plus)
    )
    alpha_v = 0.5 * (
        solid_fraction(phi_v_minus) + solid_fraction(phi_v_plus)
    )
    # tangent = (-normal_y, normal_x).  Project diag(alpha_u, alpha_v)
    # into the local tangent/normal frame, retaining the off-diagonal audit.
    tangential = normal_y * normal_y * alpha_u + normal_x * normal_x * alpha_v
    normal = normal_x * normal_x * alpha_u + normal_y * normal_y * alpha_v
    coupling = normal_x * normal_y * (alpha_v - alpha_u)
    return MACFaceMaskProfile(
        q,
        direct,
        alpha_u,
        alpha_v,
        tangential,
        normal,
        coupling,
    )
