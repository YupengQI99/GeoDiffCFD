from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_cad.brinkman_mask_calibration import (
    calibrate_planar_mask_shift,
    compact_cosine_fluid_fraction,
    mac_face_mask_profile,
    planar_brinkman_alpha,
    sharp_step_continuous_displacement,
    solve_planar_brinkman_poiseuille,
    solve_planar_brinkman_shear,
)
from jax_cad.differentiable.mac_channel import (
    create_channel_grid,
    interpolate_alpha_to_faces,
)
from jax_cad.differentiable.smooth_heaviside import smooth_heaviside


ROOT = Path(__file__).resolve().parents[1]


def test_compact_cosine_mask_and_axis_tangent_have_no_normal_convolution() -> None:
    coordinate = np.array([-2.0, -1.5, -0.75, 0.0, 0.75, 1.5, 2.0])
    fluid = compact_cosine_fluid_fraction(coordinate, 1.5)
    assert fluid[0] == 0.0
    assert fluid[-1] == 1.0
    assert fluid[3] == 0.5
    np.testing.assert_allclose(fluid + fluid[::-1], 1.0, atol=2.0e-16)

    production_points = jnp.linspace(-2.0, 2.0, 257, dtype=jnp.float64)
    np.testing.assert_allclose(
        compact_cosine_fluid_fraction(np.asarray(production_points), 1.5),
        np.asarray(smooth_heaviside(production_points, 1.5)),
        rtol=0.0,
        atol=2.0e-16,
    )

    sampled_fluid, sampled_alpha = planar_brinkman_alpha(
        coordinate,
        alpha_max=7.0,
        epsilon=1.5,
        sampling="tangential-cell",
    )
    np.testing.assert_allclose(sampled_fluid, fluid, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sampled_alpha, 7.0 * (1.0 - fluid))

    # A horizontal wall has normal angle pi/2 and tangential velocity u.  The
    # current u-face average is along x, where a planar mask is invariant.
    profile = mac_face_mask_profile(
        np.linspace(-2.5, 2.5, 101),
        normal_angle_radians=0.5 * np.pi,
        epsilon_over_h=1.5,
    )
    np.testing.assert_allclose(
        profile.tangential_alpha_fraction,
        profile.cell_alpha_fraction,
        rtol=0.0,
        atol=3.0e-16,
    )
    # The normal v component does see the intended half-cell average.
    assert np.max(
        np.abs(profile.normal_alpha_fraction - profile.cell_alpha_fraction)
    ) > 0.05

    # Lock the statement to the production interpolation, not merely to the
    # one-dimensional audit implementation.
    grid = create_channel_grid(5, 8, 5.0, 8.0)
    cell_rows = 7.0 * (
        1.0
        - compact_cosine_fluid_fraction(
            np.arange(grid.ny, dtype=float) - 3.5, 1.5
        )
    )
    alpha_cell = jnp.asarray(np.tile(cell_rows, (grid.nx, 1)))
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha_cell, grid)
    np.testing.assert_allclose(
        np.asarray(alpha_u),
        np.tile(cell_rows, (grid.nx + 1, 1)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(alpha_v[:, 1:-1]),
        np.tile(0.5 * (cell_rows[:-1] + cell_rows[1:]), (grid.nx, 1)),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_penalty_reduces_to_the_exact_affine_shear_solution() -> None:
    solution = solve_planar_brinkman_shear(
        alpha_max=0.0,
        viscosity=2.0,
        grid_spacing=0.2,
        epsilon_over_h=1.5,
        solid_cells=20,
        fluid_cells=20,
    )
    solid_depth = 20 * 0.2
    np.testing.assert_allclose(
        solution.velocity,
        solution.normal_coordinate + solid_depth,
        rtol=0.0,
        atol=6.0e-14,
    )
    np.testing.assert_allclose(solution.far_field_slope, 1.0, atol=2.0e-14)
    np.testing.assert_allclose(
        solution.displacement_length, solid_depth, atol=6.0e-14
    )
    assert solution.residual_linf < 1.0e-12


def test_sharp_mask_converges_to_the_continuous_brinkman_constant() -> None:
    delta_b = 1.0
    solid_depth = 8.0
    exact = sharp_step_continuous_displacement(
        delta_b, solid_depth=solid_depth
    )
    errors = []
    for spacing in (0.5, 0.25, 0.125):
        cells = int(round(solid_depth / spacing))
        solution = solve_planar_brinkman_shear(
            alpha_max=1.0,
            viscosity=1.0,
            grid_spacing=spacing,
            epsilon_over_h=0.0,
            solid_cells=cells,
            fluid_cells=cells,
            far_field_fit_cells=min(16, cells),
        )
        errors.append(abs(solution.displacement_length - exact))
    assert errors[1] < errors[0] / 3.8
    assert errors[2] < errors[1] / 3.8
    np.testing.assert_allclose(
        sharp_step_continuous_displacement(delta_b), delta_b
    )
    np.testing.assert_allclose(
        sharp_step_continuous_displacement(delta_b, mask_shift=0.3), 0.7
    )


def test_compact_mask_shift_roots_cover_penalisation_layers() -> None:
    expected_roots = {
        0.5: 0.166968159,
        0.75: 0.500603961,
        1.0: 0.780990334,
    }
    for delta_ratio, expected_root in expected_roots.items():
        calibration = calibrate_planar_mask_shift(
            delta_ratio, epsilon_over_h=1.5
        )
        np.testing.assert_allclose(
            calibration.shift_over_h, expected_root, atol=5.0e-9
        )
        assert calibration.unshifted_displacement_over_h > 0.0
        assert abs(calibration.displacement_over_h) < 1.0e-9

        alpha_max = 1.0 / (delta_ratio * delta_ratio)
        shift_left = solve_planar_brinkman_shear(
            alpha_max=alpha_max,
            viscosity=1.0,
            grid_spacing=1.0,
            epsilon_over_h=1.5,
            mask_shift_over_h=calibration.shift_over_h - 0.05,
        )
        shift_right = solve_planar_brinkman_shear(
            alpha_max=alpha_max,
            viscosity=1.0,
            grid_spacing=1.0,
            epsilon_over_h=1.5,
            mask_shift_over_h=calibration.shift_over_h + 0.05,
        )
        assert shift_left.displacement_length > 0.0
        assert shift_right.displacement_length < 0.0


def test_calibration_uses_the_declared_far_field_fit_extent() -> None:
    calibration = calibrate_planar_mask_shift(
        1.0,
        epsilon_over_h=1.5,
        solid_cells=64,
        fluid_cells=64,
        far_field_fit_cells=12,
    )
    np.testing.assert_allclose(
        calibration.shift_over_h, 0.7809903339948505, atol=5.0e-12
    )
    with pytest.raises(ValueError, match="far_field_fit_cells"):
        calibrate_planar_mask_shift(
            1.0,
            epsilon_over_h=1.5,
            solid_cells=64,
            fluid_cells=64,
            far_field_fit_cells=65,
        )


def test_committed_calibration_report_has_matching_checksum_and_scope() -> None:
    report_path = (
        ROOT
        / "configs"
        / "forward_verification"
        / "brinkman_mask_calibration_eps1p5.json"
    )
    checksum_path = report_path.with_suffix(".sha256")
    expected_digest = checksum_path.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == expected_digest

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["method"]["reference_data"] == "none"
    assert report["method"]["discrete_domain"] == {
        "solid_cells": 64,
        "fluid_cells": 64,
        "far_field_fit_cells": 16,
        "cell_centre_phase_over_h": 0.5,
    }
    assert report["orientation_and_curvature_audit"]["mask_shift_over_h"] == 0.0
    assert [row["curvature_radius_over_h"] for row in report[
        "orientation_and_curvature_audit"
    ]["rows"][0]["circular"]] == [5.0, 10.0, 20.0, 40.0]
    records = {row["delta_b_over_h"]: row for row in report["records"]}
    assert records[1.0]["primary_tangential_cell"][
        "calibrated_shift_over_h"
    ] == pytest.approx(0.7809903339948505, abs=0.0)
    assert records[0.75]["primary_tangential_cell"][
        "calibrated_shift_over_h"
    ] == pytest.approx(0.5006039613799658, abs=0.0)


def test_poiseuille_cross_check_converges_after_shear_calibration() -> None:
    for delta_ratio in (0.5, 0.75, 1.0):
        calibration = calibrate_planar_mask_shift(
            delta_ratio, epsilon_over_h=1.5
        )
        solutions = [
            solve_planar_brinkman_poiseuille(
                alpha_max=1.0 / (delta_ratio * delta_ratio),
                viscosity=1.0,
                grid_spacing=1.0,
                epsilon_over_h=1.5,
                mask_shift_over_h=calibration.shift_over_h,
                nominal_half_width_over_h=half_width,
            )
            for half_width in (24, 48)
        ]
        for solution in solutions:
            np.testing.assert_allclose(
                solution.quadratic_coefficient, -0.5, atol=1.0e-12
            )
            assert abs(solution.linear_coefficient) < 1.0e-11
            assert solution.fit_rms < 1.0e-11
            assert solution.residual_linf < 2.0e-12
        # The remaining next-moment correction scales as inverse channel width;
        # it is not used to retune the leading shear displacement.
        ratio = (
            solutions[0].displacement_length
            / solutions[1].displacement_length
        )
        assert 1.99 < ratio < 2.01
        unshifted = solve_planar_brinkman_poiseuille(
            alpha_max=1.0 / (delta_ratio * delta_ratio),
            viscosity=1.0,
            grid_spacing=1.0,
            epsilon_over_h=1.5,
            mask_shift_over_h=0.0,
            nominal_half_width_over_h=48,
        )
        assert abs(solutions[1].displacement_length) < (
            0.05 * abs(unshifted.displacement_length)
        )


def test_curved_two_component_audit_exposes_orientation_sensitivity() -> None:
    coordinate = np.linspace(-2.5, 2.5, 201)
    angle = np.deg2rad(30.0)
    plane = mac_face_mask_profile(
        coordinate,
        normal_angle_radians=angle,
        epsilon_over_h=1.5,
    )
    circle_coarse = mac_face_mask_profile(
        coordinate,
        normal_angle_radians=angle,
        epsilon_over_h=1.5,
        curvature_radius_over_h=10.0,
    )
    circle_fine = mac_face_mask_profile(
        coordinate,
        normal_angle_radians=angle,
        epsilon_over_h=1.5,
        curvature_radius_over_h=40.0,
    )
    assert np.max(np.abs(plane.alpha_u_fraction - plane.alpha_v_fraction)) > 0.02
    assert np.max(np.abs(plane.tangent_normal_coupling_fraction)) > 0.01
    assert np.max(
        np.abs(
            circle_coarse.tangential_alpha_fraction
            - plane.tangential_alpha_fraction
        )
    ) > 1.0e-3
    # Increasing radius removes the curvature part while retaining the planar
    # orientation sensitivity of the two component-wise face averages.
    coarse_error = np.max(
        np.abs(
            circle_coarse.tangential_alpha_fraction
            - plane.tangential_alpha_fraction
        )
    )
    fine_error = np.max(
        np.abs(
            circle_fine.tangential_alpha_fraction
            - plane.tangential_alpha_fraction
        )
    )
    assert fine_error < coarse_error
