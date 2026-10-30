#!/usr/bin/env python3
"""Calibrate planar Brinkman-mask displacement without a CFD reference value.

The primary result is the compact-cosine, cell-centred, tangential-shear root.
The normal-face average and curved/two-component profiles are reported only as
implementation sensitivities.  They are not folded into the primary root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_cad.brinkman_mask_calibration import (  # noqa: E402
    calibrate_planar_mask_shift,
    mac_face_mask_profile,
    sharp_step_continuous_displacement,
    solve_planar_brinkman_poiseuille,
    solve_planar_brinkman_shear,
)


SCHEMA_VERSION = "brinkman-mask-calibration-1.0"


def _profile_metrics(profile) -> dict[str, float]:
    return {
        "max_abs_alpha_u_minus_cell": float(
            np.max(np.abs(profile.alpha_u_fraction - profile.cell_alpha_fraction))
        ),
        "max_abs_alpha_v_minus_cell": float(
            np.max(np.abs(profile.alpha_v_fraction - profile.cell_alpha_fraction))
        ),
        "max_abs_tangential_alpha_minus_cell": float(
            np.max(
                np.abs(
                    profile.tangential_alpha_fraction
                    - profile.cell_alpha_fraction
                )
            )
        ),
        "max_abs_normal_alpha_minus_cell": float(
            np.max(
                np.abs(profile.normal_alpha_fraction - profile.cell_alpha_fraction)
            )
        ),
        "max_abs_tangent_normal_coupling": float(
            np.max(np.abs(profile.tangent_normal_coupling_fraction))
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records = []
    for delta_ratio in args.delta_b_over_h:
        primary = calibrate_planar_mask_shift(
            delta_ratio,
            epsilon_over_h=args.epsilon_over_h,
            solid_cells=args.solid_cells,
            fluid_cells=args.fluid_cells,
            far_field_fit_cells=args.far_field_fit_cells,
            sampling="tangential-cell",
        )
        normal_sensitivity = calibrate_planar_mask_shift(
            delta_ratio,
            epsilon_over_h=args.epsilon_over_h,
            solid_cells=args.solid_cells,
            fluid_cells=args.fluid_cells,
            far_field_fit_cells=args.far_field_fit_cells,
            sampling="normal-face-average-sensitivity",
        )
        alpha_max = 1.0 / (delta_ratio * delta_ratio)
        normal_at_primary = solve_planar_brinkman_shear(
            alpha_max=alpha_max,
            viscosity=1.0,
            grid_spacing=1.0,
            epsilon_over_h=args.epsilon_over_h,
            mask_shift_over_h=primary.shift_over_h,
            solid_cells=args.solid_cells,
            fluid_cells=args.fluid_cells,
            far_field_fit_cells=args.far_field_fit_cells,
            sampling="normal-face-average-sensitivity",
        )
        poiseuille = []
        for half_width in args.poiseuille_half_width_over_h:
            solution = solve_planar_brinkman_poiseuille(
                alpha_max=alpha_max,
                viscosity=1.0,
                grid_spacing=1.0,
                epsilon_over_h=args.epsilon_over_h,
                mask_shift_over_h=primary.shift_over_h,
                nominal_half_width_over_h=half_width,
                exterior_solid_cells=args.solid_cells,
            )
            poiseuille.append(
                {
                    "nominal_half_width_over_h": half_width,
                    "displacement_over_h": solution.displacement_length,
                    "quadratic_coefficient": solution.quadratic_coefficient,
                    "expected_quadratic_coefficient": -0.5,
                    "linear_coefficient": solution.linear_coefficient,
                    "fit_rms": solution.fit_rms,
                    "residual_linf": solution.residual_linf,
                }
            )
        records.append(
            {
                "delta_b_over_h": delta_ratio,
                "alpha_max_h_squared_over_nu": alpha_max,
                "sharp_continuous_unshifted_displacement_over_h": (
                    sharp_step_continuous_displacement(delta_ratio)
                ),
                "sharp_continuous_optimal_shift_over_h": delta_ratio,
                "primary_tangential_cell": {
                    "unshifted_displacement_over_h": (
                        primary.unshifted_displacement_over_h
                    ),
                    "calibrated_shift_over_h": primary.shift_over_h,
                    "root_displacement_over_h": primary.displacement_over_h,
                    "iterations": primary.iterations,
                },
                "normal_face_average_sensitivity": {
                    "calibrated_shift_over_h": normal_sensitivity.shift_over_h,
                    "root_displacement_over_h": (
                        normal_sensitivity.displacement_over_h
                    ),
                    "displacement_at_primary_shift_over_h": (
                        normal_at_primary.displacement_length
                    ),
                    "interpretation": (
                        "normal MAC component sensitivity; not a correction "
                        "to the tangential shear calibration"
                    ),
                },
                "poiseuille_cross_check_at_primary_shift": poiseuille,
            }
        )

    profile_coordinate = np.linspace(
        -args.epsilon_over_h - 1.0,
        args.epsilon_over_h + 1.0,
        args.audit_profile_points,
    )
    orientation_rows = []
    for angle_degrees in args.audit_angle_degrees:
        angle_radians = np.deg2rad(angle_degrees)
        plane = mac_face_mask_profile(
            profile_coordinate,
            normal_angle_radians=angle_radians,
            epsilon_over_h=args.epsilon_over_h,
        )
        circle_rows = []
        for radius in args.audit_radius_over_h:
            circle = mac_face_mask_profile(
                profile_coordinate,
                normal_angle_radians=angle_radians,
                epsilon_over_h=args.epsilon_over_h,
                curvature_radius_over_h=radius,
            )
            circle_rows.append(
                {
                    "curvature_radius_over_h": radius,
                    **_profile_metrics(circle),
                }
            )
        orientation_rows.append(
            {
                "normal_angle_degrees": angle_degrees,
                "planar": _profile_metrics(plane),
                "circular": circle_rows,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "primary_problem": "one-dimensional steady tangential shear",
            "equation": "-nu*d2u/dn2 + alpha(n)*u = 0",
            "mask": "cell-centred compact-cosine solid fraction",
            "far_boundary": "unit shear",
            "displacement_definition": "far-fluid intercept divided by slope",
            "shift_sign": "positive moves penalised mask into nominal fluid",
            "reference_data": "none",
            "axis_aligned_mac_fact": (
                "the tangential component has no wall-normal face-average "
                "convolution; its coded average is along an invariant tangent"
            ),
            "discrete_domain": {
                "solid_cells": args.solid_cells,
                "fluid_cells": args.fluid_cells,
                "far_field_fit_cells": args.far_field_fit_cells,
                "cell_centre_phase_over_h": 0.5,
            },
            "root_solver": {
                "method": "bisection",
                "bracket_over_h": [-2.0, 3.0],
                "displacement_tolerance_over_h": 1.0e-10,
                "shift_tolerance_over_h": 1.0e-12,
                "max_iterations": 80,
            },
            "scope_limitation": (
                "the calibrated root is planar and axis-aligned; orientation, "
                "curvature, and the normal-component face average are reported "
                "as sensitivities and are not absorbed into the root"
            ),
        },
        "epsilon_over_h": args.epsilon_over_h,
        "records": records,
        "orientation_and_curvature_audit": {
            "status": "implementation sensitivity, not calibration retuning",
            "mask_shift_over_h": 0.0,
            "profile_point_count": args.audit_profile_points,
            "profile_coordinate_range_over_h": [
                float(profile_coordinate[0]),
                float(profile_coordinate[-1]),
            ],
            "rows": orientation_rows,
        },
    }


def _print_table(report: dict[str, Any]) -> None:
    print(
        "Reference-free compact-cosine planar tangential-shear calibration"
    )
    print(
        "positive shift => mask moves into fluid; normal-face results are "
        "sensitivity only"
    )
    print(
        "delta_B/h   ell(0)/h   calibrated s/h   ell(root)/h   "
        "normal-root s/h"
    )
    for record in report["records"]:
        primary = record["primary_tangential_cell"]
        normal = record["normal_face_average_sensitivity"]
        print(
            f"{record['delta_b_over_h']:9.4f} "
            f"{primary['unshifted_displacement_over_h']:10.6f} "
            f"{primary['calibrated_shift_over_h']:16.9f} "
            f"{primary['root_displacement_over_h']:13.3e} "
            f"{normal['calibrated_shift_over_h']:15.9f}"
        )
    print("Poiseuille residual displacement at the shear-calibrated shift:")
    for record in report["records"]:
        values = ", ".join(
            f"H/h={row['nominal_half_width_over_h']}: "
            f"ell/h={row['displacement_over_h']:.6g}"
            for row in record["poiseuille_cross_check_at_primary_shift"]
        )
        print(f"  delta_B/h={record['delta_b_over_h']}: {values}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-b-over-h",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0],
    )
    parser.add_argument("--epsilon-over-h", type=float, default=1.5)
    parser.add_argument("--solid-cells", type=int, default=64)
    parser.add_argument("--fluid-cells", type=int, default=64)
    parser.add_argument("--far-field-fit-cells", type=int, default=16)
    parser.add_argument(
        "--poiseuille-half-width-over-h",
        type=int,
        nargs="+",
        default=[24, 48],
    )
    parser.add_argument(
        "--audit-angle-degrees",
        type=float,
        nargs="+",
        default=[0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
    )
    parser.add_argument(
        "--audit-radius-over-h",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 40.0],
    )
    parser.add_argument("--audit-profile-points", type=int, default=401)
    parser.add_argument(
        "--json",
        action="store_true",
        help="write the complete JSON report to stdout instead of a table",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the complete JSON report to this path",
    )
    args = parser.parse_args(argv)
    if any(value <= 0.0 for value in args.delta_b_over_h):
        parser.error("all --delta-b-over-h values must be positive")
    if args.epsilon_over_h < 0.0:
        parser.error("--epsilon-over-h must be nonnegative")
    if not 2 <= args.far_field_fit_cells <= args.fluid_cells:
        parser.error(
            "--far-field-fit-cells must be between 2 and --fluid-cells"
        )
    if args.audit_profile_points < 3:
        parser.error("--audit-profile-points must be at least 3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    if args.json:
        sys.stdout.write(encoded)
    else:
        _print_table(report)
        if args.json_output is not None:
            print(f"JSON report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
