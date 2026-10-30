"""Auditable immersed-Brinkman run of the DFG cylinder benchmark 2D-1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.benchmarks import DFG_2D1
from jax_cad.brinkman_mask_calibration import calibrate_planar_mask_shift
from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    ChannelState,
    cell_centres,
    create_channel_grid,
    create_initial_state,
    flux_diagnostics,
    interpolate_alpha_to_faces,
    sample_native_face_coefficient,
    solve_steady,
    solve_steady_chunked,
)
from jax_cad.differentiable.smooth_heaviside import smooth_heaviside
from jax_cad.run_artifacts import (
    artifact_record,
    compile_and_sample_args,
    no_gradient_record,
    peak_memory_record,
    write_json,
)
from jax_cad.verification import (
    MACStaggeredControlVolume,
    brinkman_force,
    cell_center_velocity,
    dfg_circle_exterior_trace,
    integrate_dfg_circle_traction,
    mac_staggered_control_volume_momentum_balance,
    parabolic_inlet_face_averages,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.capture_compute_environment import capture


CONFIG_SCHEMA_1_1 = "forward-verification-config-1.1"
CONFIG_SCHEMA_1_2 = "forward-verification-config-1.2"
REFERENCE_FREE_CALIBRATION_METHOD = (
    "one-dimensional-steady-planar-tangential-shear-zero-displacement"
)
REFERENCE_FREE_CALIBRATION_TOOL = "tools/calibrate_brinkman_mask.py"
REFERENCE_FREE_CALIBRATION_SAMPLING = "tangential-cell"
PRIMARY_ACCEPTANCE_OBSERVABLES = {
    "drag_coefficient": "metrics.numerical.brinkman_drag_coefficient",
    "lift_coefficient": "metrics.numerical.brinkman_lift_coefficient",
    "pressure_difference": "metrics.numerical.pressure_difference",
    "normal_surface_slip_rms": (
        "metrics.surface_measurement.normal_surface_slip_rms"
    ),
    "tangential_surface_slip_rms": (
        "metrics.surface_measurement.tangential_surface_slip_rms"
    ),
}


def _exact_json_integer(value, name: str) -> int:
    """Reject booleans and fractional JSON numbers used as audit integers."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != round(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _validate_phase_locked_brinkman_config(
    config: dict,
    *,
    dx: float,
    dy: float,
    viscosity: float,
) -> None:
    """Validate the dimensionless declarations required by config schema 1.2."""

    if not np.isclose(dx, dy, rtol=0.0, atol=1.0e-14):
        raise ValueError("schema 1.2 requires a square dx=dy=h grid")
    h_reference = dx
    immersed = config["immersed_boundary"]
    required_scalars = (
        "delta_b_over_h",
        "alpha_max_h_squared_over_nu",
        "epsilon_over_h",
        "mask_midpoint_shift_over_h",
    )
    values = {}
    for name in required_scalars:
        if name not in immersed:
            raise ValueError(
                f"schema 1.2 requires immersed_boundary.{name}"
            )
        values[name] = float(immersed[name])
        if not np.isfinite(values[name]):
            raise ValueError(f"immersed_boundary.{name} must be finite")

    q = values["delta_b_over_h"]
    if q <= 0.0:
        raise ValueError("immersed_boundary.delta_b_over_h must be positive")
    if values["epsilon_over_h"] <= 0.0:
        raise ValueError("immersed_boundary.epsilon_over_h must be positive")
    alpha_max = float(immersed["alpha_max"])
    expected_alpha = viscosity / ((q * h_reference) ** 2)
    if not np.isclose(alpha_max, expected_alpha, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            "schema 1.2 requires alpha_max = nu / (delta_b_over_h * h)^2; "
            f"got {alpha_max}, expected {expected_alpha}"
        )
    actual_alpha_ratio = alpha_max * h_reference * h_reference / viscosity
    declared_alpha_ratio = values["alpha_max_h_squared_over_nu"]
    expected_alpha_ratio = 1.0 / (q * q)
    if not np.isclose(
        declared_alpha_ratio,
        actual_alpha_ratio,
        rtol=1.0e-12,
        atol=1.0e-14,
    ) or not np.isclose(
        declared_alpha_ratio,
        expected_alpha_ratio,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(
            "immersed_boundary.alpha_max_h_squared_over_nu must equal both "
            "alpha_max*h^2/nu and 1/delta_b_over_h^2"
        )

    calibration = immersed.get("reference_free_calibration", {})
    expected_calibration = {
        "method": REFERENCE_FREE_CALIBRATION_METHOD,
        "tool": REFERENCE_FREE_CALIBRATION_TOOL,
        "sampling": REFERENCE_FREE_CALIBRATION_SAMPLING,
        "reference_data": "none",
    }
    for name, expected in expected_calibration.items():
        if calibration.get(name) != expected:
            raise ValueError(
                "immersed_boundary.reference_free_calibration."
                f"{name} must be {expected!r}"
            )
    calibration_extents = {}
    for name in ("solid_cells", "fluid_cells"):
        calibration_extents[name] = _exact_json_integer(
            calibration.get(name, 0),
            f"reference_free_calibration.{name}",
        )
        if calibration_extents[name] < 32:
            raise ValueError(
                "reference-free calibration must record at least 32 "
                f"{name} of domain extent"
            )
    far_field_fit_cells = _exact_json_integer(
        calibration.get("far_field_fit_cells", 0),
        "reference_free_calibration.far_field_fit_cells",
    )
    if not 2 <= far_field_fit_cells <= calibration_extents["fluid_cells"]:
        raise ValueError(
            "reference-free calibration must record far_field_fit_cells "
            "between 2 and fluid_cells"
        )
    cell_centre_phase = float(
        calibration.get("cell_centre_phase_over_h", np.nan)
    )
    if not np.isclose(cell_centre_phase, 0.5, rtol=0.0, atol=0.0):
        raise ValueError(
            "reference-free calibration must record the production "
            "cell-centre phase 0.5h"
        )
    bracket = tuple(
        float(value) for value in calibration.get("bracket_over_h", ())
    )
    if len(bracket) != 2 or bracket[0] >= bracket[1]:
        raise ValueError(
            "reference-free calibration must record an increasing two-value "
            "bracket_over_h"
        )
    displacement_tolerance = float(
        calibration.get("displacement_tolerance_over_h", np.nan)
    )
    shift_tolerance = float(
        calibration.get("shift_tolerance_over_h", np.nan)
    )
    max_iterations = _exact_json_integer(
        calibration.get("max_iterations", 0),
        "reference_free_calibration.max_iterations",
    )
    if (
        not np.isfinite(displacement_tolerance)
        or not 0.0 < displacement_tolerance <= 1.0e-10
        or not np.isfinite(shift_tolerance)
        or not 0.0 < shift_tolerance <= 1.0e-12
        or max_iterations < 1
    ):
        raise ValueError(
            "reference-free calibration tolerances/iteration budget must be "
            "recorded and no looser than 1e-10 displacement and 1e-12 shift"
        )
    calibrated_root = float(calibration.get("calibrated_root_over_h", np.nan))
    declared_root_residual = float(
        calibration.get("root_displacement_over_h", np.nan)
    )
    declared_unshifted_displacement = float(
        calibration.get("unshifted_displacement_over_h", np.nan)
    )
    declared_iterations = _exact_json_integer(
        calibration.get("iterations", 0),
        "reference_free_calibration.iterations",
    )
    if not np.isfinite(calibrated_root):
        raise ValueError(
            "reference-free calibration must record finite "
            "calibrated_root_over_h"
        )
    if not np.isfinite(declared_root_residual):
        raise ValueError(
            "reference-free calibration must record finite "
            "root_displacement_over_h"
        )
    if not np.isfinite(declared_unshifted_displacement):
        raise ValueError(
            "reference-free calibration must record finite "
            "unshifted_displacement_over_h"
        )
    recomputed = calibrate_planar_mask_shift(
        q,
        epsilon_over_h=values["epsilon_over_h"],
        bracket_over_h=bracket,
        displacement_tolerance_over_h=displacement_tolerance,
        shift_tolerance_over_h=shift_tolerance,
        max_iterations=max_iterations,
        solid_cells=calibration_extents["solid_cells"],
        fluid_cells=calibration_extents["fluid_cells"],
        far_field_fit_cells=far_field_fit_cells,
        sampling=REFERENCE_FREE_CALIBRATION_SAMPLING,
    )
    if not np.isclose(
        calibrated_root,
        recomputed.shift_over_h,
        rtol=0.0,
        atol=shift_tolerance,
    ):
        raise ValueError(
            "declared calibrated_root_over_h does not match the recomputed "
            "reference-free planar-shear calibration"
        )
    if not np.isclose(
        declared_root_residual,
        recomputed.displacement_over_h,
        rtol=0.0,
        atol=1.0e-14,
    ) or abs(declared_root_residual) > displacement_tolerance:
        raise ValueError(
            "declared root_displacement_over_h does not match the recomputed "
            "calibration residual or exceeds its tolerance"
        )
    if not np.isclose(
        declared_unshifted_displacement,
        recomputed.unshifted_displacement_over_h,
        rtol=0.0,
        atol=1.0e-14,
    ) or declared_iterations != recomputed.iterations:
        raise ValueError(
            "declared unshifted displacement/iteration count does not match "
            "the recomputed reference-free calibration"
        )
    application = calibration.get("application")
    mask_shift = values["mask_midpoint_shift_over_h"]
    if application == "unshifted-control":
        if mask_shift != 0.0:
            raise ValueError(
                "an unshifted-control calibration application requires "
                "mask_midpoint_shift_over_h=0"
            )
    elif application == "calibrated-root":
        if not np.isclose(
            mask_shift, calibrated_root, rtol=0.0, atol=1.0e-14
        ):
            raise ValueError(
                "a calibrated-root application requires the mask midpoint "
                "shift to equal calibrated_root_over_h"
            )
    else:
        raise ValueError(
            "reference-free calibration application must be "
            "'unshifted-control' or 'calibrated-root'"
        )

    physics = config["physics"]
    centre = tuple(float(value) for value in physics["cylinder_centre"])
    radius = float(physics["cylinder_radius"])
    computed_centre_over_h = tuple(value / h_reference for value in centre)
    computed_radius_over_h = radius / h_reference
    for name, value in (
        ("cylinder_centre_x_over_h", computed_centre_over_h[0]),
        ("cylinder_centre_y_over_h", computed_centre_over_h[1]),
        ("cylinder_radius_over_h", computed_radius_over_h),
    ):
        if not np.isclose(value, round(value), rtol=0.0, atol=1.0e-12):
            raise ValueError(f"schema 1.2 requires integer phase lock for {name}")

    phase_lock = config.get("phase_lock", {})
    if phase_lock.get("grid_spacing") != "dx=dy=h":
        raise ValueError("schema 1.2 phase_lock.grid_spacing must be 'dx=dy=h'")
    declared_centre_over_h = tuple(
        float(value) for value in phase_lock.get("cylinder_centre_over_h", ())
    )
    if len(declared_centre_over_h) != 2 or not np.allclose(
        declared_centre_over_h,
        computed_centre_over_h,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "phase_lock.cylinder_centre_over_h must match the grid and geometry"
        )
    declared_radius_over_h = float(
        phase_lock.get("cylinder_radius_over_h", np.nan)
    )
    if not np.isclose(
        declared_radius_over_h,
        computed_radius_over_h,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "phase_lock.cylinder_radius_over_h must match the grid and geometry"
        )

    if config.get("acceptance_observables") != PRIMARY_ACCEPTANCE_OBSERVABLES:
        raise ValueError(
            "schema 1.2 acceptance_observables must freeze the primary "
            "Brinkman force, one-sided pressure, and separate slip estimators"
        )


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("case") != "dfg-cylinder-2d1-re20":
        raise ValueError("configuration case must be 'dfg-cylinder-2d1-re20'")
    schema_version = config.get("schema_version")
    if schema_version not in (CONFIG_SCHEMA_1_1, CONFIG_SCHEMA_1_2):
        raise ValueError("unsupported forward-verification config version")
    grid = config["grid"]
    physics = config["physics"]
    expected = DFG_2D1
    checks = {
        "lx": (float(grid["lx"]), expected["domain"][0]),
        "ly": (float(grid["ly"]), expected["domain"][1]),
        "viscosity": (float(physics["viscosity"]), expected["viscosity"]),
        "density": (float(physics["density"]), 1.0),
        "maximum_inlet_velocity": (
            float(physics["maximum_inlet_velocity"]),
            expected["maximum_inlet_velocity"],
        ),
        "mean_inlet_velocity": (
            float(physics["mean_inlet_velocity"]),
            expected["mean_inlet_velocity"],
        ),
        "reynolds_number": (
            float(physics["reynolds_number"]),
            expected["reynolds_number"],
        ),
        "cylinder_radius": (
            float(physics["cylinder_radius"]),
            expected["cylinder_radius"],
        ),
    }
    for name, (actual, reference) in checks.items():
        if abs(actual - reference) > 1.0e-14:
            raise ValueError(f"{name}={actual} does not match DFG 2D-1 {reference}")
    centre = tuple(float(value) for value in physics["cylinder_centre"])
    if centre != expected["cylinder_centre"]:
        raise ValueError("cylinder centre does not match DFG 2D-1")
    if config.get("initial_condition") != "zero-interior-with-inlet-boundary":
        raise ValueError("DFG runs require the declared zero-interior initial state")
    mean_velocity = float(physics["mean_inlet_velocity"])
    maximum_velocity = float(physics["maximum_inlet_velocity"])
    diameter = 2.0 * float(physics["cylinder_radius"])
    viscosity = float(physics["viscosity"])
    if abs(mean_velocity - 2.0 * maximum_velocity / 3.0) > 1.0e-14:
        raise ValueError("mean inlet velocity must be exactly two-thirds of U_max")
    if abs(mean_velocity * diameter / viscosity - 20.0) > 1.0e-12:
        raise ValueError("declared physical values do not produce Reynolds number 20")
    boundary = config.get("boundary_conditions", {})
    expected_boundary = {
        "inlet": "analytic-parabolic-face-average",
        "walls": "no-slip",
        "cylinder": "immersed-brinkman-no-slip-limit",
        "outlet": "zero-total-traction-finite-volume",
    }
    for name, value in expected_boundary.items():
        if boundary.get(name) != value:
            raise ValueError(f"boundary_conditions.{name} must be {value!r}")
    measurement = config.get("surface_measurement", {})
    expected_measurement = {
        "stress_tensor": "nu-grad-u-minus-pI",
        "normal": "cylinder-outward-into-fluid",
        "trace_method": "pure-fluid-stencil-radial-polynomial",
    }
    for name, value in expected_measurement.items():
        if measurement.get(name) != value:
            raise ValueError(f"surface_measurement.{name} must be {value!r}")
    offsets = [float(value) for value in measurement.get("offsets_over_h", [])]
    if len(offsets) < 5 or any(value <= 0.0 for value in offsets):
        raise ValueError("surface measurement requires at least five positive offsets")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("surface measurement offsets must be strictly increasing")
    epsilon_over_h = float(config["immersed_boundary"]["epsilon_over_h"])
    dx = float(grid["lx"]) / int(grid["nx"])
    dy = float(grid["ly"]) / int(grid["ny"])
    h_reference = max(dx, dy)
    if schema_version == CONFIG_SCHEMA_1_1:
        mask_shift_over_h = float(
            config["immersed_boundary"].get("mask_midpoint_shift_over_h", 0.0)
        )
        if mask_shift_over_h != 0.0:
            raise ValueError(
                "schema 1.1 is the unshifted legacy contract; nonzero mask "
                "midpoint shifts require schema 1.2"
            )
    else:
        _validate_phase_locked_brinkman_config(
            config, dx=dx, dy=dy, viscosity=viscosity
        )
        mask_shift_over_h = float(
            config["immersed_boundary"]["mask_midpoint_shift_over_h"]
        )
    interpolation_reach = (dx * dx + dy * dy) ** 0.5
    alpha_face_support_reach = 0.5 * max(dx, dy)
    dependency_reach_over_h = (
        interpolation_reach + alpha_face_support_reach
    ) / h_reference
    pure_fluid_distance_over_h = max(
        0.0, epsilon_over_h + mask_shift_over_h
    )
    if offsets[0] < pure_fluid_distance_over_h + dependency_reach_over_h:
        raise ValueError(
            "surface trace offsets need a full interpolation-and-alpha-dependency "
            "stencil margin outside the shifted diffuse-mask support"
        )
    pressure_degree = int(measurement.get("pressure_polynomial_degree", 0))
    velocity_degree = int(
        measurement.get("tangential_velocity_polynomial_degree", 0)
    )
    if not (1 <= pressure_degree <= len(offsets) - 2):
        raise ValueError("pressure fit must retain at least one residual DOF")
    if not (1 <= velocity_degree <= len(offsets) - 2):
        raise ValueError("velocity fit must retain at least one residual DOF")
    polynomial_degrees = [
        int(value)
        for value in measurement.get("polynomial_sensitivity_degrees", [])
    ]
    if (
        len(polynomial_degrees) < 2
        or polynomial_degrees[0] != pressure_degree
        or pressure_degree != velocity_degree
        or any(value < 1 or value > len(offsets) - 2 for value in polynomial_degrees)
        or any(
            left >= right
            for left, right in zip(polynomial_degrees, polynomial_degrees[1:])
        )
    ):
        raise ValueError(
            "polynomial sensitivity degrees must start at the common primary "
            "degree and strictly increase while retaining residual DOFs"
        )
    sensitivity_shifts = [
        float(value)
        for value in measurement.get("offset_sensitivity_shifts_over_h", [])
    ]
    if (
        len(sensitivity_shifts) < 2
        or sensitivity_shifts[0] != 0.0
        or any(value < 0.0 for value in sensitivity_shifts)
        or any(
            left >= right
            for left, right in zip(sensitivity_shifts, sensitivity_shifts[1:])
        )
    ):
        raise ValueError(
            "surface offset sensitivity shifts must start at zero and increase"
        )
    centre_x, centre_y = (float(value) for value in physics["cylinder_centre"])
    radius = float(physics["cylinder_radius"])
    outer_sample_radius = radius + (
        max(offsets) + max(sensitivity_shifts)
    ) * h_reference
    if (
        centre_x - outer_sample_radius < 0.5 * dx
        or centre_x + outer_sample_radius > float(grid["lx"]) - 0.5 * dx
        or centre_y - outer_sample_radius < 0.5 * dy
        or centre_y + outer_sample_radius > float(grid["ly"]) - 0.5 * dy
    ):
        raise ValueError(
            "surface trace offsets and sensitivity shifts leave the common "
            "p/u/v interpolation-lattice support"
        )
    quadrature_points = int(measurement.get("quadrature_points", 0))
    if quadrature_points < 32 or quadrature_points % 2:
        raise ValueError("surface quadrature must use an even count of at least 32")
    solver = config.get("solver", {})
    if schema_version == CONFIG_SCHEMA_1_2:
        limits = solver.get("stability_limits", {})
        expected_limit_names = {
            "c_adv_max",
            "c_diff_max",
            "alpha_max_times_dt_max",
        }
        if set(limits) != expected_limit_names:
            raise ValueError(
                "schema 1.2 solver.stability_limits must contain exactly "
                "c_adv_max, c_diff_max, and alpha_max_times_dt_max"
            )
        numeric_limits = {name: float(limits[name]) for name in limits}
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in numeric_limits.values()
        ):
            raise ValueError("all solver stability limits must be finite and positive")
        dt = float(solver["dt"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("schema 1.2 solver.dt must be finite and positive")
        configured_c_diff = viscosity * dt * (
            1.0 / (dx * dx) + 1.0 / (dy * dy)
        )
        configured_alpha_dt = (
            float(config["immersed_boundary"]["alpha_max"]) * dt
        )
        if configured_c_diff > numeric_limits["c_diff_max"]:
            raise ValueError(
                "configured explicit diffusion number exceeds c_diff_max"
            )
        if configured_alpha_dt > numeric_limits["alpha_max_times_dt_max"]:
            raise ValueError(
                "configured alpha_max*dt exceeds alpha_max_times_dt_max"
            )
    scheme = config["solver"].get("advection_scheme", "donor-cell")
    if scheme not in ("donor-cell", "central"):
        raise ValueError("solver.advection_scheme must be donor-cell or central")
    if scheme != "donor-cell" and (
        config.get("study", {}).get("development_only") is not True
        or config.get("study", {}).get("paper_eligible") is not False
    ):
        raise ValueError("central advection is currently a development-only diagnostic")
    sampling = config["immersed_boundary"].get("coefficient_sampling", "cell-average")
    if sampling not in ("cell-average", "native-face-point"):
        raise ValueError("unsupported immersed_boundary.coefficient_sampling")
    if sampling != "cell-average" and (
        config.get("study", {}).get("development_only") is not True
        or config.get("study", {}).get("paper_eligible") is not False
    ):
        raise ValueError("native-face sampling is a development-only diagnostic")
    return config


def _write_history(path: Path, diagnostics) -> None:
    fieldnames = (
        "iteration",
        "nonlinear_residual_rms",
        "nonlinear_residual_max",
        "pressure_residual_absolute",
        "pressure_residual_relative",
        "divergence_rms",
        "divergence_max",
        "inlet_flux",
        "outlet_flux",
        "relative_mass_flux_mismatch",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(int(diagnostics.iterations)):
            writer.writerow(
                {
                    "iteration": index + 1,
                    "nonlinear_residual_rms": float(
                        diagnostics.nonlinear_residual_rms_history[index]
                    ),
                    "nonlinear_residual_max": float(
                        diagnostics.nonlinear_residual_max_history[index]
                    ),
                    "pressure_residual_absolute": float(
                        diagnostics.pressure_residual_abs_history[index]
                    ),
                    "pressure_residual_relative": float(
                        diagnostics.pressure_residual_rel_history[index]
                    ),
                    "divergence_rms": float(
                        diagnostics.divergence_rms_history[index]
                    ),
                    "divergence_max": float(
                        diagnostics.divergence_max_history[index]
                    ),
                    "inlet_flux": float(
                        diagnostics.inlet_flux_history[index]
                    ),
                    "outlet_flux": float(
                        diagnostics.outlet_flux_history[index]
                    ),
                    "relative_mass_flux_mismatch": float(
                        diagnostics.mass_flux_mismatch_history[index]
                    ),
                }
            )


def _write_surface_samples(
    path: Path,
    trace,
    full_traction,
    reduced_traction,
    centre_x,
    centre_y,
    radius,
) -> None:
    theta = np.asarray(trace.theta)
    normal_x = np.cos(theta)
    normal_y = np.sin(theta)
    fieldnames = (
        "theta",
        "x",
        "y",
        "normal_x",
        "normal_y",
        "pressure_trace",
        "tangential_normal_derivative",
        "normal_normal_derivative",
        "tangential_surface_velocity",
        "normal_surface_velocity",
        "pressure_fit_rms",
        "tangential_fit_rms",
        "normal_fit_rms",
        "full_drag_traction_density",
        "full_lift_traction_density",
        "dfg_reduced_drag_traction_density",
        "dfg_reduced_lift_traction_density",
    )
    columns = {
        "theta": theta,
        "x": centre_x + radius * normal_x,
        "y": centre_y + radius * normal_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "pressure_trace": np.asarray(trace.pressure),
        "tangential_normal_derivative": np.asarray(
            trace.tangential_normal_derivative
        ),
        "normal_normal_derivative": np.asarray(
            trace.normal_normal_derivative
        ),
        "tangential_surface_velocity": np.asarray(
            trace.tangential_surface_velocity
        ),
        "normal_surface_velocity": np.asarray(trace.normal_surface_velocity),
        "pressure_fit_rms": np.asarray(trace.pressure_fit_rms),
        "tangential_fit_rms": np.asarray(trace.tangential_fit_rms),
        "normal_fit_rms": np.asarray(trace.normal_fit_rms),
        "full_drag_traction_density": np.asarray(full_traction.drag_density),
        "full_lift_traction_density": np.asarray(full_traction.lift_density),
        "dfg_reduced_drag_traction_density": np.asarray(
            reduced_traction.drag_density
        ),
        "dfg_reduced_lift_traction_density": np.asarray(
            reduced_traction.lift_density
        ),
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(theta.size):
            writer.writerow(
                {name: float(values[index]) for name, values in columns.items()}
            )


def _recirculation_length(
    state: ChannelState,
    grid,
    centre_x: float,
    centre_y: float,
    radius: float,
    interface_width: float,
) -> float:
    u_cell, _ = cell_center_velocity(state)
    x = (np.arange(grid.nx) + 0.5) * grid.dx
    y = (np.arange(grid.ny) + 0.5) * grid.dy
    row = int(np.argmin(np.abs(y - centre_y)))
    values = np.asarray(u_cell[:, row])
    rear = centre_x + radius
    candidates = np.flatnonzero((x > rear + interface_width) & (values < 0.0))
    if candidates.size == 0:
        return 0.0
    start = int(candidates[0])
    end = start
    while end + 1 < grid.nx and values[end + 1] < 0.0:
        end += 1
    if end + 1 < grid.nx and values[end + 1] != values[end]:
        fraction = -values[end] / (values[end + 1] - values[end])
        zero_x = x[end] + fraction * (x[end + 1] - x[end])
    else:
        zero_x = x[end]
    return float(max(zero_x - rear, 0.0))


def _strict_nonzero_face_support(alpha, grid) -> dict:
    """Return auditable bounding boxes of the exact nonzero face support."""

    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    support = {}
    for component, values in (("u", alpha_u), ("v", alpha_v)):
        indices = np.argwhere(np.asarray(values) != 0.0)
        if indices.size == 0:
            raise ValueError(
                f"strict nonzero alpha_{component} support is empty; "
                "cannot construct a finite-Brinkman control-volume audit"
            )
        lower = indices.min(axis=0)
        upper = indices.max(axis=0)
        support[component] = {
            "i_start": int(lower[0]),
            "i_stop": int(upper[0]),
            "j_start": int(lower[1]),
            "j_stop": int(upper[1]),
            "nonzero_face_count": int(indices.shape[0]),
        }
    return support


def _nested_finite_brinkman_control_volumes(
    alpha, grid, margins=(1, 2, 3)
) -> tuple[dict, tuple[tuple[int, MACStaggeredControlVolume], ...]]:
    """Build three nested interior MAC boxes around all nonzero alpha faces.

    The small margins make this diagnostic available even on the 86-by-16
    smoke grid.  They are an algebraic/residual audit only: no claim is made
    that these boundaries are remote from the diffuse interface.
    """

    margins = tuple(margins)
    if (
        len(margins) != 3
        or any(not isinstance(value, (int, np.integer)) for value in margins)
        or any(value <= 0 for value in margins)
        or any(left >= right for left, right in zip(margins, margins[1:]))
    ):
        raise ValueError("control-volume margins must be three increasing integers")

    support = _strict_nonzero_face_support(alpha, grid)
    boxes = []
    for margin in margins:
        u_support = support["u"]
        v_support = support["v"]
        control_volume = MACStaggeredControlVolume(
            u_i_start=u_support["i_start"] - margin,
            u_i_stop=u_support["i_stop"] + margin,
            u_j_start=u_support["j_start"] - margin,
            u_j_stop=u_support["j_stop"] + margin,
            v_i_start=v_support["i_start"] - margin,
            v_i_stop=v_support["i_stop"] + margin,
            v_j_start=v_support["j_start"] - margin,
            v_j_stop=v_support["j_stop"] + margin,
        )
        valid = (
            1 <= control_volume.u_i_start <= control_volume.u_i_stop <= grid.nx - 1
            and 1
            <= control_volume.u_j_start
            <= control_volume.u_j_stop
            <= grid.ny - 2
            and 1
            <= control_volume.v_i_start
            <= control_volume.v_i_stop
            <= grid.nx - 2
            and 1
            <= control_volume.v_j_start
            <= control_volume.v_j_stop
            <= grid.ny - 1
        )
        if not valid:
            raise ValueError(
                "domain boundary leaves insufficient room for the requested "
                f"interior finite-Brinkman control volume at margin={margin}: "
                f"support={support}, requested={control_volume}"
            )
        boxes.append((int(margin), control_volume))
    return support, tuple(boxes)


def _control_volume_indices(control_volume: MACStaggeredControlVolume) -> dict:
    return {
        "u_momentum": {
            "i_start": control_volume.u_i_start,
            "i_stop": control_volume.u_i_stop,
            "j_start": control_volume.u_j_start,
            "j_stop": control_volume.u_j_stop,
        },
        "v_momentum": {
            "i_start": control_volume.v_i_start,
            "i_stop": control_volume.v_i_stop,
            "j_start": control_volume.v_j_start,
            "j_stop": control_volume.v_j_stop,
        },
    }


def _control_volume_physical_boundaries(
    control_volume: MACStaggeredControlVolume, grid
) -> dict:
    """Describe the two half-cell-offset staggered dual rectangles."""

    return {
        "u_momentum_dual_rectangle": {
            "x_min": (control_volume.u_i_start - 0.5) * grid.dx,
            "x_max": (control_volume.u_i_stop + 0.5) * grid.dx,
            "y_min": control_volume.u_j_start * grid.dy,
            "y_max": (control_volume.u_j_stop + 1.0) * grid.dy,
            "coordinate_definition": (
                "x=[(u_i_start-1/2)dx,(u_i_stop+1/2)dx], "
                "y=[u_j_start*dy,(u_j_stop+1)dy]"
            ),
        },
        "v_momentum_dual_rectangle": {
            "x_min": control_volume.v_i_start * grid.dx,
            "x_max": (control_volume.v_i_stop + 1.0) * grid.dx,
            "y_min": (control_volume.v_j_start - 0.5) * grid.dy,
            "y_max": (control_volume.v_j_stop + 0.5) * grid.dy,
            "coordinate_definition": (
                "x=[v_i_start*dx,(v_i_stop+1)dx], "
                "y=[(v_j_start-1/2)dy,(v_j_stop+1/2)dy]"
            ),
        },
    }


def _finite_brinkman_discrete_momentum_audit(
    state,
    alpha,
    inlet,
    grid,
    params,
    force_scale: float,
    support: dict,
    control_volumes,
) -> dict:
    """Serialize nested native-MAC balances without a sharp-traction claim."""

    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    alpha_u_nonzero = np.asarray(alpha_u) != 0.0
    alpha_v_nonzero = np.asarray(alpha_v) != 0.0
    entries = []
    for margin, control_volume in control_volumes:
        u_captured = np.zeros_like(alpha_u_nonzero)
        u_captured[
            control_volume.u_i_start : control_volume.u_i_stop + 1,
            control_volume.u_j_start : control_volume.u_j_stop + 1,
        ] = True
        v_captured = np.zeros_like(alpha_v_nonzero)
        v_captured[
            control_volume.v_i_start : control_volume.v_i_stop + 1,
            control_volume.v_j_start : control_volume.v_j_stop + 1,
        ] = True
        captures_u = bool(np.all(~alpha_u_nonzero | u_captured))
        captures_v = bool(np.all(~alpha_v_nonzero | v_captured))
        if not (captures_u and captures_v):
            raise RuntimeError(
                f"margin={margin} control volume does not capture all strict "
                "nonzero alpha_face support"
            )

        balance = mac_staggered_control_volume_momentum_balance(
            state, alpha, inlet, grid, params, control_volume
        )
        inferred_x = float(balance.inferred_brinkman_reaction_x)
        inferred_y = float(balance.inferred_brinkman_reaction_y)
        captured_x = float(balance.captured_brinkman_force_x)
        captured_y = float(balance.captured_brinkman_force_y)
        global_x = float(balance.global_brinkman_force_x)
        global_y = float(balance.global_brinkman_force_y)
        closure_x = float(balance.algebraic_closure_x)
        closure_y = float(balance.algebraic_closure_y)
        global_norm = max(float(np.hypot(global_x, global_y)), 1.0e-12)
        component_scale_x = max(abs(global_x), 1.0e-12)
        component_scale_y = max(abs(global_y), 1.0e-12)
        entries.append(
            {
                "margin_cells_outside_strict_support": margin,
                "indices_inclusive": _control_volume_indices(control_volume),
                "physical_boundaries": _control_volume_physical_boundaries(
                    control_volume, grid
                ),
                "captures_all_strict_nonzero_alpha_face_support": {
                    "u": captures_u,
                    "v": captures_v,
                    "both": captures_u and captures_v,
                },
                "inferred_brinkman_reaction": {
                    "force_x": inferred_x,
                    "force_y": inferred_y,
                    "drag_coefficient": 2.0 * inferred_x / force_scale,
                    "lift_coefficient": 2.0 * inferred_y / force_scale,
                },
                "flux_decomposition_body_force_sign": {
                    "convective": {
                        "force_x": float(balance.convective_contribution_x),
                        "force_y": float(balance.convective_contribution_y),
                    },
                    "pressure": {
                        "force_x": float(balance.pressure_contribution_x),
                        "force_y": float(balance.pressure_contribution_y),
                    },
                    "viscous": {
                        "force_x": float(balance.viscous_contribution_x),
                        "force_y": float(balance.viscous_contribution_y),
                    },
                    "source": {
                        "force_x": float(balance.source_contribution_x),
                        "force_y": float(balance.source_contribution_y),
                    },
                },
                "brinkman_volume_reaction": {
                    "captured_force_x": captured_x,
                    "captured_force_y": captured_y,
                    "global_force_x": global_x,
                    "global_force_y": global_y,
                    "uncaptured_force_x": float(
                        balance.uncaptured_brinkman_force_x
                    ),
                    "uncaptured_force_y": float(
                        balance.uncaptured_brinkman_force_y
                    ),
                },
                "momentum_residual_integral": {
                    "x": float(balance.momentum_residual_integral_x),
                    "y": float(balance.momentum_residual_integral_y),
                },
                "algebraic_closure": {"x": closure_x, "y": closure_y},
                "relative_cv_volume_closure": float(
                    np.hypot(inferred_x - captured_x, inferred_y - captured_y)
                    / global_norm
                ),
                "relative_cv_volume_closure_by_component": {
                    "x": abs(inferred_x - captured_x) / component_scale_x,
                    "y": abs(inferred_y - captured_y) / component_scale_y,
                    "normalization": "max(abs(global_force_component), 1e-12)",
                },
                "relative_algebraic_closure": float(
                    np.hypot(closure_x, closure_y) / global_norm
                ),
                "relative_algebraic_closure_by_component": {
                    "x": abs(closure_x) / component_scale_x,
                    "y": abs(closure_y) / component_scale_y,
                    "normalization": "max(abs(global_force_component), 1e-12)",
                },
            }
        )

    drag_values = [
        item["inferred_brinkman_reaction"]["drag_coefficient"]
        for item in entries
    ]
    lift_values = [
        item["inferred_brinkman_reaction"]["lift_coefficient"]
        for item in entries
    ]
    global_drag_coefficient = 2.0 * entries[0]["brinkman_volume_reaction"][
        "global_force_x"
    ] / force_scale
    global_lift_coefficient = 2.0 * entries[0]["brinkman_volume_reaction"][
        "global_force_y"
    ] / force_scale
    drag_span = max(drag_values) - min(drag_values)
    lift_span = max(lift_values) - min(lift_values)
    inferred_force_vectors = [
        (
            item["inferred_brinkman_reaction"]["force_x"],
            item["inferred_brinkman_reaction"]["force_y"],
        )
        for item in entries
    ]
    vector_force_spread = max(
        float(np.hypot(left[0] - right[0], left[1] - right[1]))
        for left in inferred_force_vectors
        for right in inferred_force_vectors
    )
    global_force_norm = float(
        np.hypot(
            entries[0]["brinkman_volume_reaction"]["global_force_x"],
            entries[0]["brinkman_volume_reaction"]["global_force_y"],
        )
    )
    return {
        "classification": (
            "finite-Brinkman discrete momentum-balance audit; not sharp "
            "interface traction"
        ),
        "boundary_proximity_limitation": (
            "Margins 1/2/3 ensure three nested boxes on the coarsest grid; "
            "the control surfaces are not asserted to be remote from the "
            "diffuse interface. This audit tests algebraic closure and the "
            "reported steady residual only."
        ),
        "strict_nonzero_support_definition": "alpha_face != 0.0",
        "discrete_identities": {
            "flux_balance": (
                "inferred reaction = source + convective + pressure + viscous "
                "contributions, with every contribution reported in the body-"
                "force sign convention"
            ),
            "residual_balance": (
                "inferred reaction = captured Brinkman volume reaction - "
                "momentum residual integral"
            ),
            "algebraic_closure": (
                "inferred reaction - captured Brinkman volume reaction + "
                "momentum residual integral"
            ),
            "relative_cv_volume_closure": (
                "L2 norm(inferred reaction - captured volume reaction) / "
                "max(L2 norm(global Brinkman volume reaction), 1e-12)"
            ),
            "componentwise_normalization": (
                "absolute component error / max(abs(global force component), "
                "1e-12); this is the protocol gate and does not let drag mask "
                "a small or zero lift"
            ),
        },
        "strict_nonzero_alpha_face_support": support,
        "control_volumes": entries,
        "summary": {
            "control_volume_count": len(entries),
            "global_brinkman_drag_coefficient": global_drag_coefficient,
            "global_brinkman_lift_coefficient": global_lift_coefficient,
            "vector_force_spread": vector_force_spread,
            "vector_force_spread_normalized": vector_force_spread
            / max(global_force_norm, 1.0e-12),
            "vector_force_spread_normalization": (
                "max pairwise L2 inferred-force difference / "
                "max(L2 global Brinkman force, 1e-12); protocol gate"
            ),
            "component_coefficient_spans_classification": (
                "diagnostic only; not the vector-force spread protocol gate"
            ),
            "drag_coefficient_spread": drag_span,
            "lift_coefficient_spread": lift_span,
            "normalized_drag_coefficient_spread": drag_span
            / max(abs(global_drag_coefficient), 1.0e-12),
            "normalized_lift_coefficient_spread": lift_span
            / max(abs(global_lift_coefficient), 1.0e-12),
            "maximum_relative_cv_volume_closure": max(
                item["relative_cv_volume_closure"] for item in entries
            ),
            "maximum_relative_cv_volume_closure_by_component": {
                component: max(
                    item["relative_cv_volume_closure_by_component"][component]
                    for item in entries
                )
                for component in ("x", "y")
            },
            "maximum_relative_algebraic_closure": max(
                item["relative_algebraic_closure"] for item in entries
            ),
            "maximum_relative_algebraic_closure_by_component": {
                component: max(
                    item["relative_algebraic_closure_by_component"][component]
                    for item in entries
                )
                for component in ("x", "y")
            },
        },
    }


def _require_finite_solve_output(state, diagnostics) -> None:
    """Reject a non-finite solve before any force or trace postprocessing."""

    failures = []
    for group_name, record in (("state", state), ("diagnostics", diagnostics)):
        for field_name, value in zip(record._fields, record):
            array = np.asarray(value)
            finite = np.isfinite(array)
            if bool(np.all(finite)):
                continue
            bad = np.argwhere(~finite)
            first_index = tuple(int(item) for item in bad[0]) if bad.size else ()
            first_value = array[first_index] if first_index else array.item()
            failures.append(
                f"{group_name}.{field_name}{first_index}={first_value!r}"
            )
    if failures:
        raise FloatingPointError(
            "steady solve produced non-finite state/diagnostics before "
            "postprocessing: " + "; ".join(failures)
        )


def _stability_metrics(
    state, grid, params, alpha_max: float, limits: dict | None = None
) -> dict:
    max_abs_u = float(jnp.max(jnp.abs(state.u)))
    max_abs_v = float(jnp.max(jnp.abs(state.v)))
    values = {
        "max_abs_u": max_abs_u,
        "max_abs_v": max_abs_v,
        "c_adv": float(
            params.dt * (max_abs_u / grid.dx + max_abs_v / grid.dy)
        ),
        "c_diff": float(
            params.viscosity
            * params.dt
            * (1.0 / (grid.dx * grid.dx) + 1.0 / (grid.dy * grid.dy))
        ),
        "alpha_max_times_dt": float(alpha_max * params.dt),
        "definitions": {
            "c_adv": "dt*(max_abs_u/dx + max_abs_v/dy)",
            "c_diff": "nu*dt*(1/dx^2 + 1/dy^2)",
            "alpha_max_times_dt": "alpha_max*dt",
        },
    }
    if limits is None:
        values["configured_limits"] = None
        values["within_configured_limits"] = None
        values["all_within_configured_limits"] = None
        return values

    numeric_limits = {name: float(value) for name, value in limits.items()}
    checks = {
        "c_adv": values["c_adv"] <= numeric_limits["c_adv_max"],
        "c_diff": values["c_diff"] <= numeric_limits["c_diff_max"],
        "alpha_max_times_dt": (
            values["alpha_max_times_dt"]
            <= numeric_limits["alpha_max_times_dt_max"]
        ),
    }
    values["configured_limits"] = numeric_limits
    values["within_configured_limits"] = checks
    values["all_within_configured_limits"] = bool(all(checks.values()))
    return values


def run(config_path: Path, output_dir: Path) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_dir}")

    config = _load_config(config_path)
    started_utc = datetime.now(timezone.utc)
    environment = capture("float64")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "config.json"
    shutil.copyfile(config_path, resolved_config_path)
    environment_path = output_dir / "environment.json"
    write_json(environment_path, environment)

    grid_config = config["grid"]
    physics = config["physics"]
    immersed = config["immersed_boundary"]
    solver = config["solver"]
    grid = create_channel_grid(
        int(grid_config["nx"]),
        int(grid_config["ny"]),
        float(grid_config["lx"]),
        float(grid_config["ly"]),
    )
    centre_x, centre_y = (float(value) for value in physics["cylinder_centre"])
    radius = float(physics["cylinder_radius"])
    diameter = 2.0 * radius
    h_reference = max(grid.dx, grid.dy)
    epsilon_over_h = float(immersed["epsilon_over_h"])
    epsilon = epsilon_over_h * h_reference
    alpha_max = float(immersed["alpha_max"])
    mask_shift_over_h = float(immersed.get("mask_midpoint_shift_over_h", 0.0))
    mask_midpoint_shift = mask_shift_over_h * h_reference
    minimum_nominal_pure_fluid_distance = max(
        0.0, epsilon + mask_midpoint_shift
    )
    if immersed.get("material_mapping") != "compact-cosine-heaviside":
        raise ValueError(
            "DFG validation requires the compact-cosine material mapping"
        )
    x, y = cell_centres(grid, dtype=jnp.float64)
    nominal_phi = (
        jnp.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2) - radius
    )
    # Positive shift expands the penalised mask into the nominal fluid.
    mask_phi = nominal_phi - mask_midpoint_shift
    fluid_fraction = smooth_heaviside(mask_phi, epsilon)
    cell_alpha = alpha_max * (1.0 - fluid_fraction)
    coefficient_sampling = immersed.get("coefficient_sampling", "cell-average")
    alpha = cell_alpha
    if coefficient_sampling == "native-face-point":
        def coefficient_at(x_face, y_face):
            phi = jnp.sqrt((x_face-centre_x)**2 + (y_face-centre_y)**2) - radius - mask_midpoint_shift
            return alpha_max * (1.0 - smooth_heaviside(phi, epsilon))
        alpha = sample_native_face_coefficient(coefficient_at, grid, dtype=cell_alpha.dtype)
    alpha_face_support, control_volumes = (
        _nested_finite_brinkman_control_volumes(alpha, grid)
    )

    maximum_inlet = float(physics["maximum_inlet_velocity"])
    inlet = parabolic_inlet_face_averages(
        grid, maximum_inlet, dtype=jnp.float64
    )
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    params = ChannelParams(
        viscosity=float(physics["viscosity"]),
        dt=float(solver["dt"]),
        pressure_tolerance=float(solver["pressure_tolerance"]),
        pressure_atol=float(solver["pressure_atol"]),
        pressure_maxiter=int(solver["pressure_maxiter"]),
        include_advection=bool(solver["include_advection"]),
        advection_scheme=solver.get("advection_scheme", "donor-cell"),
        nonlinear_rtol=float(solver["nonlinear_rtol"]),
        nonlinear_atol=float(solver["nonlinear_atol"]),
        nonlinear_min_steps=int(solver["nonlinear_min_steps"]),
    )
    chunk_steps = solver.get("chunk_steps")
    if chunk_steps:
        # Chunked dispatch (see solve_steady_chunked): the whole solve cannot
        # be one compiled program, so compile time is that of the first chunk
        # and the single timing sample is the total wall time of the solve.
        solve_started = time.perf_counter()
        state, diagnostics, chunk_report = solve_steady_chunked(
            state0,
            alpha,
            inlet,
            grid,
            params,
            int(solver["max_steps"]),
            int(chunk_steps),
        )
        forward_samples = [time.perf_counter() - solve_started]
        compile_seconds = float(chunk_report["chunks"][0]["seconds"]) if chunk_report["chunks"] else 0.0
    else:
        # Arrays are passed as arguments, never closed over as constants
        # (see compile_and_sample_args for the GPU miscompilation this avoids).
        result, compile_seconds, forward_samples = compile_and_sample_args(
            lambda s0, a, i, steps: solve_steady(s0, a, i, grid, params, steps),
            (state0, alpha, inlet, int(solver["max_steps"])),
            int(config["timing_samples"]),
            static_argnums=(3,),
        )
        state, diagnostics = result
        chunk_report = None
    _require_finite_solve_output(state, diagnostics)
    stability = _stability_metrics(
        state,
        grid,
        params,
        alpha_max,
        limits=solver.get("stability_limits"),
    )

    brinkman_force_x, brinkman_force_y = brinkman_force(state, alpha, grid)
    density = float(physics["density"])
    mean_velocity = float(physics["mean_inlet_velocity"])
    # Pressure, viscosity, and Brinkman reaction are all kinematic (per unit
    # density), so density cancels from the DFG coefficient normalization.
    force_scale = mean_velocity * mean_velocity * diameter
    brinkman_drag_coefficient = 2.0 * brinkman_force_x / force_scale
    brinkman_lift_coefficient = 2.0 * brinkman_force_y / force_scale
    finite_brinkman_discrete_audit = _finite_brinkman_discrete_momentum_audit(
        state,
        alpha,
        inlet,
        grid,
        params,
        force_scale,
        alpha_face_support,
        control_volumes,
    )
    surface_config = config["surface_measurement"]
    offsets_over_h = jnp.asarray(
        surface_config["offsets_over_h"], dtype=jnp.float64
    )
    quadrature_points = int(surface_config["quadrature_points"])
    trace = dfg_circle_exterior_trace(
        state,
        grid,
        centre_x,
        centre_y,
        radius,
        h_reference,
        offsets_over_h,
        quadrature_points=quadrature_points,
        pressure_degree=int(surface_config["pressure_polynomial_degree"]),
        tangential_velocity_degree=int(
            surface_config["tangential_velocity_polynomial_degree"]
        ),
        minimum_fluid_signed_distance=minimum_nominal_pure_fluid_distance,
    )
    traction = integrate_dfg_circle_traction(
        trace.pressure,
        trace.tangential_normal_derivative,
        trace.theta,
        float(physics["viscosity"]),
        radius,
        normal_normal_derivative=trace.normal_normal_derivative,
    )
    reduced_traction = integrate_dfg_circle_traction(
        trace.pressure,
        trace.tangential_normal_derivative,
        trace.theta,
        float(physics["viscosity"]),
        radius,
    )
    refined_trace = dfg_circle_exterior_trace(
        state,
        grid,
        centre_x,
        centre_y,
        radius,
        h_reference,
        offsets_over_h,
        quadrature_points=2 * quadrature_points,
        pressure_degree=int(surface_config["pressure_polynomial_degree"]),
        tangential_velocity_degree=int(
            surface_config["tangential_velocity_polynomial_degree"]
        ),
        minimum_fluid_signed_distance=minimum_nominal_pure_fluid_distance,
    )
    refined_traction = integrate_dfg_circle_traction(
        refined_trace.pressure,
        refined_trace.tangential_normal_derivative,
        refined_trace.theta,
        float(physics["viscosity"]),
        radius,
        normal_normal_derivative=refined_trace.normal_normal_derivative,
    )
    refined_reduced_traction = integrate_dfg_circle_traction(
        refined_trace.pressure,
        refined_trace.tangential_normal_derivative,
        refined_trace.theta,
        float(physics["viscosity"]),
        radius,
    )
    surface_drag_coefficient = 2.0 * refined_traction.force_x / force_scale
    surface_lift_coefficient = 2.0 * refined_traction.force_y / force_scale
    pressure_drag_coefficient = (
        2.0 * refined_traction.pressure_force_x / force_scale
    )
    viscous_drag_coefficient = (
        2.0 * refined_traction.viscous_force_x / force_scale
    )
    pressure_lift_coefficient = (
        2.0 * refined_traction.pressure_force_y / force_scale
    )
    viscous_lift_coefficient = (
        2.0 * refined_traction.viscous_force_y / force_scale
    )
    reduced_drag_coefficient = (
        2.0 * refined_reduced_traction.force_x / force_scale
    )
    reduced_lift_coefficient = (
        2.0 * refined_reduced_traction.force_y / force_scale
    )
    pressure_rear = refined_trace.pressure[0]
    pressure_front = refined_trace.pressure[quadrature_points]
    pressure_difference = pressure_front - pressure_rear
    offset_sensitivity = []
    for shift in surface_config["offset_sensitivity_shifts_over_h"]:
        shift = float(shift)
        if shift == 0.0:
            shifted_trace = refined_trace
            shifted_traction = refined_traction
            shifted_reduced_traction = refined_reduced_traction
        else:
            shifted_offsets = offsets_over_h + shift
            shifted_trace = dfg_circle_exterior_trace(
                state,
                grid,
                centre_x,
                centre_y,
                radius,
                h_reference,
                shifted_offsets,
                quadrature_points=2 * quadrature_points,
                pressure_degree=int(
                    surface_config["pressure_polynomial_degree"]
                ),
                tangential_velocity_degree=int(
                    surface_config["tangential_velocity_polynomial_degree"]
                ),
                minimum_fluid_signed_distance=minimum_nominal_pure_fluid_distance,
            )
            shifted_traction = integrate_dfg_circle_traction(
                shifted_trace.pressure,
                shifted_trace.tangential_normal_derivative,
                shifted_trace.theta,
                float(physics["viscosity"]),
                radius,
                normal_normal_derivative=shifted_trace.normal_normal_derivative,
            )
            shifted_reduced_traction = integrate_dfg_circle_traction(
                shifted_trace.pressure,
                shifted_trace.tangential_normal_derivative,
                shifted_trace.theta,
                float(physics["viscosity"]),
                radius,
            )
        shifted_drag = 2.0 * shifted_traction.force_x / force_scale
        shifted_lift = 2.0 * shifted_traction.force_y / force_scale
        shifted_reduced_drag = (
            2.0 * shifted_reduced_traction.force_x / force_scale
        )
        shifted_reduced_lift = (
            2.0 * shifted_reduced_traction.force_y / force_scale
        )
        shifted_pressure_difference = (
            shifted_trace.pressure[quadrature_points]
            - shifted_trace.pressure[0]
        )
        offset_sensitivity.append(
            {
                "shift_over_h": shift,
                "offsets_over_h": [
                    float(value) for value in offsets_over_h + shift
                ],
                "drag_coefficient": float(shifted_drag),
                "lift_coefficient": float(shifted_lift),
                "dfg_reduced_drag_coefficient": float(shifted_reduced_drag),
                "dfg_reduced_lift_coefficient": float(shifted_reduced_lift),
                "pressure_difference": float(shifted_pressure_difference),
            }
        )
    polynomial_sensitivity = []
    primary_degree = int(surface_config["pressure_polynomial_degree"])
    for degree in surface_config["polynomial_sensitivity_degrees"]:
        degree = int(degree)
        if degree == primary_degree:
            degree_trace = refined_trace
            degree_traction = refined_traction
            degree_reduced_traction = refined_reduced_traction
        else:
            degree_trace = dfg_circle_exterior_trace(
                state,
                grid,
                centre_x,
                centre_y,
                radius,
                h_reference,
                offsets_over_h,
                quadrature_points=2 * quadrature_points,
                pressure_degree=degree,
                tangential_velocity_degree=degree,
                minimum_fluid_signed_distance=minimum_nominal_pure_fluid_distance,
            )
            degree_traction = integrate_dfg_circle_traction(
                degree_trace.pressure,
                degree_trace.tangential_normal_derivative,
                degree_trace.theta,
                float(physics["viscosity"]),
                radius,
                normal_normal_derivative=degree_trace.normal_normal_derivative,
            )
            degree_reduced_traction = integrate_dfg_circle_traction(
                degree_trace.pressure,
                degree_trace.tangential_normal_derivative,
                degree_trace.theta,
                float(physics["viscosity"]),
                radius,
            )
        polynomial_sensitivity.append(
            {
                "degree": degree,
                "full_drag_coefficient": float(
                    2.0 * degree_traction.force_x / force_scale
                ),
                "full_lift_coefficient": float(
                    2.0 * degree_traction.force_y / force_scale
                ),
                "dfg_reduced_drag_coefficient": float(
                    2.0 * degree_reduced_traction.force_x / force_scale
                ),
                "dfg_reduced_lift_coefficient": float(
                    2.0 * degree_reduced_traction.force_y / force_scale
                ),
                "pressure_difference": float(
                    degree_trace.pressure[quadrature_points]
                    - degree_trace.pressure[0]
                ),
                "pressure_fit_condition_number": float(
                    degree_trace.pressure_fit_condition_number
                ),
                "velocity_fit_condition_number": float(
                    degree_trace.velocity_fit_condition_number
                ),
                "pressure_value_weight_l1": float(
                    degree_trace.pressure_value_weight_l1
                ),
                "velocity_value_weight_l1": float(
                    degree_trace.velocity_value_weight_l1
                ),
                "velocity_derivative_weight_l1_normalized": float(
                    degree_trace.velocity_derivative_weight_l1_normalized
                ),
            }
        )
    offset_drag_values = [item["drag_coefficient"] for item in offset_sensitivity]
    offset_pressure_values = [
        item["pressure_difference"] for item in offset_sensitivity
    ]
    offset_lift_values = [item["lift_coefficient"] for item in offset_sensitivity]
    degree_drag_values = [
        item["full_drag_coefficient"] for item in polynomial_sensitivity
    ]
    degree_lift_values = [
        item["full_lift_coefficient"] for item in polynomial_sensitivity
    ]
    degree_pressure_values = [
        item["pressure_difference"] for item in polynomial_sensitivity
    ]
    recirculation = _recirculation_length(
        state,
        grid,
        centre_x,
        centre_y,
        radius,
        minimum_nominal_pure_fluid_distance,
    )
    div_rms, div_max, inlet_flux, outlet_flux, mismatch = flux_diagnostics(
        state, grid
    )
    discrete_inlet_mean = inlet_flux / grid.ly
    reference_drag = float(DFG_2D1["reference_drag_coefficient"])
    reference_lift = float(DFG_2D1["reference_lift_coefficient"])
    reference_pressure = float(DFG_2D1["reference_pressure_difference"])
    original_intervals = DFG_2D1["original_reference_intervals"]

    def in_original_interval(name: str, value: float) -> bool:
        lower, upper = original_intervals[name]
        return bool(lower <= value <= upper)

    metrics = {
        "case": "dfg-cylinder-2d1-re20",
        "configuration_schema_version": config["schema_version"],
        "advection_scheme": params.advection_scheme,
        "acceptance_observables": {
            "primary": PRIMARY_ACCEPTANCE_OBSERVABLES,
            "diagnostic_only": [
                "metrics.numerical.surface_drag_coefficient",
                "metrics.numerical.surface_lift_coefficient",
                "metrics.numerical.dfg_reduced_drag_coefficient",
                "metrics.numerical.dfg_reduced_lift_coefficient",
                "metrics.finite_brinkman_discrete_momentum_audit",
            ],
            "selection_rule": (
                "frozen before runs; diagnostic estimators cannot replace a "
                "failed primary observable"
            ),
        },
        "reference": {
            "url": DFG_2D1["reference_url"],
            "original_doi": DFG_2D1["original_reference_doi"],
            "high_precision_source": DFG_2D1["high_precision_reference"],
            "drag_coefficient": reference_drag,
            "lift_coefficient": reference_lift,
            "pressure_difference": reference_pressure,
            "original_intervals": original_intervals,
        },
        "immersed_boundary": {
            "coefficient_sampling": coefficient_sampling,
            "reaction_definition": (
                "specific force: volume integral of alpha_face * velocity_face"
            ),
            "material_mapping": immersed["material_mapping"],
            "alpha_max": alpha_max,
            "alpha_max_h_squared_over_nu": (
                alpha_max * h_reference * h_reference
                / float(physics["viscosity"])
            ),
            "epsilon": epsilon,
            "epsilon_over_h": epsilon_over_h,
            "mask_midpoint_shift": mask_midpoint_shift,
            "mask_midpoint_shift_over_h": mask_shift_over_h,
            "mask_shift_sign": (
                "positive expands the penalised mask into nominal fluid"
            ),
            "minimum_nominal_pure_fluid_signed_distance": (
                minimum_nominal_pure_fluid_distance
            ),
            "minimum_nominal_pure_fluid_signed_distance_over_h": (
                minimum_nominal_pure_fluid_distance / h_reference
            ),
            "h_reference": h_reference,
            "diameter_over_h": diameter / h_reference,
            "diffuse_transition_thickness_over_diameter": 2.0
            * epsilon
            / diameter,
            "penalisation_length": (
                float(physics["viscosity"]) / alpha_max
            )
            ** 0.5,
            "penalisation_length_over_h": (
                float(physics["viscosity"]) / alpha_max
            )
            ** 0.5
            / h_reference,
            "delta_b_over_h": (
                float(physics["viscosity"]) / alpha_max
            )
            ** 0.5
            / h_reference,
            "reference_free_calibration": immersed.get(
                "reference_free_calibration"
            ),
            "phase_lock": {
                "dx_over_h": grid.dx / h_reference,
                "dy_over_h": grid.dy / h_reference,
                "cylinder_centre_over_h": [
                    centre_x / h_reference,
                    centre_y / h_reference,
                ],
                "cylinder_radius_over_h": radius / h_reference,
            },
        },
        "finite_brinkman_discrete_momentum_audit": (
            finite_brinkman_discrete_audit
        ),
        "stability": stability,
        "surface_measurement": {
            "force_definition": (
                "finite-Brinkman full specific traction: integral of "
                "(nu grad(u) - p I) n over the nominal circle"
            ),
            "dfg_reduced_definition": (
                "sharp-no-slip-limit DFG tangential-derivative formula, with "
                "the finite-field normal derivative omitted"
            ),
            "pressure_and_force_units": "kinematic, per unit density",
            "configured_density": density,
            "normal": surface_config["normal"],
            "trace_method": surface_config["trace_method"],
            "offsets_over_h": [float(value) for value in offsets_over_h],
            "base_quadrature_points": quadrature_points,
            "reported_quadrature_points": 2 * quadrature_points,
            "force_x": float(refined_traction.force_x),
            "force_y": float(refined_traction.force_y),
            "pressure_force_x": float(refined_traction.pressure_force_x),
            "pressure_force_y": float(refined_traction.pressure_force_y),
            "viscous_force_x": float(refined_traction.viscous_force_x),
            "viscous_force_y": float(refined_traction.viscous_force_y),
            "drag_coefficient": float(surface_drag_coefficient),
            "lift_coefficient": float(surface_lift_coefficient),
            "pressure_drag_coefficient": float(pressure_drag_coefficient),
            "viscous_drag_coefficient": float(viscous_drag_coefficient),
            "pressure_lift_coefficient": float(pressure_lift_coefficient),
            "viscous_lift_coefficient": float(viscous_lift_coefficient),
            "dfg_reduced_drag_coefficient": float(reduced_drag_coefficient),
            "dfg_reduced_lift_coefficient": float(reduced_lift_coefficient),
            "full_minus_dfg_reduced_drag_coefficient": float(
                surface_drag_coefficient - reduced_drag_coefficient
            ),
            "full_minus_dfg_reduced_lift_coefficient": float(
                surface_lift_coefficient - reduced_lift_coefficient
            ),
            "drag_quadrature_absolute_difference": float(
                jnp.abs(
                    2.0 * traction.force_x / force_scale
                    - surface_drag_coefficient
                )
            ),
            "lift_quadrature_absolute_difference": float(
                jnp.abs(
                    2.0 * traction.force_y / force_scale
                    - surface_lift_coefficient
                )
            ),
            "maximum_tangential_surface_slip": float(
                jnp.max(jnp.abs(refined_trace.tangential_surface_velocity))
            ),
            "maximum_normal_surface_slip": float(
                jnp.max(jnp.abs(refined_trace.normal_surface_velocity))
            ),
            "tangential_surface_slip_rms": float(
                jnp.sqrt(
                    jnp.mean(refined_trace.tangential_surface_velocity**2)
                )
            ),
            "normal_surface_slip_rms": float(
                jnp.sqrt(jnp.mean(refined_trace.normal_surface_velocity**2))
            ),
            "pressure_fit_rms": float(jnp.sqrt(jnp.mean(refined_trace.pressure_fit_rms**2))),
            "tangential_fit_rms": float(
                jnp.sqrt(jnp.mean(refined_trace.tangential_fit_rms**2))
            ),
            "normal_fit_rms": float(
                jnp.sqrt(jnp.mean(refined_trace.normal_fit_rms**2))
            ),
            "pressure_fit_condition_number": float(
                refined_trace.pressure_fit_condition_number
            ),
            "velocity_fit_condition_number": float(
                refined_trace.velocity_fit_condition_number
            ),
            "pressure_value_weight_l1": float(
                refined_trace.pressure_value_weight_l1
            ),
            "velocity_value_weight_l1": float(
                refined_trace.velocity_value_weight_l1
            ),
            "velocity_derivative_weight_l1_normalized": float(
                refined_trace.velocity_derivative_weight_l1_normalized
            ),
            "minimum_stencil_signed_distance": float(
                refined_trace.minimum_stencil_signed_distance
            ),
            "minimum_interpolation_node_signed_distance": float(
                refined_trace.minimum_interpolation_node_signed_distance
            ),
            "minimum_velocity_alpha_support_signed_distance": float(
                refined_trace.minimum_velocity_alpha_support_signed_distance
            ),
            "required_minimum_stencil_signed_distance": (
                minimum_nominal_pure_fluid_distance
            ),
            "offset_sensitivity": offset_sensitivity,
            "offset_drag_coefficient_span": max(offset_drag_values)
            - min(offset_drag_values),
            "offset_lift_coefficient_span": max(offset_lift_values)
            - min(offset_lift_values),
            "offset_pressure_difference_span": max(offset_pressure_values)
            - min(offset_pressure_values),
            "polynomial_degree_sensitivity": polynomial_sensitivity,
            "polynomial_degree_drag_coefficient_span": max(degree_drag_values)
            - min(degree_drag_values),
            "polynomial_degree_lift_coefficient_span": max(degree_lift_values)
            - min(degree_lift_values),
            "polynomial_degree_pressure_difference_span": max(
                degree_pressure_values
            )
            - min(degree_pressure_values),
        },
        "numerical": {
            "brinkman_force_x": float(brinkman_force_x),
            "brinkman_force_y": float(brinkman_force_y),
            "brinkman_drag_coefficient": float(brinkman_drag_coefficient),
            "brinkman_lift_coefficient": float(brinkman_lift_coefficient),
            "surface_drag_coefficient": float(surface_drag_coefficient),
            "surface_lift_coefficient": float(surface_lift_coefficient),
            "dfg_reduced_drag_coefficient": float(reduced_drag_coefficient),
            "dfg_reduced_lift_coefficient": float(reduced_lift_coefficient),
            "pressure_front": float(pressure_front),
            "pressure_rear": float(pressure_rear),
            "pressure_difference": float(pressure_difference),
            "recirculation_length_from_cylinder_rear": recirculation,
            "discrete_inlet_mean_velocity": float(discrete_inlet_mean),
        },
        "original_interval_membership_development_only": {
            "brinkman_drag_coefficient": in_original_interval(
                "drag_coefficient", float(brinkman_drag_coefficient)
            ),
            "brinkman_lift_coefficient": in_original_interval(
                "lift_coefficient", float(brinkman_lift_coefficient)
            ),
            "surface_drag_coefficient": in_original_interval(
                "drag_coefficient", float(surface_drag_coefficient)
            ),
            "surface_lift_coefficient": in_original_interval(
                "lift_coefficient", float(surface_lift_coefficient)
            ),
            "dfg_reduced_drag_coefficient": in_original_interval(
                "drag_coefficient", float(reduced_drag_coefficient)
            ),
            "dfg_reduced_lift_coefficient": in_original_interval(
                "lift_coefficient", float(reduced_lift_coefficient)
            ),
            "pressure_difference": in_original_interval(
                "pressure_difference", float(pressure_difference)
            ),
            "recirculation_length": in_original_interval(
                "recirculation_length", recirculation
            ),
        },
        "relative_errors": {
            "surface_drag_coefficient": float(
                jnp.abs(
                    (surface_drag_coefficient - reference_drag) / reference_drag
                )
            ),
            "surface_lift_coefficient": float(
                jnp.abs(
                    (surface_lift_coefficient - reference_lift) / reference_lift
                )
            ),
            "dfg_reduced_drag_coefficient": float(
                jnp.abs((reduced_drag_coefficient - reference_drag) / reference_drag)
            ),
            "dfg_reduced_lift_coefficient": float(
                jnp.abs((reduced_lift_coefficient - reference_lift) / reference_lift)
            ),
            "brinkman_drag_coefficient": float(
                jnp.abs(
                    (brinkman_drag_coefficient - reference_drag) / reference_drag
                )
            ),
            "brinkman_lift_coefficient": float(
                jnp.abs(
                    (brinkman_lift_coefficient - reference_lift) / reference_lift
                )
            ),
            "pressure_difference": float(
                jnp.abs(
                    (pressure_difference - reference_pressure) / reference_pressure
                )
            ),
        },
        "conservation": {
            "divergence_rms": float(div_rms),
            "divergence_max": float(div_max),
            "inlet_flux": float(inlet_flux),
            "outlet_flux": float(outlet_flux),
            "relative_mass_flux_mismatch": float(mismatch),
        },
        "convergence": {
            "converged": bool(diagnostics.converged),
            "iterations": int(diagnostics.iterations),
            "initial_residual_rms": float(diagnostics.initial_residual_rms),
            "final_residual_rms": float(diagnostics.final_residual_rms),
            "final_residual_max": float(diagnostics.final_residual_max),
            "chunked_solve": chunk_report,
        },
    }

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    history_path = output_dir / "residual_history.csv"
    _write_history(history_path, diagnostics)
    surface_path = output_dir / "surface_samples.csv"
    _write_surface_samples(
        surface_path,
        refined_trace,
        refined_traction,
        refined_reduced_traction,
        centre_x,
        centre_y,
        radius,
    )
    fields_path = output_dir / "fields.npz"
    u_cell, v_cell = cell_center_velocity(state)
    np.savez_compressed(
        fields_path,
        x=np.asarray(x),
        y=np.asarray(y),
        u=np.asarray(u_cell),
        v=np.asarray(v_cell),
        u_staggered=np.asarray(state.u),
        v_staggered=np.asarray(state.v),
        p=np.asarray(state.p),
        # Legacy cell diagnostic retained; native arrays are authoritative for
        # reconstructing the candidate residual/force, as stated in the config.
        alpha=np.asarray(cell_alpha),
        alpha_u=np.asarray(interpolate_alpha_to_faces(alpha, grid)[0]),
        alpha_v=np.asarray(interpolate_alpha_to_faces(alpha, grid)[1]),
        # ``phi`` remains a legacy alias for the nominal geometry field.
        phi=np.asarray(nominal_phi),
        nominal_phi=np.asarray(nominal_phi),
        mask_phi=np.asarray(mask_phi),
    )
    stdout_path = output_dir / "stdout.log"
    stdout_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stderr_path = output_dir / "stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    peak_host, peak_accelerator, memory_method = peak_memory_record()
    ended_utc = datetime.now(timezone.utc)
    converged = bool(diagnostics.converged)
    stability_pass = stability["all_within_configured_limits"] is not False
    accepted = converged and stability_pass
    status = "success" if accepted else "partial"
    exit_code = 0 if accepted else 1
    run_failure_reason = None
    if not converged:
        run_failure_reason = "nonlinear tolerance not met"
    elif not stability_pass:
        run_failure_reason = "pre-registered stability limit exceeded"
    result_record = {
        "schema_version": "1.0",
        "experiment_id": "dfg_cylinder_2d1_forward_verification",
        "run_id": str(config["run_id"]),
        "git": environment["git"],
        "command": {
            "argv": [sys.executable, *sys.argv],
            "working_directory": os.getcwd(),
        },
        "timestamps": {
            "start_utc": started_utc.isoformat(),
            "end_utc": ended_utc.isoformat(),
            "status": status,
            "exit_code": exit_code,
        },
        "configuration": {
            **config,
            "requires_gradient": False,
            "reference_comparison": {
                "boundary_conditions_exact": False,
                "outer_boundary_conditions_exact": True,
                "immersed_no_slip_exact": False,
                "quantity_definition_exact": False,
                "paper_eligible": False,
                "reason": (
                    "The outer steady finite-volume flux matches the DFG "
                    "do-nothing form, but finite epsilon and alpha do not impose "
                    "exact sharp-cylinder no slip. The full finite-field stress "
                    "and the reduced DFG no-slip-limit estimator are both "
                    "reported; their equality and the trace reconstruction still "
                    "require joint grid, interface, and penalisation convergence."
                ),
            },
            "primary_acceptance_observables": PRIMARY_ACCEPTANCE_OBSERVABLES,
            "run_acceptance": {
                "accepted": accepted,
                "solver_converged": converged,
                "stability_limits_passed": stability_pass,
                "failure_reason": run_failure_reason,
            },
            "force_measurement": (
                "Full finite-Brinkman nominal-circle stress and reduced DFG "
                "no-slip-limit stress from full-dependency pure-fluid stencil "
                "traces; Brinkman volume reaction and nested native-MAC finite-"
                "Brinkman discrete momentum balances retained as separate "
                "diagnostics"
            ),
            "outflow_discretization": (
                "zero total traction in the combined steady finite-volume flux"
            ),
        },
        "hardware": environment["hardware"],
        "software": environment["software"],
        "primal": {
            "converged": converged,
            "iterations": int(diagnostics.iterations),
            "residual_norm": float(diagnostics.final_residual_rms),
            "divergence_rms": float(div_rms),
            "divergence_max": float(div_max),
            "mass_flux_mismatch": float(mismatch),
            "failure_reason": (
                None if converged else "nonlinear tolerance not met"
            ),
        },
        "objective": {
            "name": "relative_error_of_brinkman_drag_against_dfg_surface_drag",
            "value": metrics["relative_errors"]["brinkman_drag_coefficient"],
            "units": "dimensionless",
            "normalization": "DFG reference C_D=5.57953523384",
        },
        "gradient": no_gradient_record(
            "Canonical forward verification intentionally has no design gradient."
        ),
        "timing": {
            "compile_seconds": compile_seconds,
            "forward_samples_seconds": forward_samples,
            "gradient_samples_seconds": [],
            "optimizer_step_samples_seconds": [],
        },
        "memory": {
            "peak_host_bytes": peak_host,
            "peak_accelerator_bytes": peak_accelerator,
            "measurement_method": memory_method,
        },
        "artifacts": [],
        "notes": (
            "Issue #5 DFG 2D-1 immersed-Brinkman comparison. The 1996 broad "
            "benchmark and the maintained Nabh do-nothing high-precision variant "
            "are distinguished explicitly. Pure-fluid trace-stencil checks, full "
            "finite-field traction, and the reduced no-slip-limit estimator are "
            "recorded separately. Three nested finite-Brinkman discrete momentum "
            "balances audit algebraic and residual closure but are not sharp "
            "traction estimates. Grid, epsilon/h, alpha_max, trace, and surface-"
            "slip convergence are required before paper use."
        ),
    }
    artifact_specs = (
        (resolved_config_path, "application/json", "config"),
        (environment_path, "application/json", "environment"),
        (metrics_path, "application/json", "table"),
        (history_path, "text/csv", "residual-history"),
        (surface_path, "text/csv", "table"),
        (fields_path, "application/x-npz", "field"),
        (stdout_path, "text/plain", "log"),
        (stderr_path, "text/plain", "log"),
    )
    result_record["artifacts"] = [
        artifact_record(path, output_dir, media_type=media, role=role)
        for path, media, role in artifact_specs
    ]
    result_path = output_dir / "result.json"
    write_json(result_path, result_record)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    print(f"Saved auditable run: {result_path}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
