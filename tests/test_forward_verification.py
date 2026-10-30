from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_cavity import (
    CavityWalls,
    apply_cavity_normal_boundaries,
    cavity_flux_diagnostics,
    cavity_laplacian_u,
    cavity_weighted_pressure_operator,
    create_cavity_initial_state,
    project_cavity_velocity,
)
from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    ChannelState,
    cell_centres,
    create_channel_grid,
    create_initial_state,
    interpolate_alpha_to_faces,
    laplacian_u,
    laplacian_v,
    parabolic_inlet,
    pressure_gradient,
    solve_steady,
)
from jax_cad.verification import (
    MACStaggeredControlVolume,
    bilinear_cell_value,
    brinkman_force,
    cell_center_velocity,
    dfg_circle_exterior_trace,
    error_norms,
    integrate_dfg_circle_traction,
    linear_slope,
    mac_staggered_control_volume_momentum_balance,
    observed_orders,
    parabolic_inlet_face_averages,
    poiseuille_pressure_gradient,
    poiseuille_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CYLINDER_EXAMPLE = ROOT / "examples" / "30_cylinder_dfg2d1.py"
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
SCHEMA_1P2_STABILITY_LIMITS = {
    "c_adv_max": 0.75,
    "c_diff_max": 0.200000000001,
    "alpha_max_times_dt_max": 0.2,
}


def _cylinder_example_module():
    spec = importlib.util.spec_from_file_location(
        "cylinder_dfg2d1_example", CYLINDER_EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_studies_define_three_grids_and_cylinder_sweeps() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "forward_verification"
    configs = []
    for path in sorted(config_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("study") and str(config.get("schema_version", "")).startswith(
            "forward-verification-config-"
        ):
            configs.append(config)

    expected_grids = {
        "planar-poiseuille": [(24, 12), (48, 24), (96, 48)],
        "lid-driven-cavity-re100": [(16, 16), (32, 32), (64, 64)],
        "dfg-cylinder-2d1-re20": [(86, 16), (172, 32), (344, 64)],
    }
    for case, expected in expected_grids.items():
        actual = sorted(
            (config["grid"]["nx"], config["grid"]["ny"])
            for config in configs
            if config["case"] == case
            and config["study"]["name"] == "grid-refinement"
        )
        assert actual == sorted(expected)

    epsilon_values = sorted(
        config["study"]["value"]
        for config in configs
        if config["study"]["name"] == "epsilon-over-h"
    )
    alpha_values = sorted(
        config["study"]["value"]
        for config in configs
        if config["study"]["name"] == "alpha-max"
    )
    # The medium-grid baseline supplies epsilon/h=1.5 and alpha_max=2000.
    assert epsilon_values == [1.0, 2.0]
    assert alpha_values == [500.0, 8000.0]
    for config in configs:
        if config["study"].get("development_only") is True:
            assert config["timing_samples"] == 1
            assert config["study"]["development_only"] is True
        else:
            assert config["timing_samples"] >= 3
        if config["case"] == "dfg-cylinder-2d1-re20":
            assert config["schema_version"] in {
                "forward-verification-config-1.1",
                "forward-verification-config-1.2",
            }
            assert (
                config["initial_condition"]
                == "zero-interior-with-inlet-boundary"
            )
            assert (
                config["boundary_conditions"]["outlet"]
                == "zero-total-traction-finite-volume"
            )
            measurement = config["surface_measurement"]
            assert measurement["stress_tensor"] == "nu-grad-u-minus-pI"
            assert measurement["normal"] == "cylinder-outward-into-fluid"
            assert (
                measurement["trace_method"]
                == "pure-fluid-stencil-radial-polynomial"
            )
            dx = config["grid"]["lx"] / config["grid"]["nx"]
            dy = config["grid"]["ly"] / config["grid"]["ny"]
            h = max(dx, dy)
            # A sampled u/v node depends on adjacent cell-centred alpha values,
            # so the pure-fluid guard needs another half-cell beyond the
            # bilinear interpolation footprint.
            stencil_margin = (np.hypot(dx, dy) + 0.5 * max(dx, dy)) / h
            mask_shift = config["immersed_boundary"].get(
                "mask_midpoint_shift_over_h", 0.0
            )
            pure_fluid_distance_over_h = max(
                0.0,
                config["immersed_boundary"]["epsilon_over_h"] + mask_shift,
            )
            assert min(measurement["offsets_over_h"]) >= (
                pure_fluid_distance_over_h
                + stencil_margin
            )
            assert len(measurement["offsets_over_h"]) >= 5
            assert measurement["offset_sensitivity_shifts_over_h"][0] == 0.0
            centre_x, centre_y = config["physics"]["cylinder_centre"]
            outer_radius = config["physics"]["cylinder_radius"] + h * (
                max(measurement["offsets_over_h"])
                + max(measurement["offset_sensitivity_shifts_over_h"])
            )
            assert centre_x - outer_radius >= 0.5 * dx
            assert centre_x + outer_radius <= config["grid"]["lx"] - 0.5 * dx
            assert centre_y - outer_radius >= 0.5 * dy
            assert centre_y + outer_radius <= config["grid"]["ly"] - 0.5 * dy
            assert measurement["pressure_polynomial_degree"] == 1
            assert measurement["tangential_velocity_polynomial_degree"] == 1
            assert measurement["polynomial_sensitivity_degrees"] == [1, 2]

    schema_1p2_configs = [
        config
        for config in configs
        if config["case"] == "dfg-cylinder-2d1-re20"
        and config["schema_version"] == "forward-verification-config-1.2"
        and config["study"]["name"] != "cylinder-iteration-timestep-isolation-2026-09-18"
    ]
    assert len(schema_1p2_configs) == 6
    common_schema_1p2_measurement = schema_1p2_configs[0][
        "surface_measurement"
    ]
    for config in schema_1p2_configs:
        assert config["acceptance_observables"] == (
            PRIMARY_ACCEPTANCE_OBSERVABLES
        )
        assert config["surface_measurement"] == common_schema_1p2_measurement
        calibration = config["immersed_boundary"][
            "reference_free_calibration"
        ]
        assert calibration["far_field_fit_cells"] == 16
        assert calibration["cell_centre_phase_over_h"] == 0.5
        assert config["solver"]["stability_limits"] == (
            SCHEMA_1P2_STABILITY_LIMITS
        )

        dx = config["grid"]["lx"] / config["grid"]["nx"]
        dy = config["grid"]["ly"] / config["grid"]["ny"]
        dt = config["solver"]["dt"]
        viscosity = config["physics"]["viscosity"]
        c_diff = viscosity * dt * (1.0 / dx**2 + 1.0 / dy**2)
        alpha_dt = config["immersed_boundary"]["alpha_max"] * dt
        assert c_diff == pytest.approx(0.2)
        if config["immersed_boundary"]["delta_b_over_h"] == 1.0:
            assert alpha_dt == pytest.approx(0.1)
        else:
            assert config["immersed_boundary"]["delta_b_over_h"] == 0.75
            assert alpha_dt == pytest.approx(0.17777777777777778)

    pilots = {
        config["study"]["level"]: config
        for config in configs
        if config["study"]["name"] == "mask-shift-pilot"
    }
    assert set(pilots) == {"A-unshifted", "B-calibrated"}
    for pilot in pilots.values():
        assert (pilot["grid"]["nx"], pilot["grid"]["ny"]) == (440, 82)
        assert pilot["solver"]["dt"] == 0.0025
        assert pilot["solver"]["max_steps"] == 12000
        immersed = pilot["immersed_boundary"]
        assert immersed["delta_b_over_h"] == 1.0
        assert immersed["alpha_max"] == 40.0
        assert immersed["alpha_max_h_squared_over_nu"] == 1.0
        assert immersed["epsilon_over_h"] == 1.5
        assert (
            immersed["reference_free_calibration"]["sampling"]
            == "tangential-cell"
        )
        assert pilot["phase_lock"] == {
            "grid_spacing": "dx=dy=h",
            "cylinder_centre_over_h": [40.0, 40.0],
            "cylinder_radius_over_h": 10.0,
        }
    assert pilots["A-unshifted"]["immersed_boundary"][
        "mask_midpoint_shift_over_h"
    ] == 0.0
    assert pilots["B-calibrated"]["immersed_boundary"][
        "mask_midpoint_shift_over_h"
    ] == pytest.approx(0.7809903339948505, abs=0.0)
    assert min(
        pilots["B-calibrated"]["surface_measurement"]["offsets_over_h"]
    ) >= 4.5
    assert (
        pilots["A-unshifted"]["surface_measurement"]
        == pilots["B-calibrated"]["surface_measurement"]
    )

    cylinder_module = _cylinder_example_module()
    for path in sorted(config_dir.glob("cylinder_dfg2d1_*.json")):
        cylinder_module._load_config(path)


def test_cylinder_iteration_timestep_diagnostics_change_only_solver_controls() -> None:
    config_dir = ROOT / "configs" / "forward_verification"
    module = _cylinder_example_module()
    base = module._load_config(config_dir / "cylinder_dfg2d1_development_joint_q1_medium.json")
    for tag, ratio in (("tight", 1), ("tight_dt_half", 0.5)):
        config = module._load_config(config_dir / f"cylinder_dfg2d1_diagnostic_medium_{tag}.json")
        assert set(config) == set(base)
        for key in base:
            if key not in ("run_id", "study", "solver"):
                assert config[key] == base[key], key
        assert config["study"]["development_only"] is True
        assert config["study"]["paper_eligible"] is False
        solver = config["solver"]
        assert solver["dt"] == base["solver"]["dt"] * ratio
        assert solver["nonlinear_rtol"] == 1e-8
        assert solver["nonlinear_atol"] == 1e-9
        assert solver["pressure_tolerance"] == 1e-12
        assert solver["pressure_atol"] == 1e-14
        assert solver["max_steps"] * ratio == 18000
        assert solver["nonlinear_min_steps"] * ratio == 100
        for key in ("chunk_steps", "pressure_maxiter", "include_advection", "stability_limits"):
            assert solver[key] == base["solver"][key]


def test_central_cylinder_diagnostics_keep_physics_and_primary_force(tmp_path) -> None:
    directory = ROOT / "configs" / "forward_verification"
    module = _cylinder_example_module()
    for level in ("coarse", "medium"):
        candidate_path = directory / "diagnostics" / f"cylinder_central_{level}.json"
        candidate = module._load_config(candidate_path)
        base_name = (
            "cylinder_dfg2d1_development_joint_q1_coarse.json" if level == "coarse"
            else "cylinder_dfg2d1_diagnostic_medium_tight.json"
        )
        base = module._load_config(directory / base_name)
        for key in base:
            if key not in ("run_id", "study", "solver"):
                assert candidate[key] == base[key], key
        assert candidate["solver"]["advection_scheme"] == "central"
        if level == "medium":
            assert candidate["solver"] == dict(base["solver"], advection_scheme="central")
    invalid = json.loads(candidate_path.read_text())
    invalid["solver"]["advection_scheme"] = "silent-typo"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="advection_scheme"):
        module._load_config(path)
    invalid["solver"]["advection_scheme"] = "central"
    invalid["study"]["paper_eligible"] = True
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        module._load_config(path)


def test_central_q_sensitivity_reuses_preexisting_reference_free_calibration() -> None:
    directory = ROOT / "configs" / "forward_verification"
    module = _cylinder_example_module()
    candidate = module._load_config(directory / "diagnostics/cylinder_central_medium_q0p75.json")
    baseline = module._load_config(directory / "diagnostics/cylinder_central_medium.json")
    original = module._load_config(directory / "cylinder_dfg2d1_development_q0p75_sensitivity_440x82.json")
    assert candidate["immersed_boundary"] == original["immersed_boundary"]
    for key in baseline:
        if key not in ("run_id", "study", "immersed_boundary"):
            assert candidate[key] == baseline[key], key
    assert candidate["study"]["development_only"] is True
    assert candidate["study"]["paper_eligible"] is False


def test_stronger_penalty_diagnostic_keeps_fixed_grid_and_stability_budget() -> None:
    directory = ROOT / "configs/forward_verification/diagnostics"
    module = _cylinder_example_module()
    candidate = module._load_config(directory / "cylinder_central_medium_q0p5.json")
    baseline = module._load_config(directory / "cylinder_central_medium_q0p75.json")
    for key in baseline:
        if key not in ("run_id", "study", "immersed_boundary", "solver"):
            assert candidate[key] == baseline[key], key
    assert candidate["solver"] == dict(baseline["solver"], dt=.00125, max_steps=36000)
    ib = candidate["immersed_boundary"]
    assert ib["alpha_max"] == 160
    assert ib["delta_b_over_h"] == .5
    assert ib["epsilon_over_h"] == baseline["immersed_boundary"]["epsilon_over_h"]
    assert ib["alpha_max"] * candidate["solver"]["dt"] == pytest.approx(.2)
    assert candidate["study"]["development_only"] is True
    assert candidate["study"]["paper_eligible"] is False


def test_thinner_interface_is_not_a_penalty_or_measurement_change() -> None:
    directory = ROOT / "configs/forward_verification/diagnostics"
    module = _cylinder_example_module()
    candidate = module._load_config(directory / "cylinder_central_medium_q0p5_eps0p75.json")
    baseline = module._load_config(directory / "cylinder_central_medium_q0p5.json")
    for key in baseline:
        if key not in ("run_id", "study", "immersed_boundary"):
            assert candidate[key] == baseline[key], key
    ib = candidate["immersed_boundary"]
    for key in baseline["immersed_boundary"]:
        if key not in ("epsilon_over_h", "mask_midpoint_shift_over_h", "reference_free_calibration"):
            assert ib[key] == baseline["immersed_boundary"][key]
    assert ib["epsilon_over_h"] == .5 * baseline["immersed_boundary"]["epsilon_over_h"]
    assert candidate["study"]["development_only"] is True
    assert candidate["study"]["paper_eligible"] is False


def test_fine_fe_source_only_changes_trace_to_match_common_physical_offsets() -> None:
    directory = ROOT / "configs/forward_verification"
    module = _cylinder_example_module()
    candidate = module._load_config(directory / "diagnostics/cylinder_fine_mask_common_trace_fe_source.json")
    baseline = module._load_config(directory / "cylinder_dfg2d1_development_joint_q1_fine.json")
    for key in baseline:
        if key not in ("run_id", "study", "surface_measurement"):
            assert candidate[key] == baseline[key], key
    for key in baseline["surface_measurement"]:
        expected = baseline["surface_measurement"][key]
        if key in ("offsets_over_h", "offset_sensitivity_shifts_over_h"):
            expected = [2 * value for value in expected]
        assert candidate["surface_measurement"][key] == expected
    from tools.guard_cylinder_run import check_scope
    with pytest.raises(ValueError, match="fine grids"):
        check_scope(candidate, True)


def test_development_joint_refinement_and_q0p75_sensitivity_contracts() -> None:
    config_dir = ROOT / "configs" / "forward_verification"

    joint_paths = sorted(
        config_dir.glob("cylinder_dfg2d1_development_joint_q1_*.json")
    )
    joint = {}
    for path in joint_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        joint[config["study"]["level"]] = config

    expected = {
        "coarse": {
            "grid": (220, 41),
            "phase": ([20.0, 20.0], 5.0),
            "alpha_max": 10.0,
            "dt": 0.01,
            "max_steps": 4000,
        },
        "medium": {
            "grid": (440, 82),
            "phase": ([40.0, 40.0], 10.0),
            "alpha_max": 40.0,
            "dt": 0.0025,
            "max_steps": 12000,
        },
        "fine": {
            "grid": (880, 164),
            "phase": ([80.0, 80.0], 20.0),
            "alpha_max": 160.0,
            "dt": 0.000625,
            "max_steps": 36000,
        },
    }
    assert set(joint) == set(expected)

    common_measurement = None
    for level, config in joint.items():
        contract = expected[level]
        assert config["schema_version"] == "forward-verification-config-1.2"
        assert config["study"]["name"] == (
            "phase-locked-joint-refinement-development"
        )
        assert config["study"]["development_only"] is True
        assert config["study"]["paper_eligible"] is False
        assert config["timing_samples"] == 1
        assert (config["grid"]["nx"], config["grid"]["ny"]) == contract[
            "grid"
        ]
        assert config["phase_lock"] == {
            "grid_spacing": "dx=dy=h",
            "cylinder_centre_over_h": contract["phase"][0],
            "cylinder_radius_over_h": contract["phase"][1],
        }

        h = config["grid"]["lx"] / config["grid"]["nx"]
        assert h == pytest.approx(
            config["grid"]["ly"] / config["grid"]["ny"],
            abs=1.0e-15,
        )
        immersed = config["immersed_boundary"]
        assert immersed["delta_b_over_h"] == 1.0
        assert immersed["epsilon_over_h"] == 1.5
        assert immersed["alpha_max"] == contract["alpha_max"]
        assert immersed["alpha_max"] == pytest.approx(
            config["physics"]["viscosity"] / h**2
        )
        assert immersed["alpha_max_h_squared_over_nu"] == 1.0
        assert immersed["mask_midpoint_shift_over_h"] == pytest.approx(
            0.7809903339948505,
            abs=0.0,
        )
        assert immersed["reference_free_calibration"]["application"] == (
            "calibrated-root"
        )

        solver = config["solver"]
        assert solver["dt"] == contract["dt"]
        assert solver["max_steps"] == contract["max_steps"]
        assert solver["stability_limits"] == SCHEMA_1P2_STABILITY_LIMITS
        assert solver["dt"] / h**2 == pytest.approx(100.0)
        assert (
            2.0 * config["physics"]["viscosity"] * solver["dt"] / h**2
        ) == pytest.approx(0.2)
        assert immersed["alpha_max"] * solver["dt"] == pytest.approx(0.1)
        assert config["acceptance_observables"] == (
            PRIMARY_ACCEPTANCE_OBSERVABLES
        )

        measurement = config["surface_measurement"]
        assert measurement["offsets_over_h"] == [4.5, 4.8, 5.1, 5.4, 5.7]
        if common_measurement is None:
            common_measurement = measurement
        else:
            assert measurement == common_measurement

    sensitivity_path = (
        config_dir
        / "cylinder_dfg2d1_development_q0p75_sensitivity_440x82.json"
    )
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    assert sensitivity["schema_version"] == "forward-verification-config-1.2"
    assert sensitivity["study"] == {
        "name": "phase-locked-delta-b-sensitivity-development",
        "level": "q0.75",
        "value": 0.75,
        "development_only": True,
        "paper_eligible": False,
    }
    assert sensitivity["timing_samples"] == 1
    assert (sensitivity["grid"]["nx"], sensitivity["grid"]["ny"]) == (
        440,
        82,
    )
    assert sensitivity["phase_lock"] == {
        "grid_spacing": "dx=dy=h",
        "cylinder_centre_over_h": [40.0, 40.0],
        "cylinder_radius_over_h": 10.0,
    }

    sensitivity_h = sensitivity["grid"]["lx"] / sensitivity["grid"]["nx"]
    sensitivity_immersed = sensitivity["immersed_boundary"]
    assert sensitivity_immersed["delta_b_over_h"] == 0.75
    assert sensitivity_immersed["epsilon_over_h"] == 1.5
    assert sensitivity_immersed["alpha_max"] == pytest.approx(
        sensitivity["physics"]["viscosity"] / (0.75 * sensitivity_h) ** 2
    )
    assert sensitivity_immersed["alpha_max_h_squared_over_nu"] == (
        pytest.approx(1.0 / 0.75**2)
    )
    calibration = sensitivity_immersed["reference_free_calibration"]
    assert sensitivity_immersed["mask_midpoint_shift_over_h"] == pytest.approx(
        0.5006039613799658,
        abs=0.0,
    )
    assert calibration["calibrated_root_over_h"] == pytest.approx(
        0.5006039613799658,
        abs=0.0,
    )
    assert calibration["unshifted_displacement_over_h"] == pytest.approx(
        0.4713726820153748,
        abs=0.0,
    )
    assert calibration["root_displacement_over_h"] == pytest.approx(
        3.201685238213373e-11,
        abs=0.0,
    )
    assert calibration["iterations"] == 35
    assert calibration["application"] == "calibrated-root"
    assert sensitivity["solver"]["dt"] / sensitivity_h**2 == pytest.approx(
        100.0
    )
    assert sensitivity["solver"]["stability_limits"] == (
        SCHEMA_1P2_STABILITY_LIMITS
    )
    assert sensitivity_immersed["alpha_max"] * sensitivity["solver"][
        "dt"
    ] == pytest.approx(0.17777777777777778)
    assert sensitivity["solver"]["max_steps"] == 12000
    assert sensitivity["surface_measurement"] == common_measurement
    assert sensitivity["acceptance_observables"] == (
        PRIMARY_ACCEPTANCE_OBSERVABLES
    )

    cylinder_module = _cylinder_example_module()
    for path in [*joint_paths, sensitivity_path]:
        cylinder_module._load_config(path)


def test_schema_1p2_recomputes_calibration_and_shifted_support(tmp_path) -> None:
    module = _cylinder_example_module()
    source = (
        ROOT
        / "configs"
        / "forward_verification"
        / "cylinder_dfg2d1_mask_shift_pilot_b.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))

    config["immersed_boundary"]["reference_free_calibration"][
        "calibrated_root_over_h"
    ] += 1.0e-4
    invalid_root = tmp_path / "invalid-root.json"
    invalid_root.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed reference-free"):
        module._load_config(invalid_root)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["surface_measurement"]["offsets_over_h"] = [
        3.5,
        3.8,
        4.1,
        4.4,
        4.7,
    ]
    invalid_support = tmp_path / "invalid-support.json"
    invalid_support.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="shifted diffuse-mask support"):
        module._load_config(invalid_support)

    config = json.loads(source.read_text(encoding="utf-8"))
    del config["immersed_boundary"]["reference_free_calibration"][
        "far_field_fit_cells"
    ]
    missing_fit_extent = tmp_path / "missing-fit-extent.json"
    missing_fit_extent.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="far_field_fit_cells"):
        module._load_config(missing_fit_extent)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["immersed_boundary"]["reference_free_calibration"][
        "cell_centre_phase_over_h"
    ] = 0.0
    wrong_calibration_phase = tmp_path / "wrong-calibration-phase.json"
    wrong_calibration_phase.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cell-centre phase"):
        module._load_config(wrong_calibration_phase)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["acceptance_observables"]["drag_coefficient"] = (
        "metrics.numerical.surface_drag_coefficient"
    )
    selectable_estimator = tmp_path / "selectable-estimator.json"
    selectable_estimator.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="acceptance_observables"):
        module._load_config(selectable_estimator)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["solver"]["stability_limits"]["c_diff_max"] = 0.19
    unsafe_diffusion = tmp_path / "unsafe-diffusion.json"
    unsafe_diffusion.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="diffusion number"):
        module._load_config(unsafe_diffusion)

    for invalid_dt in (-0.0025, float("nan")):
        config = json.loads(source.read_text(encoding="utf-8"))
        config["solver"]["dt"] = invalid_dt
        invalid_dt_path = tmp_path / f"invalid-dt-{invalid_dt}.json"
        invalid_dt_path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            module._load_config(invalid_dt_path)

    config = json.loads(source.read_text(encoding="utf-8"))
    config["immersed_boundary"]["reference_free_calibration"][
        "far_field_fit_cells"
    ] = 16.9
    fractional_audit_integer = tmp_path / "fractional-audit-integer.json"
    fractional_audit_integer.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="far_field_fit_cells must be an integer"):
        module._load_config(fractional_audit_integer)


def test_cylinder_runner_rejects_nonfinite_solve_output_before_postprocess() -> None:
    module = _cylinder_example_module()
    finite = ChannelState(
        jnp.zeros((2, 1), dtype=jnp.float64),
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros((1, 1), dtype=jnp.float64),
    )
    module._require_finite_solve_output(finite, finite)
    nonfinite = finite._replace(p=finite.p.at[0, 0].set(jnp.nan))
    with pytest.raises(FloatingPointError, match=r"state\.p\(0, 0\)=.*nan"):
        module._require_finite_solve_output(nonfinite, finite)

    grid = create_channel_grid(4, 3, 2.0, 1.5)
    state = ChannelState(
        jnp.full((5, 3), 2.0, dtype=jnp.float64),
        jnp.full((4, 4), -0.5, dtype=jnp.float64),
        jnp.zeros((4, 3), dtype=jnp.float64),
    )
    params = ChannelParams(viscosity=0.1, dt=0.2)
    stability = module._stability_metrics(state, grid, params, alpha_max=3.0)
    assert stability["max_abs_u"] == 2.0
    assert stability["max_abs_v"] == 0.5
    assert stability["c_adv"] == pytest.approx(1.0)
    assert stability["c_diff"] == pytest.approx(0.16)
    assert stability["alpha_max_times_dt"] == pytest.approx(0.6)
    limited = module._stability_metrics(
        state,
        grid,
        params,
        alpha_max=3.0,
        limits={
            "c_adv_max": 0.9,
            "c_diff_max": 0.2,
            "alpha_max_times_dt_max": 0.7,
        },
    )
    assert limited["within_configured_limits"] == {
        "c_adv": False,
        "c_diff": True,
        "alpha_max_times_dt": True,
    }
    assert limited["all_within_configured_limits"] is False


def test_cylinder_cv_audit_normalizes_small_lift_by_component() -> None:
    module = _cylinder_example_module()
    grid = create_channel_grid(20, 16, 2.0, 1.6)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha = alpha.at[8:11, 6:9].set(5.0)
    support, control_volumes = module._nested_finite_brinkman_control_volumes(
        alpha, grid
    )
    inlet = jnp.zeros(grid.ny, dtype=jnp.float64)
    params = ChannelParams(viscosity=0.1, dt=0.01)

    def audit(v_value: float) -> dict:
        state = ChannelState(
            jnp.zeros((grid.nx + 1, grid.ny), dtype=jnp.float64),
            jnp.full((grid.nx, grid.ny + 1), v_value, dtype=jnp.float64),
            jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64),
        )
        return module._finite_brinkman_discrete_momentum_audit(
            state,
            alpha,
            inlet,
            grid,
            params,
            1.0,
            support,
            control_volumes,
        )

    zero_lift = audit(0.0)
    assert zero_lift["summary"]["global_brinkman_lift_coefficient"] == 0.0
    assert zero_lift["summary"]["normalized_lift_coefficient_spread"] == 0.0
    assert zero_lift["summary"]["vector_force_spread_normalized"] == 0.0

    small_lift = audit(1.0e-16)
    for entry in small_lift["control_volumes"]:
        global_y = entry["brinkman_volume_reaction"]["global_force_y"]
        assert 0.0 < abs(global_y) < 1.0e-12
        expected = abs(
            entry["inferred_brinkman_reaction"]["force_y"]
            - entry["brinkman_volume_reaction"]["captured_force_y"]
        ) / 1.0e-12
        assert entry["relative_cv_volume_closure_by_component"][
            "y"
        ] == pytest.approx(expected)
    assert small_lift["summary"][
        "vector_force_spread_normalized"
    ] == pytest.approx(
        small_lift["summary"]["vector_force_spread"] / 1.0e-12
    )


def test_cavity_pressure_operator_is_symmetric_positive_definite() -> None:
    grid = create_channel_grid(7, 5, 1.4, 1.0)
    alpha = jnp.linspace(
        0.0, 12.0, grid.nx * grid.ny, dtype=jnp.float64
    ).reshape(grid.nx, grid.ny)
    alpha_u, alpha_v = interpolate_alpha_to_faces(alpha, grid)
    mobility_u = 1.0 / (1.0 + 0.01 * alpha_u)
    mobility_v = 1.0 / (1.0 + 0.01 * alpha_v)
    x = jnp.sin(jnp.arange(alpha.size, dtype=jnp.float64)).reshape(alpha.shape)
    y = jnp.cos(
        0.7 * jnp.arange(alpha.size, dtype=jnp.float64)
    ).reshape(alpha.shape)
    ax = cavity_weighted_pressure_operator(x, mobility_u, mobility_v, grid)
    ay = cavity_weighted_pressure_operator(y, mobility_u, mobility_v, grid)
    np.testing.assert_allclose(
        float(jnp.vdot(x, ay)),
        float(jnp.vdot(ax, y)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(jnp.vdot(x, ax)) > 0.0
    np.testing.assert_allclose(
        np.asarray(
            cavity_weighted_pressure_operator(
                jnp.ones_like(x), mobility_u, mobility_v, grid
            )
        ),
        1.0,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cavity_projection_enforces_closed_flux_and_zero_mean_gauge() -> None:
    grid = create_channel_grid(13, 9, 1.3, 1.0)
    ii = jnp.arange((grid.nx + 1) * grid.ny, dtype=jnp.float64).reshape(
        grid.nx + 1, grid.ny
    )
    jj = jnp.arange(grid.nx * (grid.ny + 1), dtype=jnp.float64).reshape(
        grid.nx, grid.ny + 1
    )
    u, v = apply_cavity_normal_boundaries(
        0.2 * jnp.sin(0.17 * ii),
        0.15 * jnp.cos(0.11 * jj),
        grid,
    )
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    mobility_u, mobility_v = interpolate_alpha_to_faces(alpha, grid)
    mobility_u = jnp.ones_like(mobility_u)
    mobility_v = jnp.ones_like(mobility_v)
    params = ChannelParams(
        viscosity=0.1,
        dt=0.01,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
    )
    state, diagnostics = project_cavity_velocity(
        u, v, mobility_u, mobility_v, grid, params
    )
    assert float(diagnostics.divergence_rms) < 2.0e-10
    assert float(diagnostics.divergence_max) < 2.0e-9
    assert abs(float(diagnostics.pressure_mean)) < 1.0e-13
    assert float(diagnostics.boundary_normal_flux_l1) == 0.0
    assert float(diagnostics.net_boundary_flux) == 0.0
    np.testing.assert_allclose(np.asarray(state.u[[0, -1], :]), 0.0)
    np.testing.assert_allclose(np.asarray(state.v[:, [0, -1]]), 0.0)


def test_moving_lid_enters_tangential_viscous_stencil() -> None:
    grid = create_channel_grid(8, 6, 1.0, 1.0)
    state = create_cavity_initial_state(grid, dtype=jnp.float64)
    walls = CavityWalls(top_u=1.0)
    lap = cavity_laplacian_u(state.u, grid, walls)
    expected = 2.0 / (grid.dy * grid.dy)
    np.testing.assert_allclose(np.asarray(lap[1:-1, -1]), expected)
    np.testing.assert_allclose(np.asarray(lap[1:-1, 0]), 0.0)
    np.testing.assert_allclose(np.asarray(lap[[0, -1], :]), 0.0)


def test_verification_norms_orders_and_poiseuille_formula() -> None:
    y = jnp.array([0.125, 0.375, 0.625, 0.875], dtype=jnp.float64)
    reference = poiseuille_profile(y, height=1.0, mean_velocity=0.5)
    numerical = reference + jnp.array([0.01, -0.02, 0.02, -0.01])
    norms = error_norms(numerical, reference)
    np.testing.assert_allclose(float(norms.l2_absolute), np.sqrt(0.00025))
    np.testing.assert_allclose(float(norms.linf_absolute), 0.02)

    orders = observed_orders(
        jnp.array([0.2, 0.1, 0.05]), jnp.array([0.04, 0.01, 0.0025])
    )
    np.testing.assert_allclose(np.asarray(orders), 2.0)
    np.testing.assert_allclose(
        poiseuille_pressure_gradient(0.1, 1.0, 0.5), -0.6
    )

    x = jnp.array([0.5, 1.5, 2.5, 3.5])
    values = 3.0 - 0.6 * x
    np.testing.assert_allclose(float(linear_slope(x, values)), -0.6)


def test_cavity_flux_diagnostics_reject_no_hidden_open_boundary() -> None:
    grid = create_channel_grid(6, 5, 1.2, 1.0)
    state = create_cavity_initial_state(grid, dtype=jnp.float64)
    div_rms, div_max, p_mean, flux_l1, net_flux = cavity_flux_diagnostics(
        state, grid
    )
    assert float(div_rms) == 0.0
    assert float(div_max) == 0.0
    assert float(p_mean) == 0.0
    assert float(flux_l1) == 0.0
    assert float(net_flux) == 0.0


def test_cylinder_measurement_utilities_use_native_staggered_volumes() -> None:
    grid = create_channel_grid(4, 3, 2.0, 1.0)
    u = jnp.full((grid.nx + 1, grid.ny), 2.0, dtype=jnp.float64)
    v = jnp.full((grid.nx, grid.ny + 1), -1.0, dtype=jnp.float64)
    p = jnp.arange(grid.nx * grid.ny, dtype=jnp.float64).reshape(
        grid.nx, grid.ny
    )
    state = ChannelState(u, v, p)
    alpha = jnp.full((grid.nx, grid.ny), 3.0, dtype=jnp.float64)
    force_x, force_y = brinkman_force(state, alpha, grid)
    np.testing.assert_allclose(float(force_x), 12.0)
    np.testing.assert_allclose(float(force_y), -6.0)

    u_cell, v_cell = cell_center_velocity(state)
    np.testing.assert_allclose(np.asarray(u_cell), 2.0)
    np.testing.assert_allclose(np.asarray(v_cell), -1.0)
    centre_value = bilinear_cell_value(p, x=1.0, y=0.5, grid=grid)
    np.testing.assert_allclose(float(centre_value), 5.5)


@pytest.mark.parametrize("scheme", ["donor-cell", "central"])
def test_mac_control_volume_balance_has_exact_random_field_identity(scheme) -> None:
    rng = np.random.default_rng(20260902)
    grid = create_channel_grid(11, 9, 1.7, 1.1)
    state = ChannelState(
        jnp.asarray(rng.normal(size=(grid.nx + 1, grid.ny))),
        jnp.asarray(rng.normal(size=(grid.nx, grid.ny + 1))),
        jnp.asarray(rng.normal(size=(grid.nx, grid.ny))),
    )
    inlet = jnp.asarray(rng.normal(size=grid.ny))
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha = alpha.at[3:7, 3:6].set(
        jnp.asarray(rng.uniform(2.0, 9.0, size=(4, 3)))
    )
    params = ChannelParams(
        viscosity=0.07,
        dt=0.01,
        force_x=0.13,
        force_y=-0.04,
        include_advection=True,
        advection_scheme=scheme,
    )
    control_volume = MACStaggeredControlVolume(
        u_i_start=2,
        u_i_stop=8,
        u_j_start=2,
        u_j_stop=6,
        v_i_start=2,
        v_i_stop=8,
        v_j_start=2,
        v_j_stop=7,
    )
    balance = mac_staggered_control_volume_momentum_balance(
        state, alpha, inlet, grid, params, control_volume
    )

    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_x),
        float(
            balance.captured_brinkman_force_x
            - balance.momentum_residual_integral_x
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_y),
        float(
            balance.captured_brinkman_force_y
            - balance.momentum_residual_integral_y
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_mac_control_volume_bounds_are_inclusive_and_coordinates_are_anchored() -> None:
    grid = create_channel_grid(12, 10, 1.2, 1.0)
    control_volume = MACStaggeredControlVolume(
        u_i_start=2,
        u_i_stop=8,
        u_j_start=2,
        u_j_stop=6,
        v_i_start=2,
        v_i_stop=8,
        v_j_start=2,
        v_j_stop=7,
    )
    u = jnp.zeros((grid.nx + 1, grid.ny), dtype=jnp.float64)
    u = u.at[2, 2].set(1.0).at[8, 6].set(1.0)
    v = jnp.zeros((grid.nx, grid.ny + 1), dtype=jnp.float64)
    v = v.at[2, 2].set(-1.0).at[8, 7].set(-1.0)
    state = ChannelState(
        u,
        v,
        jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64),
    )
    alpha_value = 4.0
    alpha = jnp.full((grid.nx, grid.ny), alpha_value, dtype=jnp.float64)
    inlet = jnp.zeros(grid.ny, dtype=jnp.float64)
    params = ChannelParams(viscosity=0.0, dt=0.01, include_advection=False)

    balance = mac_staggered_control_volume_momentum_balance(
        state, alpha, inlet, grid, params, control_volume
    )
    two_boundary_points = 2.0 * alpha_value * grid.dx * grid.dy
    assert float(balance.captured_brinkman_force_x) == pytest.approx(
        two_boundary_points
    )
    assert float(balance.captured_brinkman_force_y) == pytest.approx(
        -two_boundary_points
    )

    module = _cylinder_example_module()
    boundaries = module._control_volume_physical_boundaries(control_volume, grid)
    assert boundaries["u_momentum_dual_rectangle"] == {
        "x_min": pytest.approx(0.15),
        "x_max": pytest.approx(0.85),
        "y_min": pytest.approx(0.2),
        "y_max": pytest.approx(0.7),
        "coordinate_definition": (
            "x=[(u_i_start-1/2)dx,(u_i_stop+1/2)dx], "
            "y=[u_j_start*dy,(u_j_stop+1)dy]"
        ),
    }
    assert boundaries["v_momentum_dual_rectangle"] == {
        "x_min": pytest.approx(0.2),
        "x_max": pytest.approx(0.9),
        "y_min": pytest.approx(0.15),
        "y_max": pytest.approx(0.75),
        "coordinate_definition": (
            "x=[v_i_start*dx,(v_i_stop+1)dx], "
            "y=[(v_j_start-1/2)dy,(v_j_stop+1/2)dy]"
        ),
    }
    np.testing.assert_allclose(float(balance.algebraic_closure_x), 0.0, atol=2.0e-13)
    np.testing.assert_allclose(float(balance.algebraic_closure_y), 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        float(balance.uncaptured_brinkman_force_x), 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        float(balance.uncaptured_brinkman_force_y), 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_x),
        float(
            balance.convective_contribution_x
            + balance.pressure_contribution_x
            + balance.viscous_contribution_x
            + balance.source_contribution_x
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_y),
        float(
            balance.convective_contribution_y
            + balance.pressure_contribution_y
            + balance.viscous_contribution_y
            + balance.source_contribution_y
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_converged_mac_control_volume_force_matches_brinkman_reaction() -> None:
    grid = create_channel_grid(12, 8, 1.5, 1.0)
    inlet = parabolic_inlet(grid, 0.08, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    x, y = cell_centres(grid, dtype=jnp.float64)
    alpha = 80.0 * ((x > 0.5) & (x < 0.9) & (y > 0.3) & (y < 0.7))
    params = ChannelParams(
        viscosity=0.2,
        dt=0.005,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=1.0e-6,
        nonlinear_atol=1.0e-8,
    )
    state, diagnostics = solve_steady(
        state0, alpha, inlet, grid, params, max_steps=500
    )
    control_volume = MACStaggeredControlVolume(
        u_i_start=2,
        u_i_stop=grid.nx - 2,
        u_j_start=1,
        u_j_stop=grid.ny - 2,
        v_i_start=2,
        v_i_stop=grid.nx - 3,
        v_j_start=1,
        v_j_stop=grid.ny - 1,
    )
    balance = mac_staggered_control_volume_momentum_balance(
        state, alpha, inlet, grid, params, control_volume
    )

    assert bool(diagnostics.converged)
    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_x),
        float(balance.global_brinkman_force_x),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        float(balance.inferred_brinkman_reaction_y),
        float(balance.global_brinkman_force_y),
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(float(balance.algebraic_closure_x), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(float(balance.algebraic_closure_y), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        float(balance.uncaptured_brinkman_force_x), 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        float(balance.uncaptured_brinkman_force_y), 0.0, atol=1.0e-12
    )


def test_cylinder_runner_builds_three_nested_strict_support_control_volumes() -> None:
    module = _cylinder_example_module()
    grid = create_channel_grid(20, 16, 2.0, 0.8)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha = alpha.at[6:9, 5:8].set(100.0)

    support, boxes = module._nested_finite_brinkman_control_volumes(alpha, grid)

    assert [margin for margin, _ in boxes] == [1, 2, 3]
    assert support == {
        "u": {
            "i_start": 6,
            "i_stop": 9,
            "j_start": 5,
            "j_stop": 7,
            "nonzero_face_count": 12,
        },
        "v": {
            "i_start": 6,
            "i_stop": 8,
            "j_start": 5,
            "j_stop": 8,
            "nonzero_face_count": 12,
        },
    }
    for margin, box in boxes:
        assert box.u_i_start == support["u"]["i_start"] - margin
        assert box.u_i_stop == support["u"]["i_stop"] + margin
        assert box.u_j_start == support["u"]["j_start"] - margin
        assert box.u_j_stop == support["u"]["j_stop"] + margin
        assert box.v_i_start == support["v"]["i_start"] - margin
        assert box.v_i_stop == support["v"]["i_stop"] + margin
        assert box.v_j_start == support["v"]["j_start"] - margin
        assert box.v_j_stop == support["v"]["j_stop"] + margin


def test_cylinder_runner_rejects_control_volume_that_reaches_domain_boundary() -> None:
    module = _cylinder_example_module()
    grid = create_channel_grid(12, 10, 1.2, 1.0)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha = alpha.at[1:4, 1:4].set(10.0)

    with pytest.raises(ValueError, match="insufficient room.*margin=1"):
        module._nested_finite_brinkman_control_volumes(alpha, grid)


def test_cylinder_runner_serializes_finite_brinkman_discrete_audit_contract() -> None:
    module = _cylinder_example_module()
    rng = np.random.default_rng(20260902)
    grid = create_channel_grid(20, 16, 2.0, 0.8)
    state = ChannelState(
        jnp.asarray(rng.normal(size=(grid.nx + 1, grid.ny))),
        jnp.asarray(rng.normal(size=(grid.nx, grid.ny + 1))),
        jnp.asarray(rng.normal(size=(grid.nx, grid.ny))),
    )
    inlet = jnp.asarray(rng.normal(size=grid.ny))
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    alpha = alpha.at[6:9, 5:8].set(100.0)
    params = ChannelParams(
        viscosity=0.03,
        dt=0.01,
        force_x=0.02,
        force_y=-0.01,
        include_advection=True,
    )
    support, boxes = module._nested_finite_brinkman_control_volumes(alpha, grid)

    audit = module._finite_brinkman_discrete_momentum_audit(
        state,
        alpha,
        inlet,
        grid,
        params,
        force_scale=0.4,
        support=support,
        control_volumes=boxes,
    )

    assert "not sharp interface traction" in audit["classification"]
    assert audit["summary"]["control_volume_count"] == 3
    assert len(audit["control_volumes"]) == 3
    json.dumps(audit, allow_nan=False)
    for entry in audit["control_volumes"]:
        assert entry["captures_all_strict_nonzero_alpha_face_support"] == {
            "u": True,
            "v": True,
            "both": True,
        }
        assert set(entry["indices_inclusive"]) == {"u_momentum", "v_momentum"}
        assert set(entry["physical_boundaries"]) == {
            "u_momentum_dual_rectangle",
            "v_momentum_dual_rectangle",
        }
        assert set(entry["flux_decomposition_body_force_sign"]) == {
            "convective",
            "pressure",
            "viscous",
            "source",
        }
        volume = entry["brinkman_volume_reaction"]
        np.testing.assert_allclose(volume["uncaptured_force_x"], 0.0, atol=1e-12)
        np.testing.assert_allclose(volume["uncaptured_force_y"], 0.0, atol=1e-12)
        np.testing.assert_allclose(entry["algebraic_closure"]["x"], 0.0, atol=1e-12)
        np.testing.assert_allclose(entry["algebraic_closure"]["y"], 0.0, atol=1e-12)


def test_dfg_parabolic_face_averages_have_exact_flux() -> None:
    for ny in (16, 32, 64):
        grid = create_channel_grid(11, ny, 2.2, 0.41)
        inlet = parabolic_inlet_face_averages(
            grid, maximum_velocity=0.3, dtype=jnp.float64
        )
        np.testing.assert_allclose(
            float(jnp.sum(inlet) * grid.dy),
            0.2 * grid.ly,
            rtol=0.0,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(np.asarray(inlet), np.asarray(inlet[::-1]))


def test_split_outlet_terms_form_one_zero_total_traction_flux() -> None:
    grid = create_channel_grid(9, 6, 1.8, 0.7)
    ii = jnp.arange((grid.nx + 1) * grid.ny, dtype=jnp.float64).reshape(
        grid.nx + 1, grid.ny
    )
    u = 0.2 * jnp.sin(0.17 * ii)
    p = jnp.cos(
        0.11 * jnp.arange(grid.nx * grid.ny, dtype=jnp.float64)
    ).reshape(grid.nx, grid.ny)
    viscosity = 0.037
    grad_u, _ = pressure_gradient(p, grid)
    combined = -viscosity * laplacian_u(u, grid)[-1, :] + grad_u[-1, :]
    outlet_lap_y = jnp.zeros(grid.ny, dtype=u.dtype)
    outlet_lap_y = outlet_lap_y.at[1:-1].set(
        (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2])
        / (grid.dy * grid.dy)
    )
    outlet_lap_y = outlet_lap_y.at[0].set(
        (u[-1, 1] - 3.0 * u[-1, 0]) / (grid.dy * grid.dy)
    )
    outlet_lap_y = outlet_lap_y.at[-1].set(
        (u[-1, -2] - 3.0 * u[-1, -1]) / (grid.dy * grid.dy)
    )
    adjacent_total_traction = (
        viscosity * (u[-1, :] - u[-2, :]) / grid.dx - p[-1, :]
    )
    np.testing.assert_allclose(
        np.asarray(combined),
        np.asarray(
            2.0 * adjacent_total_traction / grid.dx
            - viscosity * outlet_lap_y
        ),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_outlet_tangential_velocity_has_zero_viscous_flux() -> None:
    grid = create_channel_grid(9, 6, 1.8, 0.7)
    ii = jnp.arange(grid.nx, dtype=jnp.float64)[:, None]
    jj = jnp.arange(grid.ny + 1, dtype=jnp.float64)[None, :]
    v = 0.13 * jnp.sin(0.21 * ii) + 0.07 * jnp.cos(0.17 * jj)
    lap = laplacian_v(v, grid)
    transverse = jnp.zeros(grid.ny + 1, dtype=v.dtype)
    transverse = transverse.at[1:-1].set(
        (v[-1, 2:] - 2.0 * v[-1, 1:-1] + v[-1, :-2])
        / (grid.dy * grid.dy)
    )
    np.testing.assert_allclose(
        np.asarray(lap[-1, 1:-1] - transverse[1:-1]),
        np.asarray((v[-2, 1:-1] - v[-1, 1:-1]) / (grid.dx * grid.dx)),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_dfg_circle_traction_matches_linear_pressure_identity() -> None:
    count = 128
    radius = 0.3
    theta = 2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / count
    x = 0.7 + radius * jnp.cos(theta)
    y = -0.2 + radius * jnp.sin(theta)
    coefficient_x = 1.4
    coefficient_y = -0.8
    pressure = 2.1 + coefficient_x * x + coefficient_y * y
    traction = integrate_dfg_circle_traction(
        pressure,
        jnp.zeros_like(pressure),
        theta,
        viscosity=0.01,
        radius=radius,
    )
    np.testing.assert_allclose(
        float(traction.force_x),
        -np.pi * radius * radius * coefficient_x,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        float(traction.force_y),
        -np.pi * radius * radius * coefficient_y,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_dfg_circle_viscous_force_uses_published_tangent_convention() -> None:
    count = 128
    radius = 0.23
    viscosity = 0.017
    theta = 2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / count
    derivative = jnp.sin(theta)
    traction = integrate_dfg_circle_traction(
        jnp.zeros_like(theta), derivative, theta, viscosity, radius
    )
    np.testing.assert_allclose(
        float(traction.force_x), viscosity * np.pi * radius, atol=2.0e-15
    )
    np.testing.assert_allclose(float(traction.force_y), 0.0, atol=2.0e-15)


def test_dfg_full_traction_includes_finite_normal_velocity_gradient() -> None:
    count = 128
    radius = 0.23
    viscosity = 0.017
    theta = 2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / count
    normal_derivative = jnp.cos(theta)
    traction = integrate_dfg_circle_traction(
        jnp.zeros_like(theta),
        jnp.zeros_like(theta),
        theta,
        viscosity,
        radius,
        normal_normal_derivative=normal_derivative,
    )
    np.testing.assert_allclose(
        float(traction.force_x), viscosity * np.pi * radius, atol=2.0e-15
    )
    np.testing.assert_allclose(float(traction.force_y), 0.0, atol=2.0e-15)


def test_dfg_exterior_pressure_trace_is_one_sided_and_linear_exact() -> None:
    grid = create_channel_grid(80, 40, 2.0, 1.0)
    u = jnp.zeros((grid.nx + 1, grid.ny), dtype=jnp.float64)
    v = jnp.zeros((grid.nx, grid.ny + 1), dtype=jnp.float64)
    x = (jnp.arange(grid.nx, dtype=jnp.float64) + 0.5) * grid.dx
    y = (jnp.arange(grid.ny, dtype=jnp.float64) + 0.5) * grid.dy
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    p = 1.2 + 0.7 * xx - 0.4 * yy
    trace = dfg_circle_exterior_trace(
        ChannelState(u, v, p),
        grid,
        centre_x=0.7,
        centre_y=0.45,
        radius=0.12,
        h_reference=max(grid.dx, grid.dy),
        offsets_over_h=jnp.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        quadrature_points=64,
        pressure_degree=2,
        tangential_velocity_degree=3,
    )
    expected = (
        1.2
        + 0.7 * (0.7 + 0.12 * jnp.cos(trace.theta))
        - 0.4 * (0.45 + 0.12 * jnp.sin(trace.theta))
    )
    np.testing.assert_allclose(
        np.asarray(trace.pressure), np.asarray(expected), rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(np.asarray(trace.tangential_normal_derivative), 0.0)
    assert float(trace.minimum_stencil_signed_distance) >= 0.0


def test_dfg_exterior_trace_checks_staggered_alpha_dependencies() -> None:
    grid = create_channel_grid(86, 16, 2.2, 0.41)
    state = ChannelState(
        jnp.zeros((grid.nx + 1, grid.ny), dtype=jnp.float64),
        jnp.zeros((grid.nx, grid.ny + 1), dtype=jnp.float64),
        jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64),
    )
    h_reference = max(grid.dx, grid.dy)
    epsilon = 1.5 * h_reference

    # These offsets clear every p/u/v interpolation node, but some u/v nodes
    # still average alpha from cell centres inside the compact diffuse band.
    with pytest.raises(ValueError, match="staggered-alpha dependency"):
        dfg_circle_exterior_trace(
            state,
            grid,
            centre_x=0.2,
            centre_y=0.2,
            radius=0.05,
            h_reference=h_reference,
            offsets_over_h=jnp.array([3.0, 3.4, 3.8, 4.2, 4.6]),
            quadrature_points=64,
            pressure_degree=1,
            tangential_velocity_degree=1,
            minimum_fluid_signed_distance=epsilon,
        )

    trace = dfg_circle_exterior_trace(
        state,
        grid,
        centre_x=0.2,
        centre_y=0.2,
        radius=0.05,
        h_reference=h_reference,
        offsets_over_h=jnp.array([3.5, 3.8, 4.1, 4.4, 4.7]),
        quadrature_points=64,
        pressure_degree=1,
        tangential_velocity_degree=1,
        minimum_fluid_signed_distance=epsilon,
    )
    assert float(trace.minimum_interpolation_node_signed_distance) >= epsilon
    assert float(trace.minimum_velocity_alpha_support_signed_distance) >= epsilon
    assert float(trace.minimum_stencil_signed_distance) >= epsilon


def test_dfg_exterior_velocity_trace_preserves_finite_slip_and_gradient() -> None:
    grid = create_channel_grid(80, 40, 2.0, 1.0)
    x_u = jnp.arange(grid.nx + 1, dtype=jnp.float64) * grid.dx
    y_u = (jnp.arange(grid.ny, dtype=jnp.float64) + 0.5) * grid.dy
    x_v = (jnp.arange(grid.nx, dtype=jnp.float64) + 0.5) * grid.dx
    y_v = jnp.arange(grid.ny + 1, dtype=jnp.float64) * grid.dy
    u = 0.3 + 0.2 * x_u[:, None] - 0.1 * y_u[None, :]
    v = -0.2 + 0.05 * x_v[:, None] + 0.4 * y_v[None, :]
    p = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    trace = dfg_circle_exterior_trace(
        ChannelState(u, v, p),
        grid,
        centre_x=0.7,
        centre_y=0.45,
        radius=0.12,
        h_reference=max(grid.dx, grid.dy),
        offsets_over_h=jnp.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        quadrature_points=64,
        pressure_degree=2,
        tangential_velocity_degree=3,
    )
    nx = jnp.cos(trace.theta)
    ny = jnp.sin(trace.theta)
    surface_x = 0.7 + 0.12 * nx
    surface_y = 0.45 + 0.12 * ny
    surface_u = 0.3 + 0.2 * surface_x - 0.1 * surface_y
    surface_v = -0.2 + 0.05 * surface_x + 0.4 * surface_y
    expected_tangent = surface_u * ny - surface_v * nx
    expected_normal = surface_u * nx + surface_v * ny
    du_dn = 0.2 * nx - 0.1 * ny
    dv_dn = 0.05 * nx + 0.4 * ny
    expected_tangent_derivative = du_dn * ny - dv_dn * nx
    expected_normal_derivative = du_dn * nx + dv_dn * ny
    np.testing.assert_allclose(
        np.asarray(trace.tangential_surface_velocity),
        np.asarray(expected_tangent),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(trace.normal_surface_velocity),
        np.asarray(expected_normal),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(trace.tangential_normal_derivative),
        np.asarray(expected_tangent_derivative),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(trace.normal_normal_derivative),
        np.asarray(expected_normal_derivative),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
