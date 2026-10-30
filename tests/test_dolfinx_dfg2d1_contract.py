"""Host-side contract checks for the container-only DOLFINx reference."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.validate_experiment_result import (
    DEFAULT_SCHEMA,
    ValidationProblem,
    load_json,
    validate_record,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dolfinx_dfg2d1.py"


def _module():
    spec = importlib.util.spec_from_file_location("dolfinx_dfg2d1", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["smoke", "coarse", "medium", "reference"])
def test_dolfinx_dfg2d1_configs_pin_geometry_formulation_and_image(name: str) -> None:
    path = ROOT / "configs" / "forward_verification" / f"dolfinx_dfg2d1_{name}.json"
    config = _module()._load_config(path)
    assert config["container"]["digest"] == (
        "sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8"
    )
    assert config["finite_elements"]["velocity_degree"] == 2
    assert config["finite_elements"]["pressure_degree"] == 1
    assert config["boundary_conditions"]["outlet"] == "natural-zero-traction"
    assert config["force_measurement"]["stress_tensor"] == "nu-grad-u-minus-pI"
    assert config["reference"]["paper_eligible"] is False


def test_dolfinx_dfg2d1_config_rejects_unpinned_outlet(tmp_path: Path) -> None:
    source = ROOT / "configs" / "forward_verification" / "dolfinx_dfg2d1_smoke.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["boundary_conditions"]["outlet"] = "zero-pressure"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary_conditions"):
        _module()._load_config(bad)


def test_dolfinx_force_quadrature_is_attached_to_ufl_measure() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert 'metadata={"quadrature_degree": quadrature_degree}' in source
    assert "form_compiler_options=compiler_options" not in source


def test_dolfinx_reference_study_has_three_refinement_levels() -> None:
    configs = []
    for name in ("coarse", "medium", "reference"):
        path = (
            ROOT
            / "configs"
            / "forward_verification"
            / f"dolfinx_dfg2d1_{name}.json"
        )
        configs.append(_module()._load_config(path))
    assert [config["study"]["level"] for config in configs] == [
        "coarse",
        "medium",
        "fine",
    ]
    assert [config["mesh"]["cylinder_size"] for config in configs] == [
        0.008,
        0.004,
        0.002,
    ]


def test_dolfinx_source_snapshot_detects_implicit_input_change(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path = (
        ROOT
        / "configs"
        / "forward_verification"
        / "dolfinx_dfg2d1_coarse.json"
    )
    implicit_options = tmp_path / "ffcx_options.json"
    before = module._git_snapshot(
        (TOOL, config_path), observed_paths=(implicit_options,)
    )
    implicit_options.write_text('{"scalar_type": "float64"}\n', encoding="utf-8")
    after = module._git_snapshot(
        (TOOL, config_path), observed_paths=(implicit_options,)
    )
    assert before != after
    assert before["input_file_sha256"][str(implicit_options)] is None
    assert len(after["input_file_sha256"][str(implicit_options)]) == 64


def test_dolfinx_bundle_obeys_experiment_result_contract(tmp_path: Path) -> None:
    module = _module()
    config_path = (
        ROOT
        / "configs"
        / "forward_verification"
        / "dolfinx_dfg2d1_coarse.json"
    )
    config = module._load_config(config_path)
    solver_result = {
        "case": config["case"],
        "run_id": config["run_id"],
        "formulation": {},
        "physical_tags": {},
        "mesh": {},
        "metrics": {
            "drag_coefficient": 5.58,
            "lift_coefficient": 0.0106,
            "pressure_difference": 0.1175,
        },
        "reference": config["reference"],
        "original_interval_membership": {
            "drag_coefficient": True,
            "lift_coefficient": True,
            "pressure_difference": True,
        },
        "relative_errors": {"drag_coefficient": 1.0e-4},
        "conservation": {
            "relative_inlet_outlet_mismatch": 2.0e-12,
            "divergence_rms": 3.0e-5,
            "divergence_max_at_dg1_nodes": 8.0e-5,
        },
        "nonlinear_convergence": {
            "converged": True,
            "iterations": 4,
            "final_residual_l2": 2.0e-13,
            "history": [
                {"iteration": 0, "residual_l2": 1.0e-3},
                {"iteration": 4, "residual_l2": 2.0e-13},
            ],
        },
        "timing_seconds": {"nonlinear_solve": 0.25},
        "environment": {
            "hostname": "test-host",
            "platform": "test-platform",
            "python": "3.12.0",
            "numpy": "2.0.0",
            "git_commit": "0" * 40,
            "git_branch": "test",
            "git_repository": "https://example.test/repo.git",
            "git_dirty": False,
            "reproducibility_environment": {},
        },
    }
    output_dir = tmp_path / "bundle"
    result_path = module._write_experiment_bundle(
        output_dir,
        config_path,
        config,
        solver_result,
        started_utc="2026-09-02T10:00:00+00:00",
        ended_utc="2026-09-02T10:00:01+00:00",
    )
    validate_record(
        load_json(result_path),
        schema=load_json(DEFAULT_SCHEMA),
        result_path=result_path,
        verify_artifacts=True,
        verify_git_ancestry=False,
        paper_eligible=False,
    )
    with pytest.raises(
        ValidationProblem, match="reference comparison is explicitly ineligible"
    ):
        validate_record(
            load_json(result_path),
            schema=load_json(DEFAULT_SCHEMA),
            result_path=result_path,
            verify_artifacts=True,
            verify_git_ancestry=False,
            paper_eligible=True,
        )
