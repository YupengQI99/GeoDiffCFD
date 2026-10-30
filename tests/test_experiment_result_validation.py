from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_experiment_result.py"
SPEC = importlib.util.spec_from_file_location("validate_experiment_result", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator_module)

SCHEMA = validator_module.load_json(ROOT / "schemas" / "experiment-result-v1.json")
FIXTURE = (
    ROOT
    / "runs"
    / "example_cpu_smoke"
    / "83dbe38"
    / "ci-reference"
    / "result.json"
)


def _validate(record: dict) -> None:
    validator_module.validate_record(
        record,
        schema=SCHEMA,
        result_path=FIXTURE,
        verify_artifacts=False,
        verify_git_ancestry=False,
        paper_eligible=False,
    )


def test_reference_result_and_artifacts_pass() -> None:
    record = validator_module.load_json(FIXTURE)
    validator_module.validate_record(
        record,
        schema=SCHEMA,
        result_path=FIXTURE,
        verify_artifacts=True,
        verify_git_ancestry=False,
        paper_eligible=False,
    )


def test_gradient_norm_mismatch_is_rejected() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["gradient"]["norm"] *= 1.2
    with pytest.raises(validator_module.ValidationProblem, match="Euclidean norm"):
        _validate(record)


def test_successful_nonconverged_run_is_rejected() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["primal"]["converged"] = False
    record["primal"]["failure_reason"] = "synthetic failure"
    with pytest.raises(validator_module.ValidationProblem, match="converged=true"):
        _validate(record)


def test_taylor_array_length_mismatch_is_rejected() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["gradient"]["taylor_test"]["first_order_rates"].append(2.0)
    with pytest.raises(validator_module.ValidationProblem, match="rate count"):
        _validate(record)


def test_nonfinite_number_is_rejected() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["objective"]["value"] = math.inf
    with pytest.raises(validator_module.ValidationProblem, match="not finite"):
        _validate(record)


def test_artifact_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    record["artifacts"] = [record["artifacts"][0]]
    record["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(validator_module.ValidationProblem, match="checksum mismatch"):
        validator_module.validate_record(
            record,
            schema=SCHEMA,
            result_path=tmp_path / "result.json",
            verify_artifacts=True,
            verify_git_ancestry=False,
            paper_eligible=False,
        )


def test_forward_only_paper_result_is_explicitly_supported() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["configuration"]["requires_gradient"] = False
    record["gradient"] = {
        "success": False,
        "backend": "none",
        "parameterization": None,
        "parameter_names": [],
        "values": [],
        "norm": None,
        "adjoint_residual": None,
        "message": "Canonical forward verification has no design gradient.",
        "taylor_test": None,
    }
    record["timing"]["gradient_samples_seconds"] = []
    validator_module.validate_record(
        record,
        schema=SCHEMA,
        result_path=FIXTURE,
        verify_artifacts=False,
        verify_git_ancestry=False,
        paper_eligible=True,
    )


def test_forward_only_paper_result_requires_explicit_opt_out() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["gradient"] = {
        "success": False,
        "backend": "none",
        "parameterization": None,
        "parameter_names": [],
        "values": [],
        "norm": None,
        "adjoint_residual": None,
        "message": "No gradient.",
        "taylor_test": None,
    }
    with pytest.raises(
        validator_module.ValidationProblem, match="successful gradient required"
    ):
        validator_module.validate_record(
            record,
            schema=SCHEMA,
            result_path=FIXTURE,
            verify_artifacts=False,
            verify_git_ancestry=False,
            paper_eligible=True,
        )


def test_explicit_reference_mismatch_is_not_paper_eligible() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["configuration"]["reference_comparison"] = {
        "boundary_conditions_exact": False,
        "quantity_definition_exact": False,
        "paper_eligible": False,
        "reason": "Development comparison only.",
    }
    with pytest.raises(
        validator_module.ValidationProblem,
        match="reference comparison is explicitly ineligible",
    ):
        validator_module.validate_record(
            record,
            schema=SCHEMA,
            result_path=FIXTURE,
            verify_artifacts=False,
            verify_git_ancestry=False,
            paper_eligible=True,
        )


def test_paper_flag_cannot_override_inexact_reference_definition() -> None:
    record = copy.deepcopy(validator_module.load_json(FIXTURE))
    record["configuration"]["requires_gradient"] = False
    record["configuration"]["reference_comparison"] = {
        "boundary_conditions_exact": False,
        "quantity_definition_exact": True,
        "paper_eligible": True,
        "reason": "A finite immersed boundary is not exact sharp no slip.",
    }
    with pytest.raises(
        validator_module.ValidationProblem,
        match="reference boundary conditions are not exact",
    ):
        validator_module.validate_record(
            record,
            schema=SCHEMA,
            result_path=FIXTURE,
            verify_artifacts=False,
            verify_git_ancestry=False,
            paper_eligible=True,
        )

    record["configuration"]["reference_comparison"] = {
        "boundary_conditions_exact": True,
        "quantity_definition_exact": False,
        "paper_eligible": True,
        "reason": "The observable differs from the cited reference.",
    }
    with pytest.raises(
        validator_module.ValidationProblem,
        match="reference quantity definition is not exact",
    ):
        validator_module.validate_record(
            record,
            schema=SCHEMA,
            result_path=FIXTURE,
            verify_artifacts=False,
            verify_git_ancestry=False,
            paper_eligible=True,
        )
