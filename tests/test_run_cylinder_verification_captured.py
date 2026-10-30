from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "tools" / "run_cylinder_verification_captured.py"
VALIDATOR_PATH = ROOT / "tools" / "validate_experiment_result.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wrapper = _load_module("run_cylinder_verification_captured", WRAPPER_PATH)
validator = _load_module("captured_result_validator", VALIDATOR_PATH)
SCHEMA = validator.load_json(ROOT / "schemas" / "experiment-result-v1.json")


def _write_config(path: Path, *, behavior: str, run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "forward-verification-config-test-1.0",
                "case": "dfg-cylinder-2d1-re20",
                "run_id": run_id,
                "dummy_behavior": behavior,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dummy_runner(path: Path) -> None:
    path.write_text(
        r'''from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(path, role, media_type):
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "media_type": media_type,
        "role": role,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text(encoding="utf-8"))
args.output_dir.mkdir(parents=True, exist_ok=True)
config_path = args.output_dir / "config.json"
shutil.copyfile(args.config, config_path)
environment = {
    "git": {
        "repository": "https://example.invalid/jax_cad.git",
        "commit": "1" * 40,
        "branch": "dummy",
        "dirty": False,
    },
    "hardware": {
        "hostname": "dummy-host",
        "platform": "dummy-platform",
        "cpu_model": None,
        "logical_cpu_count": 1,
        "process_cpu_affinity": None,
        "memory_bytes": None,
        "device_visibility": {},
        "accelerators": [],
    },
    "software": {
        "python": "3.11",
        "jax": None,
        "jaxlib": None,
        "numpy": None,
        "scipy": None,
        "jax_cuda_plugin": None,
        "jax_cuda_pjrt": None,
        "backend": None,
        "backend_platform_version": None,
        "cuda_toolkit_nvcc": None,
        "precision": "float64",
        "environment_variables": {},
    },
}
environment_path = args.output_dir / "environment.json"
write_json(environment_path, environment)
if config["dummy_behavior"] == "failure":
    sys.stdout.buffer.write(b"captured failure stdout\n")
    sys.stderr.buffer.write(b"captured failure stderr\n")
    raise SystemExit(7)
if config["dummy_behavior"] == "timeout":
    sys.stdout.write("captured timeout stdout\n")
    sys.stdout.flush()
    time.sleep(10.0)

stdout_path = args.output_dir / "stdout.log"
stderr_path = args.output_dir / "stderr.log"
stdout_path.write_bytes(b"synthetic inner stdout\n")
stderr_path.write_bytes(b"synthetic inner stderr\n")
result = {
    "schema_version": "1.0",
    "experiment_id": "dfg_cylinder_2d1_forward_verification",
    "run_id": config["run_id"],
    "git": environment["git"],
    "command": {"argv": ["synthetic"], "working_directory": "/synthetic"},
    "timestamps": {
        "start_utc": "2000-01-01T00:00:00+00:00",
        "end_utc": "2000-01-01T00:00:01+00:00",
        "status": "success",
        "exit_code": 0,
    },
    "configuration": {**config, "requires_gradient": False},
    "hardware": environment["hardware"],
    "software": environment["software"],
    "primal": {
        "converged": True,
        "iterations": 1,
        "residual_norm": 0.0,
        "divergence_rms": 0.0,
        "divergence_max": 0.0,
        "mass_flux_mismatch": 0.0,
        "failure_reason": None,
    },
    "objective": {
        "name": "dummy",
        "value": 0.0,
        "units": "dimensionless",
        "normalization": None,
    },
    "gradient": {
        "success": False,
        "backend": "none",
        "parameterization": None,
        "parameter_names": [],
        "values": [],
        "norm": None,
        "adjoint_residual": None,
        "message": "forward only",
        "taylor_test": None,
    },
    "timing": {
        "compile_seconds": 0.0,
        "forward_samples_seconds": [0.0],
        "gradient_samples_seconds": [],
        "optimizer_step_samples_seconds": [],
    },
    "memory": {
        "peak_host_bytes": None,
        "peak_accelerator_bytes": None,
        "measurement_method": "dummy",
    },
    "artifacts": [
        artifact(config_path, "config", "application/json"),
        artifact(environment_path, "environment", "application/json"),
        artifact(stdout_path, "log", "text/plain"),
        artifact(stderr_path, "log", "text/plain"),
    ],
    "notes": "dummy inner result",
}
write_json(args.output_dir / "result.json", result)
sys.stdout.buffer.write(b"captured success stdout\n")
sys.stderr.buffer.write(b"captured success stderr\n")
''',
        encoding="utf-8",
    )


def _validate(result_path: Path) -> dict:
    record = validator.load_json(result_path)
    validator.validate_record(
        record,
        schema=SCHEMA,
        result_path=result_path,
        verify_artifacts=True,
        verify_git_ancestry=False,
        paper_eligible=False,
    )
    return record


def test_success_replaces_synthetic_logs_and_records_exact_capture(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    runner = tmp_path / "dummy_runner.py"
    output = tmp_path / "output"
    _write_config(config, behavior="success", run_id="dummy-success")
    _write_dummy_runner(runner)

    exit_code = wrapper.run(config, output, inner_runner=runner)

    assert exit_code == 0
    assert (output / "stdout.log").read_bytes() == b"captured success stdout\n"
    assert (output / "stderr.log").read_bytes() == b"captured success stderr\n"
    record = _validate(output / "result.json")
    capture = record["configuration"]["subprocess_capture"]
    assert capture["inner_result"]["accepted"] is True
    assert capture["streams"]["stdout"]["byte_count"] == 24
    assert capture["streams"]["stderr"]["byte_count"] == 24
    assert capture["streams"]["stdout"]["sha256"] == wrapper._sha256_bytes(
        (output / "stdout.log").read_bytes()
    )
    assert record["timestamps"]["start_utc"] != "2000-01-01T00:00:00+00:00"
    assert capture["inner_reported"]["timestamps"]["start_utc"].startswith("2000-")
    assert record["command"]["argv"][1] == str(runner.resolve())
    log_artifacts = {
        item["path"]: item for item in record["artifacts"] if item["role"] == "log"
    }
    assert log_artifacts["stdout.log"]["sha256"] == capture["streams"]["stdout"]["sha256"]
    assert log_artifacts["stderr.log"]["sha256"] == capture["streams"]["stderr"]["sha256"]


def test_nonzero_exit_without_result_writes_valid_failed_evidence(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    runner = tmp_path / "dummy_runner.py"
    output = tmp_path / "output"
    _write_config(config, behavior="failure", run_id="dummy-failure")
    _write_dummy_runner(runner)

    exit_code = wrapper.run(config, output, inner_runner=runner)

    assert exit_code == 7
    assert (output / "stdout.log").read_bytes() == b"captured failure stdout\n"
    assert (output / "stderr.log").read_bytes() == b"captured failure stderr\n"
    assert json.loads((output / "config.json").read_text()) == json.loads(
        config.read_text()
    )
    assert (output / "environment.json").is_file()
    record = _validate(output / "result.json")
    assert record["timestamps"] ["status"] == "failed"
    assert record["timestamps"]["exit_code"] == 7
    assert record["primal"]["converged"] is False
    assert record["configuration"]["subprocess_capture"]["inner_result"] == {
        "present": False,
        "accepted": False,
        "normalization_error": None,
    }


def test_non_cylinder_failure_keeps_explicit_experiment_identity(tmp_path: Path) -> None:
    config, output, runner = tmp_path/'input.json', tmp_path/'output', tmp_path/'runner.py'
    _write_config(config, behavior='failure', run_id='pressure-response-failure')
    data = json.loads(config.read_text())
    data['case'] = 'manufactured-pressure-equilibrium'
    data['experiment_id'] = 'brinkman_pressure_response_diagnostic'
    config.write_text(json.dumps(data))
    _write_dummy_runner(runner)
    assert wrapper.run(config, output, inner_runner=runner) == 7
    record = _validate(output/'result.json')
    assert record['experiment_id'] == data['experiment_id']
    assert record['objective']['name'] == data['experiment_id']+'_capture'


def test_launch_exception_writes_valid_partial_evidence(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    output = tmp_path / "output"
    _write_config(config, behavior="unused", run_id="dummy-launch-error")

    exit_code = wrapper.run(
        config,
        output,
        inner_runner=tmp_path / "not_reached.py",
        python_executable=str(tmp_path / "missing-python"),
    )

    assert exit_code == 2
    record = _validate(output / "result.json")
    assert record["timestamps"]["status"] == "partial"
    assert record["timestamps"]["exit_code"] is None
    capture = record["configuration"]["subprocess_capture"]
    assert capture["exception"]["type"] == "FileNotFoundError"
    assert (output / "stdout.log").read_bytes() == b""
    assert (output / "stderr.log").read_bytes() == b""


def test_timeout_writes_valid_partial_evidence_and_keeps_streams(
    tmp_path: Path,
) -> None:
    config = tmp_path / "input.json"
    runner = tmp_path / "dummy_runner.py"
    output = tmp_path / "output"
    _write_config(config, behavior="timeout", run_id="dummy-timeout")
    _write_dummy_runner(runner)

    exit_code = wrapper.run(
        config,
        output,
        inner_runner=runner,
        timeout_seconds=0.2,
    )

    assert exit_code == 2
    record = _validate(output / "result.json")
    assert record["timestamps"]["status"] == "partial"
    assert record["timestamps"]["exit_code"] is None
    capture = record["configuration"]["subprocess_capture"]
    assert capture["exception"]["type"] == "TimeoutExpired"
    assert (output / "stdout.log").read_bytes() == b"captured timeout stdout\n"


def test_keyboard_interrupt_writes_valid_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "input.json"
    runner = tmp_path / "dummy_runner.py"
    output = tmp_path / "output"
    _write_config(config, behavior="success", run_id="dummy-interrupt")
    _write_dummy_runner(runner)
    real_run = wrapper.subprocess.run

    def interrupt_child(command, **kwargs):
        if command and command[0] == sys.executable:
            raise KeyboardInterrupt
        return real_run(command, **kwargs)

    monkeypatch.setattr(wrapper.subprocess, "run", interrupt_child)
    exit_code = wrapper.run(config, output, inner_runner=runner)

    assert exit_code == 2
    record = _validate(output / "result.json")
    assert record["timestamps"]["status"] == "partial"
    assert record["timestamps"]["exit_code"] is None
    capture = record["configuration"]["subprocess_capture"]
    assert capture["exception"]["type"] == "KeyboardInterrupt"
    assert b"received SIGINT" in (output / "stderr.log").read_bytes()


def test_nonempty_output_is_rejected_before_child_launch(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    runner = tmp_path / "dummy_runner.py"
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not overwrite\n", encoding="utf-8")
    _write_config(config, behavior="success", run_id="dummy-refusal")
    _write_dummy_runner(runner)

    with pytest.raises(FileExistsError, match="refusing to overwrite nonempty"):
        wrapper.run(config, output, inner_runner=runner)

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite\n"
    assert not (output / "result.json").exists()
