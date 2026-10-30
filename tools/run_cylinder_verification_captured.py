#!/usr/bin/env python3
"""Run the DFG cylinder case behind a lossless subprocess evidence boundary.

The cylinder driver writes convenient in-process summaries to ``stdout.log``
and ``stderr.log``.  Those files are not, however, the process streams seen by
the caller.  This wrapper owns a new output directory, captures the child
process streams as bytes, and makes the result record refer to those captured
bytes.  It also leaves a schema-valid failure record when the child exits
before it can write one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_INNER_RUNNER = REPO_ROOT / "examples" / "30_cylinder_dfg2d1.py"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "experiment-result-v1.json"

# ``tools`` is deliberately not a package.  Make sibling utilities importable
# both when this file is executed and when tests load it with importlib.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from validate_experiment_result import (  # noqa: E402
    ValidationProblem,
    load_json,
    validate_record,
)

from jax_cad.run_artifacts import (  # noqa: E402
    artifact_record,
    no_gradient_record,
    write_json,
)


class CaptureProblem(RuntimeError):
    """Raised when the outer evidence boundary cannot finalize a result."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="surrogateescape")


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _portable_environment() -> dict[str, Any]:
    """Return a schema-compatible manifest without initializing a JAX backend."""

    tracked_names = (
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "JAX_COMPILATION_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "XLA_FLAGS",
    )
    commit = _git_value("rev-parse", "HEAD")
    repository = _git_value("config", "--get", "remote.origin.url")
    branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        dirty = unstaged != 0 or staged != 0
    except OSError:
        dirty = True
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        "git": {
            "repository": repository or "unknown",
            "commit": commit if commit and len(commit) == 40 else "0" * 40,
            "branch": branch,
            "dirty": bool(dirty),
        },
        "hardware": {
            "hostname": socket.gethostname() or "unknown",
            "platform": platform.platform() or sys.platform,
            "cpu_model": platform.processor() or None,
            "logical_cpu_count": max(1, os.cpu_count() or 1),
            "process_cpu_affinity": affinity,
            "memory_bytes": None,
            "device_visibility": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "unset_policy": (
                    "runtime default; all enumerated devices may be visible"
                ),
            },
            "accelerators": [],
        },
        "software": {
            "python": sys.version.split()[0],
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "jax_cuda_plugin": _package_version("jax-cuda12-plugin"),
            "jax_cuda_pjrt": _package_version("jax-cuda12-pjrt"),
            "backend": None,
            "backend_platform_version": None,
            "cuda_toolkit_nvcc": None,
            "precision": "float64",
            "environment_variables": {
                name: os.environ[name]
                for name in tracked_names
                if name in os.environ
            },
        },
    }


def _read_environment_or_fallback(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            value = load_json(path)
            if all(name in value for name in ("git", "hardware", "software")):
                return value
        except (OSError, ValueError, TypeError):
            # Preserve the child's file as evidence; use portable fields in the
            # wrapper result rather than silently rewriting a malformed file.
            pass
    return _portable_environment()


def _capture_record(
    *,
    command: list[str],
    started: datetime,
    ended: datetime,
    return_code: int | None,
    stdout: bytes,
    stderr: bytes,
    exception: BaseException | None,
    inner_result_present: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "subprocess-capture-1.0",
        "method": "subprocess.run with binary stdout/stderr pipes",
        "argv": command,
        "working_directory": str(REPO_ROOT),
        "start_utc": started.isoformat(),
        "end_utc": ended.isoformat(),
        "exit_code": return_code,
        "exception": (
            None
            if exception is None
            else {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        ),
        "streams": {
            "stdout": {
                "path": "stdout.log",
                "byte_count": len(stdout),
                "sha256": _sha256_bytes(stdout),
            },
            "stderr": {
                "path": "stderr.log",
                "byte_count": len(stderr),
                "sha256": _sha256_bytes(stderr),
            },
        },
        "inner_result": {
            "present": inner_result_present,
            "accepted": False,
            "normalization_error": None,
        },
    }


def _replace_log_artifacts(
    record: dict[str, Any], output_dir: Path
) -> None:
    replacements = {
        "stdout.log": artifact_record(
            output_dir / "stdout.log",
            output_dir,
            media_type="text/plain",
            role="log",
        ),
        "stderr.log": artifact_record(
            output_dir / "stderr.log",
            output_dir,
            media_type="text/plain",
            role="log",
        ),
    }
    artifacts = []
    seen: set[str] = set()
    for artifact in record.get("artifacts", []):
        path = artifact.get("path")
        if path in replacements:
            if path not in seen:
                artifacts.append(replacements[path])
                seen.add(path)
        else:
            artifacts.append(artifact)
    for path in ("stdout.log", "stderr.log"):
        if path not in seen:
            artifacts.append(replacements[path])
    record["artifacts"] = artifacts


def _status_for_capture(
    inner_status: str | None,
    return_code: int | None,
    exception: BaseException | None,
) -> str:
    if exception is not None or return_code is None:
        return "partial"
    if return_code != 0:
        return inner_status if inner_status in {"failed", "partial"} else "partial"
    return inner_status if inner_status in {"success", "partial"} else "partial"


def _normalize_inner_result(
    record: dict[str, Any],
    *,
    output_dir: Path,
    command: list[str],
    capture_record: dict[str, Any],
    started: datetime,
    ended: datetime,
    return_code: int | None,
    exception: BaseException | None,
) -> dict[str, Any]:
    inner_timestamps = record.get("timestamps")
    inner_command = record.get("command")
    normalized_capture = dict(capture_record)
    normalized_capture["inner_reported"] = {
        "timestamps": inner_timestamps,
        "command": inner_command,
    }
    normalized_capture["inner_result"] = {
        "present": True,
        "accepted": True,
        "normalization_error": None,
    }
    configuration = record.get("configuration")
    if not isinstance(configuration, dict):
        raise CaptureProblem("inner result configuration is not an object")
    configuration["subprocess_capture"] = normalized_capture
    record["command"] = {
        "argv": command,
        "working_directory": str(REPO_ROOT),
    }
    inner_status = (
        inner_timestamps.get("status")
        if isinstance(inner_timestamps, dict)
        else None
    )
    record["timestamps"] = {
        "start_utc": started.isoformat(),
        "end_utc": ended.isoformat(),
        "status": _status_for_capture(inner_status, return_code, exception),
        "exit_code": return_code,
    }
    _replace_log_artifacts(record, output_dir)
    return record


def _ensure_support_files(
    *,
    config_path: Path,
    output_dir: Path,
    stdout: bytes,
    stderr: bytes,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output = output_dir / "config.json"
    if not config_output.exists():
        shutil.copyfile(config_path, config_output)
    environment_output = output_dir / "environment.json"
    environment = _read_environment_or_fallback(environment_output)
    if not environment_output.exists():
        write_json(environment_output, environment)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    return (
        config_output,
        environment_output,
        stdout_path,
        stderr_path,
        environment,
    )


def _fallback_result(
    *,
    config: dict[str, Any],
    environment: dict[str, Any],
    output_dir: Path,
    support_paths: tuple[Path, Path, Path, Path],
    command: list[str],
    capture_record: dict[str, Any],
    started: datetime,
    ended: datetime,
    return_code: int | None,
    exception: BaseException | None,
    reason: str,
) -> dict[str, Any]:
    status = "partial" if exception is not None or return_code in (None, 0) else "failed"
    configuration = dict(config)
    configuration["requires_gradient"] = False
    configuration["subprocess_capture"] = capture_record
    config_path, environment_path, stdout_path, stderr_path = support_paths
    return {
        "schema_version": "1.0",
        "experiment_id": str(config.get("experiment_id", "dfg_cylinder_2d1_forward_verification")),
        "run_id": str(config.get("run_id", "captured-cylinder-failure")),
        "git": environment["git"],
        "command": {
            "argv": command,
            "working_directory": str(REPO_ROOT),
        },
        "timestamps": {
            "start_utc": started.isoformat(),
            "end_utc": ended.isoformat(),
            "status": status,
            "exit_code": return_code,
        },
        "configuration": configuration,
        "hardware": environment["hardware"],
        "software": environment["software"],
        "primal": {
            "converged": False,
            "iterations": 0,
            "residual_norm": None,
            "divergence_rms": None,
            "divergence_max": None,
            "mass_flux_mismatch": None,
            "failure_reason": reason,
        },
        "objective": {
            "name": str(config["experiment_id"]) + "_capture" if "experiment_id" in config else "cylinder_verification_capture",
            "value": None,
            "units": "dimensionless",
            "normalization": None,
        },
        "gradient": no_gradient_record(
            "The inner subprocess did not produce an accepted result record."
        ),
        "timing": {
            "compile_seconds": None,
            "forward_samples_seconds": [],
            "gradient_samples_seconds": [],
            "optimizer_step_samples_seconds": [],
        },
        "memory": {
            "peak_host_bytes": None,
            "peak_accelerator_bytes": None,
            "measurement_method": "unavailable after subprocess failure",
        },
        "artifacts": [
            artifact_record(
                config_path, output_dir, media_type="application/json", role="config"
            ),
            artifact_record(
                environment_path,
                output_dir,
                media_type="application/json",
                role="environment",
            ),
            artifact_record(
                stdout_path, output_dir, media_type="text/plain", role="log"
            ),
            artifact_record(
                stderr_path, output_dir, media_type="text/plain", role="log"
            ),
        ],
        "notes": (
            "Outer subprocess capture retained configuration, environment, and "
            "both process streams after the inner runner failed to "
            "produce an accepted result.json."
        ),
    }


def _validate_written_result(result_path: Path) -> None:
    validate_record(
        load_json(result_path),
        schema=load_json(DEFAULT_SCHEMA),
        result_path=result_path,
        verify_artifacts=True,
        verify_git_ancestry=False,
        paper_eligible=False,
    )


def run(
    config_path: Path,
    output_dir: Path,
    *,
    inner_runner: Path = DEFAULT_INNER_RUNNER,
    python_executable: str = sys.executable,
    timeout_seconds: float | None = None,
) -> int:
    """Run and capture one cylinder case; return the child status when available."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    inner_runner = inner_runner.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_dir}")
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")

    command = [
        python_executable,
        str(inner_runner),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    started = _utc_now()
    return_code: int | None = None
    exception: BaseException | None = None
    stdout = b""
    stderr = b""
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        exception = error
        stdout = _as_bytes(error.stdout)
        stderr = _as_bytes(error.stderr)
    except KeyboardInterrupt as error:
        exception = error
        stderr = (
            stderr
            + b"capture wrapper received SIGINT; child execution interrupted\n"
        )
    except OSError as error:
        exception = error
    ended = _utc_now()

    result_path = output_dir / "result.json"
    capture_record = _capture_record(
        command=command,
        started=started,
        ended=ended,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        exception=exception,
        inner_result_present=result_path.is_file(),
    )
    (
        config_output,
        environment_output,
        stdout_path,
        stderr_path,
        environment,
    ) = _ensure_support_files(
        config_path=config_path,
        output_dir=output_dir,
        stdout=stdout,
        stderr=stderr,
    )
    support_paths = (
        config_output,
        environment_output,
        stdout_path,
        stderr_path,
    )

    normalized = False
    normalization_error: str | None = None
    if result_path.is_file():
        try:
            inner_record = load_json(result_path)
            if not isinstance(inner_record, dict):
                raise CaptureProblem("inner result root is not an object")
            record = _normalize_inner_result(
                inner_record,
                output_dir=output_dir,
                command=command,
                capture_record=capture_record,
                started=started,
                ended=ended,
                return_code=return_code,
                exception=exception,
            )
            write_json(result_path, record)
            _validate_written_result(result_path)
            normalized = True
        except (OSError, ValueError, TypeError, ValidationProblem, CaptureProblem) as error:
            normalization_error = str(error)

    if not normalized:
        capture_record["inner_result"] = {
            "present": result_path.is_file(),
            "accepted": False,
            "normalization_error": normalization_error,
        }
        if exception is not None:
            reason = f"subprocess launch or execution failed: {type(exception).__name__}: {exception}"
        elif return_code is None:
            reason = "subprocess ended without an exit status or result"
        elif return_code != 0:
            reason = f"cylinder subprocess exited with status {return_code} without an accepted result"
        else:
            reason = "cylinder subprocess exited successfully but produced no accepted result"
        record = _fallback_result(
            config=config,
            environment=environment,
            output_dir=output_dir,
            support_paths=support_paths,
            command=command,
            capture_record=capture_record,
            started=started,
            ended=ended,
            return_code=return_code,
            exception=exception,
            reason=reason,
        )
        write_json(result_path, record)
        _validate_written_result(result_path)

    if normalized and return_code is not None:
        return return_code
    if return_code not in (None, 0):
        return return_code
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--inner-runner",
        type=Path,
        default=DEFAULT_INNER_RUNNER,
        help="cylinder driver path (the default is examples/30_cylinder_dfg2d1.py)",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used for the child process",
    )
    parser.add_argument("--timeout-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(
            args.config,
            args.output_dir,
            inner_runner=args.inner_runner,
            python_executable=args.python_executable,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, ValidationProblem, CaptureProblem) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
