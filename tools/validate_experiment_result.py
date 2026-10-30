#!/usr/bin/env python3
"""Validate auditable experiment-result records.

JSON Schema is followed by semantic checks for finite values, timestamp order,
gradient-vector consistency, Taylor-array lengths, artifact hashes, optional Git
ancestry, and a stricter manuscript-eligibility gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "experiment-result-v1.json"


class ValidationProblem(RuntimeError):
    """Raised when a result fails a semantic validation rule."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant)


def _json_path(parts: Iterable[Any]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _walk_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValidationProblem(f"{path}: numeric value is not finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{path}.{key}")
        return
    raise ValidationProblem(f"{path}: unsupported value type {type(value).__name__}")


def _parse_datetime(value: str, path: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationProblem(f"{path}: invalid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValidationProblem(f"{path}: timestamp must include a UTC offset")
    return parsed


def _check_status(record: dict[str, Any]) -> None:
    timestamps = record["timestamps"]
    start = _parse_datetime(timestamps["start_utc"], "$.timestamps.start_utc")
    end = _parse_datetime(timestamps["end_utc"], "$.timestamps.end_utc")
    if end < start:
        raise ValidationProblem("$.timestamps: end_utc precedes start_utc")

    if timestamps["status"] == "success":
        if timestamps["exit_code"] != 0:
            raise ValidationProblem("$.timestamps: success requires exit_code=0")
        if not record["primal"]["converged"]:
            raise ValidationProblem("$.primal: success requires converged=true")
        if record["objective"]["value"] is None:
            raise ValidationProblem("$.objective.value: success requires a value")
    elif timestamps["status"] == "failed" and timestamps["exit_code"] == 0:
        raise ValidationProblem("$.timestamps: failed runs cannot have exit_code=0")

    if record["primal"]["converged"] and record["primal"]["failure_reason"]:
        raise ValidationProblem(
            "$.primal.failure_reason: converged runs cannot report failure"
        )


def _check_gradient(record: dict[str, Any]) -> None:
    gradient = record["gradient"]
    names = gradient["parameter_names"]
    values = gradient["values"]
    if len(names) != len(values):
        raise ValidationProblem(
            "$.gradient: parameter_names and values must have equal lengths"
        )

    if gradient["backend"] == "none":
        if gradient["success"] or values or gradient["norm"] is not None:
            raise ValidationProblem(
                "$.gradient: backend='none' requires success=false, empty values, norm=null"
            )
        return

    if gradient["success"]:
        if not values or gradient["norm"] is None:
            raise ValidationProblem(
                "$.gradient: a successful gradient requires values and norm"
            )
        computed = math.sqrt(sum(float(value) ** 2 for value in values))
        reported = float(gradient["norm"])
        tolerance = max(1.0e-12, 1.0e-8 * max(1.0, computed))
        if abs(reported - computed) > tolerance:
            raise ValidationProblem(
                "$.gradient.norm: does not match Euclidean norm of values "
                f"(reported={reported:.16g}, computed={computed:.16g})"
            )

    taylor = gradient["taylor_test"]
    if taylor is not None:
        epsilons = taylor["epsilons"]
        remainders = taylor["first_order_remainders"]
        rates = taylor["first_order_rates"]
        if len(remainders) != len(epsilons):
            raise ValidationProblem(
                "$.gradient.taylor_test: remainder count must equal epsilon count"
            )
        if len(rates) != len(epsilons) - 1:
            raise ValidationProblem(
                "$.gradient.taylor_test: rate count must be epsilon count minus one"
            )
        if any(a <= b for a, b in zip(epsilons, epsilons[1:])):
            raise ValidationProblem(
                "$.gradient.taylor_test.epsilons: values must strictly decrease"
            )


def _check_unique_artifacts(record: dict[str, Any]) -> None:
    paths = [artifact["path"] for artifact in record["artifacts"]]
    if len(paths) != len(set(paths)):
        raise ValidationProblem("$.artifacts: paths must be unique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifacts(record: dict[str, Any], result_path: Path) -> None:
    base = result_path.parent.resolve()
    for index, artifact in enumerate(record["artifacts"]):
        relative = Path(artifact["path"])
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError as error:
            raise ValidationProblem(
                f"$.artifacts[{index}].path: escapes the result directory"
            ) from error
        if not target.is_file():
            raise ValidationProblem(
                f"$.artifacts[{index}].path: missing {relative.as_posix()}"
            )
        if _sha256(target).lower() != artifact["sha256"].lower():
            raise ValidationProblem(
                f"$.artifacts[{index}].sha256: checksum mismatch for "
                f"{relative.as_posix()}"
            )


def check_git_ancestry(record: dict[str, Any], cwd: Path) -> None:
    commit = record["git"]["commit"]
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValidationProblem(
            f"$.git.commit: {commit} is unavailable or not an ancestor of HEAD"
        ) from error


def check_paper_eligibility(record: dict[str, Any]) -> None:
    if record["timestamps"]["status"] != "success":
        raise ValidationProblem("paper gate: run status is not success")
    if record["git"]["dirty"]:
        raise ValidationProblem("paper gate: dirty working tree is inadmissible")
    if not record["primal"]["converged"]:
        raise ValidationProblem("paper gate: primal solve did not converge")
    required = (
        "residual_norm",
        "divergence_rms",
        "divergence_max",
        "mass_flux_mismatch",
    )
    missing = [name for name in required if record["primal"][name] is None]
    if missing:
        raise ValidationProblem(
            "paper gate: missing primal diagnostics: " + ", ".join(missing)
        )
    if not record["timing"]["forward_samples_seconds"]:
        raise ValidationProblem("paper gate: forward timing samples required")
    reference_comparison = record["configuration"].get("reference_comparison")
    if reference_comparison is not None:
        if not isinstance(reference_comparison, dict):
            raise ValidationProblem(
                "paper gate: configuration.reference_comparison must be an object"
            )
        if reference_comparison.get("paper_eligible") is not True:
            raise ValidationProblem(
                "paper gate: reference comparison is explicitly ineligible"
            )
        if reference_comparison.get("boundary_conditions_exact") is not True:
            raise ValidationProblem(
                "paper gate: reference boundary conditions are not exact"
            )
        if reference_comparison.get("quantity_definition_exact") is not True:
            raise ValidationProblem(
                "paper gate: reference quantity definition is not exact"
            )
    requires_gradient = record["configuration"].get("requires_gradient", True)
    if not isinstance(requires_gradient, bool):
        raise ValidationProblem(
            "paper gate: configuration.requires_gradient must be boolean"
        )
    if requires_gradient:
        gradient = record["gradient"]
        if not gradient["success"] or gradient["backend"] == "none":
            raise ValidationProblem("paper gate: successful gradient required")
        if not record["timing"]["gradient_samples_seconds"]:
            raise ValidationProblem("paper gate: gradient timing samples required")


def validate_record(
    record: dict[str, Any],
    *,
    schema: dict[str, Any],
    result_path: Path,
    verify_artifacts: bool,
    verify_git_ancestry: bool,
    paper_eligible: bool,
) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        details = "\n  ".join(
            f"{_json_path(error.path)}: {error.message}" for error in errors
        )
        raise ValidationProblem("schema validation failed:\n  " + details)
    _walk_finite(record)
    _check_status(record)
    _check_gradient(record)
    _check_unique_artifacts(record)
    if verify_artifacts:
        check_artifacts(record, result_path)
    if verify_git_ancestry:
        check_git_ancestry(record, REPO_ROOT)
    if paper_eligible:
        check_paper_eligibility(record)


def _collect_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        candidate = Path(raw)
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("result.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(sorted(Path().glob(raw)))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValidationProblem("no result JSON files found")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="result files, directories, or globs")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--check-git-ancestry", action="store_true")
    parser.add_argument("--paper-eligible", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        schema = load_json(args.schema)
        inputs = _collect_inputs(args.inputs)
    except (OSError, ValueError, ValidationProblem) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    failures = 0
    for path in inputs:
        try:
            record = load_json(path)
            validate_record(
                record,
                schema=schema,
                result_path=path,
                verify_artifacts=args.check_artifacts,
                verify_git_ancestry=args.check_git_ancestry,
                paper_eligible=args.paper_eligible,
            )
        except (OSError, ValueError, ValidationProblem) as error:
            failures += 1
            print(f"[FAIL] {path}: {error}", file=sys.stderr)
        else:
            if not args.quiet:
                print(f"[OK]   {path}")

    if failures:
        print(f"Validated {len(inputs)} result(s): {failures} failed", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"Validated {len(inputs)} result(s): all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
