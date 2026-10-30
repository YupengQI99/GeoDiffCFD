#!/usr/bin/env python3
"""Build convergence tables and figures only from archived run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .validate_experiment_result import (
        DEFAULT_SCHEMA,
        ValidationProblem,
        load_json,
        validate_record,
    )
except ImportError:  # Direct script execution places tools/ on sys.path.
    from validate_experiment_result import (
        DEFAULT_SCHEMA,
        ValidationProblem,
        load_json,
        validate_record,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_result_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        candidate = Path(raw)
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("result.json")))
        elif candidate.is_file() and candidate.name == "result.json":
            paths.append(candidate)
        else:
            paths.extend(sorted(Path().glob(raw)))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValidationProblem("no result.json files found")
    return unique


def _metrics_path(result_path: Path, record: dict[str, Any]) -> Path:
    matches = [
        result_path.parent / artifact["path"]
        for artifact in record["artifacts"]
        if artifact["role"] == "table"
        and Path(artifact["path"]).name == "metrics.json"
    ]
    if len(matches) != 1:
        raise ValidationProblem(
            f"{result_path}: expected one metrics.json table artifact"
        )
    return matches[0]


def _base_row(
    result_path: Path, record: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    config = record["configuration"]
    grid = config.get("grid")
    if grid is None:
        mesh = config.get("mesh", {})
        nx = None
        ny = None
        dx = None
        dy = None
        h = mesh.get("cylinder_size")
        dt = config.get("solver", {}).get("dt")
    else:
        nx = grid["nx"]
        ny = grid["ny"]
        dx = grid["lx"] / grid["nx"]
        dy = grid["ly"] / grid["ny"]
        h = max(dx, dy)
        dt = config["solver"]["dt"]
    accelerators = record["hardware"]["accelerators"]
    samples = record["timing"]["forward_samples_seconds"]
    try:
        archived_result_path = result_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        archived_result_path = result_path.resolve()
    return {
        "case": config["case"],
        "config_schema_version": config.get("schema_version"),
        "experiment_id": record["experiment_id"],
        "run_id": record["run_id"],
        "study": config.get("study", {}).get("name"),
        "study_level": config.get("study", {}).get("level"),
        "study_value": config.get("study", {}).get("value"),
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "h": h,
        "dt": dt,
        "backend": record["software"]["backend"],
        "device": accelerators[0]["name"] if accelerators else None,
        "commit": record["git"]["commit"],
        "dirty": record["git"]["dirty"],
        "status": record["timestamps"]["status"],
        "converged": record["primal"]["converged"],
        "iterations": record["primal"]["iterations"],
        "residual_rms": record["primal"]["residual_norm"],
        "divergence_rms": record["primal"]["divergence_rms"],
        "divergence_max": record["primal"]["divergence_max"],
        "mass_flux_mismatch": record["primal"]["mass_flux_mismatch"],
        "objective_name": record["objective"]["name"],
        "objective": record["objective"]["value"],
        "compile_seconds": record["timing"]["compile_seconds"],
        "forward_median_seconds": median(samples) if samples else None,
        "peak_host_bytes": record["memory"]["peak_host_bytes"],
        "peak_accelerator_bytes": record["memory"]["peak_accelerator_bytes"],
        "result_path": archived_result_path.as_posix(),
        "result_sha256": _sha256(result_path),
        "metrics": metrics,
    }


def _flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "metrics"}
    metrics = row["metrics"]
    case = row["case"]
    if case == "planar-poiseuille":
        flat.update(
            {
                "velocity_l2_relative": metrics["errors"]["velocity_l2_relative"],
                "velocity_linf_relative": metrics["errors"]["velocity_linf_relative"],
                "mean_velocity_relative": metrics["errors"]["mean_velocity_relative"],
                "centreline_velocity_relative": metrics["errors"]["centreline_velocity_relative"],
                "pressure_gradient_relative": metrics["errors"]["pressure_gradient_relative"],
            }
        )
    elif case == "lid-driven-cavity-re100":
        flat.update(
            {
                "profile_l2_relative": metrics["profiles"]["combined_l2_relative"],
                "u_l2_relative": metrics["profiles"]["u_l2_relative"],
                "v_l2_relative": metrics["profiles"]["v_l2_relative"],
                "vortex_x": metrics["primary_vortex"]["x"],
                "vortex_y": metrics["primary_vortex"]["y"],
                "vortex_position_error": metrics["primary_vortex"]["position_error"],
            }
        )
    elif case == "dfg-cylinder-2d1-finite-brinkman":
        independent = metrics["metrics"]
        model = metrics["formulation"]["model"]
        flat.update({
            "measurement_revision": "dolfinx-full-domain-finite-brinkman-p2p1-v1",
            "coefficient_sampling": "continuous-at-fe-quadrature",
            "alpha_max": model["alpha"], "epsilon": model["epsilon"],
            "mask_midpoint_shift": model["shift"],
            "brinkman_drag_coefficient": independent["drag_coefficient"],
            "brinkman_lift_coefficient": independent["lift_coefficient"],
            "pressure_difference_direct": independent["pressure_difference"],
            "pressure_difference": independent["pressure_difference_radial_extrapolated"],
            "cells_global": metrics["mesh"]["cells"],
            "mixed_dofs_global": metrics["mesh"]["mixed_dofs"],
            "local_refinements": metrics["mesh"].get("local_refinements", 0),
            "triangle_diagonal": metrics["mesh"]["diagonal"],
            "quadrature_degree": metrics["formulation"]["quadrature_degree"],
            "advection_scheme": metrics["formulation"].get("advection_form", "advective"),
        })
    elif case == "dfg-cylinder-2d1-re20":
        if row["experiment_id"] == "dolfinx_dfg2d1_independent_reference":
            independent = metrics["metrics"]
            errors = metrics["relative_errors"]
            conservation = metrics["conservation"]
            flat.update(
                {
                    "measurement_revision": "dolfinx-boundary-fitted-p2p1-v1",
                    "surface_drag_coefficient": independent["drag_coefficient"],
                    "surface_lift_coefficient": independent["lift_coefficient"],
                    "pressure_difference": independent["pressure_difference"],
                    "surface_drag_relative_error": errors["drag_coefficient"],
                    "surface_lift_relative_error": errors["lift_coefficient"],
                    "pressure_relative_error": errors["pressure_difference"],
                    "relative_inlet_outlet_mismatch": conservation[
                        "relative_inlet_outlet_mismatch"
                    ],
                    "cells_global": metrics["mesh"]["cells_global"],
                    "mixed_dofs_global": metrics["mesh"]["mixed_dofs_global"],
                }
            )
            return flat
        immersed = metrics["immersed_boundary"]
        numerical = metrics["numerical"]
        errors = metrics["relative_errors"]
        surface_drag = numerical.get("surface_drag_coefficient")
        surface_lift = numerical.get("surface_lift_coefficient")
        brinkman_drag = numerical.get(
            "brinkman_drag_coefficient", numerical.get("drag_coefficient")
        )
        brinkman_lift = numerical.get(
            "brinkman_lift_coefficient", numerical.get("lift_coefficient")
        )
        flat.update(
            {
                "measurement_revision": (
                    "finite-brinkman-full-surface-v1"
                    if surface_drag is not None
                    else "brinkman-volume-reaction-v0"
                ),
                "epsilon_over_h": immersed["epsilon_over_h"],
                "alpha_max": immersed["alpha_max"],
                "advection_scheme": metrics.get("advection_scheme", "donor-cell"),
                "coefficient_sampling": metrics.get("immersed_boundary", {}).get("coefficient_sampling", "cell-average"),
                "delta_b_over_h": immersed.get("delta_b_over_h"),
                "mask_midpoint_shift_over_h": immersed.get("mask_midpoint_shift_over_h"),
                "drag_coefficient": numerical.get("drag_coefficient"),
                "lift_coefficient": numerical.get("lift_coefficient"),
                "surface_drag_coefficient": surface_drag,
                "surface_lift_coefficient": surface_lift,
                "brinkman_drag_coefficient": brinkman_drag,
                "brinkman_lift_coefficient": brinkman_lift,
                "pressure_difference": numerical["pressure_difference"],
                "recirculation_length": numerical[
                    "recirculation_length_from_cylinder_rear"
                ],
                "drag_relative_error": errors.get("drag_coefficient"),
                "surface_drag_relative_error": errors.get(
                    "surface_drag_coefficient"
                ),
                "brinkman_drag_relative_error": errors.get(
                    "brinkman_drag_coefficient", errors.get("drag_coefficient")
                ),
                "pressure_relative_error": errors["pressure_difference"],
            }
        )
    return flat


def _convergence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    convergence: list[dict[str, Any]] = []
    series = sorted(
        {
            (
                row["case"],
                row["objective_name"],
                row.get("config_schema_version"),
            )
            for row in rows
        },
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    )
    for case, objective_name, config_schema_version in series:
        selected = sorted(
            (
                row
                for row in rows
                if row["case"] == case
                and row["objective_name"] == objective_name
                and row.get("config_schema_version") == config_schema_version
                and row["study"] == "grid-refinement"
            ),
            key=lambda row: row["h"],
            reverse=True,
        )
        spacings = [row["h"] for row in selected]
        if len(spacings) != len(set(spacings)):
            raise ValidationProblem(
                f"duplicate grid spacing in convergence series {case!r}, "
                f"{objective_name!r}, {config_schema_version!r}"
            )
        previous = None
        for row in selected:
            order = None
            if (
                previous is not None
                and previous["objective"] is not None
                and row["objective"] is not None
                and previous["objective"] > 0.0
                and row["objective"] > 0.0
            ):
                order = math.log(previous["objective"] / row["objective"]) / math.log(
                    previous["h"] / row["h"]
                )
            convergence.append(
                {
                    "case": case,
                    "config_schema_version": config_schema_version,
                    "objective_name": objective_name,
                    "run_id": row["run_id"],
                    "nx": row["nx"],
                    "ny": row["ny"],
                    "h": row["h"],
                    "objective": row["objective"],
                    "observed_order_from_previous": order,
                }
            )
            previous = row
    return convergence


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for axis, case, title in zip(
        axes,
        (
            "planar-poiseuille",
            "lid-driven-cavity-re100",
            "dfg-cylinder-2d1-re20",
        ),
        ("Poiseuille", "Cavity Re=100", "DFG cylinder 2D-1"),
    ):
        series = sorted(
            {
                (row["objective_name"], row.get("config_schema_version"))
                for row in rows
                if row["case"] == case and row["study"] == "grid-refinement"
            },
            key=lambda item: tuple(
                "" if value is None else str(value) for value in item
            ),
        )
        for objective_name, config_schema_version in series:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case
                    and row["objective_name"] == objective_name
                    and row.get("config_schema_version")
                    == config_schema_version
                    and row["study"] == "grid-refinement"
                    and row["objective"] is not None
                    and row["objective"] > 0.0
                ),
                key=lambda row: row["h"],
            )
            if not selected:
                continue
            axis.loglog(
                [row["h"] for row in selected],
                [row["objective"] for row in selected],
                marker="o",
                label=f"{objective_name} [{config_schema_version}]",
            )
        axis.set_title(title)
        axis.set_xlabel("h = max(dx, dy)")
        axis.set_ylabel("archived objective error")
        axis.grid(True, which="both", alpha=0.3)
        if len(series) > 1:
            axis.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="result files, directories, or globs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(DEFAULT_SCHEMA)
    rows: list[dict[str, Any]] = []
    for result_path in _collect_result_paths(args.inputs):
        record = load_json(result_path)
        validate_record(
            record,
            schema=schema,
            result_path=result_path,
            verify_artifacts=True,
            verify_git_ancestry=False,
            paper_eligible=False,
        )
        metrics = load_json(_metrics_path(result_path, record))
        rows.append(_base_row(result_path, record, metrics))

    flat_rows = [_flatten_metrics(row) for row in rows]
    convergence = _convergence_rows(rows)
    _write_csv(args.output_dir / "summary.csv", flat_rows)
    _write_csv(args.output_dir / "convergence.csv", convergence)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"runs": flat_rows, "convergence": convergence},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.figure:
        _write_figure(args.output_dir / "grid_convergence.svg", rows)
    print(f"summarized {len(rows)} validated run(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
