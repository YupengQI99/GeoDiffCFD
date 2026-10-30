"""Auditable Re=100 lid-driven-cavity verification against Ghia et al."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_cad.benchmarks import (
    GHIA_1982_DOI,
    GHIA_RE100_PRIMARY_VORTEX,
    GHIA_RE100_U,
    GHIA_RE100_U_Y,
    GHIA_RE100_V,
    GHIA_RE100_V_X,
)
from jax_cad.differentiable.mac_cavity import (
    CavityWalls,
    cavity_flux_diagnostics,
    create_cavity_initial_state,
    solve_cavity_steady,
)
from jax_cad.differentiable.mac_channel import ChannelParams, create_channel_grid
from jax_cad.run_artifacts import (
    artifact_record,
    compile_and_sample_args,
    no_gradient_record,
    peak_memory_record,
    write_json,
)
from jax_cad.verification import error_norms, primary_vortex_vertex

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.capture_compute_environment import capture


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("case") != "lid-driven-cavity-re100":
        raise ValueError("configuration case must be 'lid-driven-cavity-re100'")
    if config.get("schema_version") != "forward-verification-config-1.0":
        raise ValueError("unsupported forward-verification config version")
    physics = config["physics"]
    reynolds = (
        float(physics["lid_velocity"])
        * float(config["grid"]["lx"])
        / float(physics["viscosity"])
    )
    if abs(reynolds - 100.0) > 1.0e-12:
        raise ValueError(f"configured cavity Reynolds number is {reynolds}, not 100")
    if config["reference"].get("doi") != GHIA_1982_DOI:
        raise ValueError("configuration does not name the frozen Ghia DOI")
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
        "pressure_mean",
        "boundary_normal_flux_l1",
        "net_boundary_flux",
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
                    "pressure_mean": float(
                        diagnostics.pressure_mean_history[index]
                    ),
                    "boundary_normal_flux_l1": float(
                        diagnostics.boundary_normal_flux_l1_history[index]
                    ),
                    "net_boundary_flux": float(
                        diagnostics.net_boundary_flux_history[index]
                    ),
                }
            )


def _write_profiles(
    path: Path,
    ghia_y,
    u_computed,
    u_reference,
    ghia_x,
    v_computed,
    v_reference,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "component",
                "coordinate",
                "computed",
                "reference",
                "error",
            )
        )
        for coordinate, computed, reference in zip(
            ghia_y, u_computed, u_reference
        ):
            writer.writerow(
                (
                    "u_at_x_half",
                    float(coordinate),
                    float(computed),
                    float(reference),
                    float(computed - reference),
                )
            )
        for coordinate, computed, reference in zip(
            ghia_x, v_computed, v_reference
        ):
            writer.writerow(
                (
                    "v_at_y_half",
                    float(coordinate),
                    float(computed),
                    float(reference),
                    float(computed - reference),
                )
            )


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
    solver = config["solver"]
    grid = create_channel_grid(
        int(grid_config["nx"]),
        int(grid_config["ny"]),
        float(grid_config["lx"]),
        float(grid_config["ly"]),
    )
    if grid.nx % 2 or grid.ny % 2:
        raise ValueError("cavity centreline extraction requires even nx and ny")
    lid_velocity = float(physics["lid_velocity"])
    state0 = create_cavity_initial_state(grid, dtype=jnp.float64)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    walls = CavityWalls(top_u=lid_velocity)
    params = ChannelParams(
        viscosity=float(physics["viscosity"]),
        dt=float(solver["dt"]),
        pressure_tolerance=float(solver["pressure_tolerance"]),
        pressure_atol=float(solver["pressure_atol"]),
        pressure_maxiter=int(solver["pressure_maxiter"]),
        include_advection=bool(solver["include_advection"]),
        nonlinear_rtol=float(solver["nonlinear_rtol"]),
        nonlinear_atol=float(solver["nonlinear_atol"]),
        nonlinear_min_steps=int(solver["nonlinear_min_steps"]),
    )
    result, compile_seconds, forward_samples = compile_and_sample_args(
        lambda s0, a, steps: solve_cavity_steady(s0, a, grid, params, walls, steps),
        (state0, alpha, int(solver["max_steps"])),
        int(config["timing_samples"]),
        static_argnums=(2,),
    )
    state, diagnostics = result

    y_centres = (jnp.arange(grid.ny, dtype=jnp.float64) + 0.5) * grid.dy
    x_centres = (jnp.arange(grid.nx, dtype=jnp.float64) + 0.5) * grid.dx
    u_line = state.u[grid.nx // 2, :]
    v_line = state.v[:, grid.ny // 2]
    u_coordinates = jnp.concatenate(
        (jnp.asarray([0.0]), y_centres, jnp.asarray([grid.ly]))
    )
    u_values = jnp.concatenate(
        (
            jnp.asarray([walls.bottom_u]),
            u_line,
            jnp.asarray([walls.top_u]),
        )
    )
    v_coordinates = jnp.concatenate(
        (jnp.asarray([0.0]), x_centres, jnp.asarray([grid.lx]))
    )
    v_values = jnp.concatenate(
        (
            jnp.asarray([walls.left_v]),
            v_line,
            jnp.asarray([walls.right_v]),
        )
    )
    ghia_y = jnp.asarray(GHIA_RE100_U_Y, dtype=jnp.float64) * grid.ly
    ghia_u = jnp.asarray(GHIA_RE100_U, dtype=jnp.float64) * lid_velocity
    ghia_x = jnp.asarray(GHIA_RE100_V_X, dtype=jnp.float64) * grid.lx
    ghia_v = jnp.asarray(GHIA_RE100_V, dtype=jnp.float64) * lid_velocity
    computed_u = jnp.interp(ghia_y, u_coordinates, u_values)
    computed_v = jnp.interp(ghia_x, v_coordinates, v_values)
    u_norms = error_norms(computed_u, ghia_u)
    v_norms = error_norms(computed_v, ghia_v)
    vortex_x, vortex_y, vortex_psi = primary_vortex_vertex(state, grid)
    reference_vortex = GHIA_RE100_PRIMARY_VORTEX
    vortex_position_error = jnp.sqrt(
        (vortex_x / grid.lx - reference_vortex["x"]) ** 2
        + (vortex_y / grid.ly - reference_vortex["y"]) ** 2
    )
    div_rms, div_max, pressure_mean, flux_l1, net_flux = (
        cavity_flux_diagnostics(state, grid)
    )
    combined_profile_error = jnp.sqrt(
        0.5 * (u_norms.l2_relative**2 + v_norms.l2_relative**2)
    )

    metrics = {
        "case": "lid-driven-cavity-re100",
        "reference": {
            "name": "Ghia, Ghia, and Shin (1982)",
            "doi": GHIA_1982_DOI,
            "primary_vortex": reference_vortex,
        },
        "profiles": {
            "u_l2_absolute": float(u_norms.l2_absolute),
            "u_linf_absolute": float(u_norms.linf_absolute),
            "u_l2_relative": float(u_norms.l2_relative),
            "u_linf_relative": float(u_norms.linf_relative),
            "v_l2_absolute": float(v_norms.l2_absolute),
            "v_linf_absolute": float(v_norms.linf_absolute),
            "v_l2_relative": float(v_norms.l2_relative),
            "v_linf_relative": float(v_norms.linf_relative),
            "combined_l2_relative": float(combined_profile_error),
        },
        "primary_vortex": {
            "x": float(vortex_x / grid.lx),
            "y": float(vortex_y / grid.ly),
            "streamfunction": float(vortex_psi / (lid_velocity * grid.lx)),
            "position_error": float(vortex_position_error),
        },
        "conservation": {
            "divergence_rms": float(div_rms),
            "divergence_max": float(div_max),
            "pressure_mean": float(pressure_mean),
            "boundary_normal_flux_l1": float(flux_l1),
            "net_boundary_flux": float(net_flux),
        },
        "convergence": {
            "converged": bool(diagnostics.converged),
            "iterations": int(diagnostics.iterations),
            "initial_residual_rms": float(diagnostics.initial_residual_rms),
            "final_residual_rms": float(diagnostics.final_residual_rms),
            "final_residual_max": float(diagnostics.final_residual_max),
        },
    }

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    profiles_path = output_dir / "profiles.csv"
    _write_profiles(
        profiles_path,
        ghia_y,
        computed_u,
        ghia_u,
        ghia_x,
        computed_v,
        ghia_v,
    )
    history_path = output_dir / "residual_history.csv"
    _write_history(history_path, diagnostics)
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
    status = "success" if converged else "partial"
    exit_code = 0 if converged else 1
    flux_scale = max(abs(lid_velocity) * grid.lx, 1.0e-300)
    result_record = {
        "schema_version": "1.0",
        "experiment_id": "cavity_re100_forward_verification",
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
            "pressure_gauge": "zero discrete mean",
            "reference_comparison": {
                "boundary_conditions_exact": True,
                "quantity_definition_exact": True,
                "paper_eligible": True,
            },
            "mass_flux_mismatch_definition": (
                "closed-boundary normal-flux L1 divided by U_lid*L"
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
            "mass_flux_mismatch": float(flux_l1 / flux_scale),
            "failure_reason": None if converged else "nonlinear tolerance not met",
        },
        "objective": {
            "name": "ghia_centerline_combined_relative_l2_error",
            "value": float(combined_profile_error),
            "units": "dimensionless",
            "normalization": "published Re=100 Ghia centreline tables",
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
            "Issue #5 Re=100 cavity; the open-channel pressure outlet is not used."
        ),
    }
    artifact_specs = (
        (resolved_config_path, "application/json", "config"),
        (environment_path, "application/json", "environment"),
        (metrics_path, "application/json", "table"),
        (profiles_path, "text/csv", "table"),
        (history_path, "text/csv", "residual-history"),
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
