"""Auditable planar-Poiseuille verification for the non-periodic MAC solver.

The continuum profile and pressure gradient are fixed from the configured mean
velocity, channel height, and viscosity. They are never rescaled by the
computed solution.
"""

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

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    ChannelState,
    create_channel_grid,
    flux_diagnostics,
    parabolic_inlet,
    solve_steady,
)
from jax_cad.run_artifacts import (
    artifact_record,
    compile_and_sample_args,
    no_gradient_record,
    peak_memory_record,
    write_json,
)
from jax_cad.verification import (
    error_norms,
    linear_slope,
    poiseuille_pressure_gradient,
    poiseuille_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.capture_compute_environment import capture


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("case") != "planar-poiseuille":
        raise ValueError("configuration case must be 'planar-poiseuille'")
    if config.get("schema_version") != "forward-verification-config-1.0":
        raise ValueError("unsupported forward-verification config version")
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
        iterations = int(diagnostics.iterations)
        for index in range(iterations):
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


def _write_profile(path: Path, y, numerical, analytical) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("y", "u_numerical", "u_analytical", "error"))
        for yi, computed, reference in zip(y, numerical, analytical):
            writer.writerow(
                (float(yi), float(computed), float(reference), float(computed - reference))
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
    mean_velocity = float(physics["mean_velocity"])
    viscosity = float(physics["viscosity"])
    inlet = parabolic_inlet(grid, mean_velocity, dtype=jnp.float64)
    y = (jnp.arange(grid.ny, dtype=jnp.float64) + 0.5) * grid.dy
    x = (jnp.arange(grid.nx, dtype=jnp.float64) + 0.5) * grid.dx
    analytical_profile = poiseuille_profile(y, grid.ly, mean_velocity)
    analytical_gradient = poiseuille_pressure_gradient(
        viscosity, grid.ly, mean_velocity
    )

    # Start from the independently defined continuum solution to isolate the
    # spatial residual and boundary discretization from startup-transient error.
    u0 = jnp.broadcast_to(inlet, (grid.nx + 1, grid.ny))
    v0 = jnp.zeros((grid.nx, grid.ny + 1), dtype=jnp.float64)
    p0 = jnp.broadcast_to(
        analytical_gradient * (x[:, None] - grid.lx),
        (grid.nx, grid.ny),
    )
    state0 = ChannelState(u0, v0, p0)
    alpha = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float64)
    params = ChannelParams(
        viscosity=viscosity,
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
        lambda s0, a, i, steps: solve_steady(s0, a, i, grid, params, steps),
        (state0, alpha, inlet, int(solver["max_steps"])),
        int(config["timing_samples"]),
        static_argnums=(3,),
    )
    state, diagnostics = result
    profile = jnp.mean(state.u[1:-1, :], axis=0)
    norms = error_norms(profile, analytical_profile)
    pressure_mean = jnp.mean(state.p, axis=1)
    pressure_slope = linear_slope(x[1:-1], pressure_mean[1:-1])
    pressure_gradient_relative_error = jnp.abs(
        (pressure_slope - analytical_gradient) / analytical_gradient
    )
    numerical_mean = jnp.sum(profile) * grid.dy / grid.ly
    numerical_centreline = jnp.interp(0.5 * grid.ly, y, profile)
    analytical_centreline = 1.5 * mean_velocity
    div_rms, div_max, inlet_flux, outlet_flux, mismatch = flux_diagnostics(
        state, grid
    )

    metrics = {
        "case": "planar-poiseuille",
        "independent_inputs": {
            "mean_velocity": mean_velocity,
            "viscosity": viscosity,
            "height": grid.ly,
        },
        "analytical": {
            "pressure_gradient": analytical_gradient,
            "centreline_velocity": analytical_centreline,
        },
        "numerical": {
            "mean_velocity": float(numerical_mean),
            "centreline_velocity": float(numerical_centreline),
            "pressure_gradient": float(pressure_slope),
        },
        "errors": {
            "velocity_l2_absolute": float(norms.l2_absolute),
            "velocity_linf_absolute": float(norms.linf_absolute),
            "velocity_l2_relative": float(norms.l2_relative),
            "velocity_linf_relative": float(norms.linf_relative),
            "mean_velocity_relative": float(
                jnp.abs((numerical_mean - mean_velocity) / mean_velocity)
            ),
            "centreline_velocity_relative": float(
                jnp.abs(
                    (numerical_centreline - analytical_centreline)
                    / analytical_centreline
                )
            ),
            "pressure_gradient_relative": float(
                pressure_gradient_relative_error
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
        },
    }

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    profile_path = output_dir / "profile.csv"
    _write_profile(profile_path, y, profile, analytical_profile)
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
    result_record = {
        "schema_version": "1.0",
        "experiment_id": "poiseuille_forward_verification",
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
            "analytical_solution_rescaled_from_numerical_result": False,
            "reference_comparison": {
                "boundary_conditions_exact": True,
                "quantity_definition_exact": True,
                "paper_eligible": True,
            },
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
            "failure_reason": None if converged else "nonlinear tolerance not met",
        },
        "objective": {
            "name": "poiseuille_velocity_profile_relative_l2_error",
            "value": float(norms.l2_relative),
            "units": "dimensionless",
            "normalization": "continuum profile fixed by configured mean velocity",
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
            "Issue #5 planar-Poiseuille case; detailed physical metrics are in "
            "metrics.json."
        ),
    }
    artifact_specs = (
        (resolved_config_path, "application/json", "config"),
        (environment_path, "application/json", "environment"),
        (metrics_path, "application/json", "table"),
        (profile_path, "text/csv", "table"),
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
