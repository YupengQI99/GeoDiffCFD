"""Verify the conservative non-periodic Navier--Stokes--Brinkman kernel.

This is a lightweight CPU development gate for issue #4. It is not a
production benchmark and does not replace the canonical validations in issue
#5 or residual-based implicit differentiation in issue #6.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    cell_centres,
    create_channel_grid,
    create_initial_state,
    flux_diagnostics,
    parabolic_inlet,
    residual_max,
    residual_rms,
    simulate,
    solve_steady,
    steady_residual,
)


def _block_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def _compile_and_sample(function, samples: int = 2):
    started = time.perf_counter()
    compiled = jax.jit(function).lower().compile()
    compile_seconds = time.perf_counter() - started
    durations = []
    result = None
    for _ in range(samples):
        started = time.perf_counter()
        result = _block_tree(compiled())
        durations.append(time.perf_counter() - started)
    return result, compile_seconds, durations


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tracked_tree_is_dirty() -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"], check=False
    ).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], check=False
    ).returncode
    return unstaged != 0 or staged != 0


def _untracked_count() -> int:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(line.startswith("??") for line in result.stdout.splitlines())


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_summary(
    state,
    diagnostics,
    grid,
    params,
    mean_velocity: float,
    alpha_max: float,
    interface_width: float,
    compile_seconds: float,
    execution_samples: list[float],
) -> dict:
    div_rms, div_max, inlet_flux, outlet_flux, mismatch = flux_diagnostics(
        state, grid
    )
    iterations = int(diagnostics.iterations)
    final_index = max(iterations - 1, 0)
    threshold = max(
        params.nonlinear_atol,
        params.nonlinear_rtol * float(diagnostics.initial_residual_rms),
    )
    return {
        "configuration": {
            "grid": [grid.nx, grid.ny],
            "domain": [grid.lx, grid.ly],
            "spacing": [grid.dx, grid.dy],
            "mean_inlet_velocity": mean_velocity,
            "reynolds_number": mean_velocity * grid.ly / params.viscosity,
            "viscosity": params.viscosity,
            "time_step": params.dt,
            "alpha_max": alpha_max,
            "interface_width": interface_width,
            "pressure_tolerance": params.pressure_tolerance,
            "pressure_absolute_tolerance": params.pressure_atol,
            "pressure_max_iterations": params.pressure_maxiter,
            "nonlinear_relative_tolerance": params.nonlinear_rtol,
            "nonlinear_absolute_tolerance": params.nonlinear_atol,
            "precision": "float64",
            "gradient_backend": "none",
        },
        "convergence": {
            "criterion": "residual_rms <= max(atol, rtol * initial_rms)",
            "threshold": threshold,
            "converged": bool(diagnostics.converged),
            "iterations": iterations,
            "initial_residual_rms": float(diagnostics.initial_residual_rms),
            "final_residual_rms": float(diagnostics.final_residual_rms),
            "final_residual_max": float(diagnostics.final_residual_max),
            "final_pressure_residual_absolute": float(
                diagnostics.pressure_residual_abs_history[final_index]
            ),
            "final_pressure_residual_relative": float(
                diagnostics.pressure_residual_rel_history[final_index]
            ),
        },
        "conservation": {
            "divergence_rms": float(div_rms),
            "divergence_max": float(div_max),
            "inlet_mass_flux": float(inlet_flux),
            "outlet_mass_flux": float(outlet_flux),
            "relative_mass_flux_mismatch": float(mismatch),
        },
        "objective": {
            "name": "pressure_drop",
            "definition": (
                "arithmetic mean of the inlet-adjacent cell-centred pressure "
                "relative to the prescribed zero-pressure outlet face"
            ),
            "value": float(jnp.mean(state.p[0, :])),
            "units": "nondimensional",
        },
        "timing": {
            "compile_seconds": compile_seconds,
            "execution_samples_seconds": execution_samples,
            "synchronization": "all returned JAX leaves blocked until ready",
        },
    }


def _history_rows(case_name: str, diagnostics):
    fields = {
        "nonlinear_residual_rms": np.asarray(
            diagnostics.nonlinear_residual_rms_history
        ),
        "nonlinear_residual_max": np.asarray(
            diagnostics.nonlinear_residual_max_history
        ),
        "pressure_residual_absolute": np.asarray(
            diagnostics.pressure_residual_abs_history
        ),
        "pressure_residual_relative": np.asarray(
            diagnostics.pressure_residual_rel_history
        ),
        "divergence_rms": np.asarray(diagnostics.divergence_rms_history),
        "divergence_max": np.asarray(diagnostics.divergence_max_history),
        "inlet_mass_flux": np.asarray(diagnostics.inlet_flux_history),
        "outlet_mass_flux": np.asarray(diagnostics.outlet_flux_history),
        "relative_mass_flux_mismatch": np.asarray(
            diagnostics.mass_flux_mismatch_history
        ),
    }
    for index in range(int(diagnostics.iterations)):
        yield {
            "case": case_name,
            "iteration": index + 1,
            **{name: float(values[index]) for name, values in fields.items()},
        }



def main() -> None:
    started_utc = datetime.now(timezone.utc)
    tracked_dirty_at_start = _tracked_tree_is_dirty()
    untracked_count_at_start = _untracked_count()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "27_nonperiodic_navier_stokes_verification.json"
    csv_path = output_dir / "27_nonperiodic_navier_stokes_residual_history.csv"

    clear_grid = create_channel_grid(12, 8, 1.5, 1.0)
    clear_mean_velocity = 0.2
    clear_inlet = parabolic_inlet(
        clear_grid, clear_mean_velocity, dtype=jnp.float64
    )
    clear_state0 = create_initial_state(
        clear_grid, clear_inlet, dtype=jnp.float64
    )
    clear_alpha = jnp.zeros(
        (clear_grid.nx, clear_grid.ny), dtype=jnp.float64
    )
    clear_params = ChannelParams(
        viscosity=0.2,
        dt=0.005,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=1.0e-6,
        nonlinear_atol=1.0e-8,
    )
    clear_result, clear_compile, clear_samples = _compile_and_sample(
        lambda: solve_steady(
            clear_state0,
            clear_alpha,
            clear_inlet,
            clear_grid,
            clear_params,
            max_steps=500,
        )
    )
    clear_state, clear_diagnostics = clear_result
    clear_report = _case_summary(
        clear_state,
        clear_diagnostics,
        clear_grid,
        clear_params,
        clear_mean_velocity,
        alpha_max=0.0,
        interface_width=0.0,
        compile_seconds=clear_compile,
        execution_samples=clear_samples,
    )

    blockage_grid = create_channel_grid(18, 12, 2.0, 1.0)
    blockage_mean_velocity = 0.12
    blockage_inlet = parabolic_inlet(
        blockage_grid, blockage_mean_velocity, dtype=jnp.float64
    )
    blockage_state0 = create_initial_state(
        blockage_grid, blockage_inlet, dtype=jnp.float64
    )
    blockage_x, blockage_y = cell_centres(
        blockage_grid, dtype=jnp.float64
    )
    slab = jnp.abs(blockage_x - 1.0) < 1.1 * blockage_grid.dx
    gap = jnp.abs(blockage_y - 0.5) < 1.1 * blockage_grid.dy
    blockage_alpha_max = 2.0e4
    blockage_alpha = blockage_alpha_max * (slab & ~gap)
    blockage_params = ChannelParams(
        viscosity=0.12,
        dt=0.003,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=2.0e-6,
        nonlinear_atol=1.0e-9,
    )
    blockage_result, blockage_compile, blockage_samples = _compile_and_sample(
        lambda: solve_steady(
            blockage_state0,
            blockage_alpha,
            blockage_inlet,
            blockage_grid,
            blockage_params,
            max_steps=700,
        )
    )
    blockage_state, blockage_diagnostics = blockage_result
    blockage_report = _case_summary(
        blockage_state,
        blockage_diagnostics,
        blockage_grid,
        blockage_params,
        blockage_mean_velocity,
        alpha_max=blockage_alpha_max,
        interface_width=0.0,
        compile_seconds=blockage_compile,
        execution_samples=blockage_samples,
    )
    barrier_face = blockage_grid.nx // 2
    blockage_y_faces = (
        jnp.arange(blockage_grid.ny, dtype=jnp.float64) + 0.5
    ) * blockage_grid.dy
    solid_rows = jnp.abs(blockage_y_faces - 0.5) >= 1.1 * blockage_grid.dy
    blockage_inlet_flux = blockage_report["conservation"]["inlet_mass_flux"]
    leakage_fraction = float(
        jnp.sum(jnp.abs(blockage_state.u[barrier_face, solid_rows]))
        * blockage_grid.dy
        / abs(blockage_inlet_flux)
    )
    resolved_gap_fraction = float(
        jnp.sum(jnp.abs(blockage_state.u[barrier_face, ~solid_rows]))
        * blockage_grid.dy
        / abs(blockage_inlet_flux)
    )
    blockage_report["blockage_resolution"] = {
        "barrier_face_index": barrier_face,
        "resolved_gap_cells": int(jnp.sum(~solid_rows)),
        "gap_width": float(jnp.sum(~solid_rows) * blockage_grid.dy),
        "solid_penetration_flux_fraction": leakage_fraction,
        "resolved_gap_flux_fraction": resolved_gap_fraction,
        "interpretation": (
            "flux is measured on the face inside the two-cell-thick "
            "Brinkman barrier; no sub-cell gap is credited"
        ),
    }


    gradient_grid = create_channel_grid(8, 5, 1.6, 1.0)
    gradient_mean_velocity = 0.2
    gradient_inlet = parabolic_inlet(
        gradient_grid, gradient_mean_velocity, dtype=jnp.float64
    )
    gradient_state0 = create_initial_state(
        gradient_grid, gradient_inlet, dtype=jnp.float64
    )
    gradient_x, gradient_y = cell_centres(
        gradient_grid, dtype=jnp.float64
    )
    gradient_params = ChannelParams(
        viscosity=0.15,
        dt=0.005,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
    )
    interface_width = 0.7 * min(gradient_grid.dx, gradient_grid.dy)
    gradient_alpha_max = 50.0

    def resistance(theta):
        cx, cy, radius = theta
        phi = (
            jnp.sqrt(
                (gradient_x - cx) ** 2
                + (gradient_y - cy) ** 2
                + 1.0e-30
            )
            - radius
        )
        return gradient_alpha_max * jax.nn.sigmoid(-phi / interface_width)

    def objective_with_aux(theta):
        alpha = resistance(theta)
        state, projection = simulate(
            gradient_state0,
            alpha,
            gradient_inlet,
            gradient_grid,
            gradient_params,
            n_steps=400,
        )
        residual = steady_residual(
            state, alpha, gradient_inlet, gradient_grid, gradient_params
        )
        metrics = jnp.asarray(
            [
                residual_rms(residual),
                residual_max(residual),
                projection.pressure_residual_abs,
                projection.pressure_residual_rel,
                projection.divergence_rms,
                projection.divergence_max,
                projection.inlet_flux,
                projection.outlet_flux,
                projection.mass_flux_mismatch,
            ]
        )
        return jnp.mean(state.p[0, :]), metrics

    theta = jnp.array([0.75, 0.51, 0.14], dtype=jnp.float64)
    direction = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    gradient_function = jax.jit(
        jax.value_and_grad(objective_with_aux, has_aux=True)
    )
    started = time.perf_counter()
    compiled_gradient = gradient_function.lower(theta).compile()
    gradient_compile_seconds = time.perf_counter() - started
    gradient_samples = []
    gradient_result = None
    for _ in range(2):
        started = time.perf_counter()
        gradient_result = _block_tree(compiled_gradient(theta))
        gradient_samples.append(time.perf_counter() - started)
    (objective_value, gradient_metrics), gradient = gradient_result
    directional_derivative = jnp.vdot(gradient, direction)
    epsilons = jnp.array(
        [4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4], dtype=jnp.float64
    )
    remainders = []
    for epsilon in epsilons:
        candidate_result = _block_tree(
            compiled_gradient(theta + epsilon * direction)
        )
        candidate_value = candidate_result[0][0]
        remainders.append(
            jnp.abs(
                candidate_value
                - objective_value
                - epsilon * directional_derivative
            )
        )
    remainders = jnp.asarray(remainders)
    rates = jnp.log(remainders[:-1] / remainders[1:]) / jnp.log(
        epsilons[:-1] / epsilons[1:]
    )
    gradient_metrics = np.asarray(gradient_metrics)
    gradient_report = {
        "configuration": {
            "grid": [gradient_grid.nx, gradient_grid.ny],
            "domain": [gradient_grid.lx, gradient_grid.ly],
            "spacing": [gradient_grid.dx, gradient_grid.dy],
            "mean_inlet_velocity": gradient_mean_velocity,
            "reynolds_number": (
                gradient_mean_velocity
                * gradient_grid.ly
                / gradient_params.viscosity
            ),
            "viscosity": gradient_params.viscosity,
            "time_step": gradient_params.dt,
            "fixed_steps": 400,
            "alpha_max": gradient_alpha_max,
            "interface_width": interface_width,
            "pressure_tolerance": gradient_params.pressure_tolerance,
            "pressure_absolute_tolerance": gradient_params.pressure_atol,
            "precision": "float64",
        },
        "state": {
            "final_residual_rms": float(gradient_metrics[0]),
            "final_residual_max": float(gradient_metrics[1]),
            "pressure_residual_absolute": float(gradient_metrics[2]),
            "pressure_residual_relative": float(gradient_metrics[3]),
            "divergence_rms": float(gradient_metrics[4]),
            "divergence_max": float(gradient_metrics[5]),
            "inlet_mass_flux": float(gradient_metrics[6]),
            "outlet_mass_flux": float(gradient_metrics[7]),
            "relative_mass_flux_mismatch": float(gradient_metrics[8]),
        },
        "objective": {
            "name": "pressure_drop",
            "definition": (
                "arithmetic mean of the inlet-adjacent cell-centred pressure "
                "relative to the prescribed zero-pressure outlet face"
            ),
            "value": float(objective_value),
            "units": "nondimensional",
        },
        "gradient": {
            "backend": "reverse-mode AD through a fixed 400-step JAX scan",
            "call": "one jax.value_and_grad call for the base gradient",
            "parameterization": "analytic circle SDF mapped by a sigmoid",
            "parameter_names": ["centre_x", "centre_y", "radius"],
            "parameters": np.asarray(theta).tolist(),
            "values": np.asarray(gradient).tolist(),
            "norm": float(jnp.linalg.norm(gradient)),
            "direction": np.asarray(direction).tolist(),
            "directional_derivative": float(directional_derivative),
            "taylor_test": {
                "epsilons": np.asarray(epsilons).tolist(),
                "first_order_remainders": np.asarray(remainders).tolist(),
                "first_order_rates": np.asarray(rates).tolist(),
                "minimum_asymptotic_rate": float(jnp.min(rates[1:])),
            },
        },
        "timing": {
            "compile_seconds": gradient_compile_seconds,
            "gradient_execution_samples_seconds": gradient_samples,
            "synchronization": "all returned JAX leaves blocked until ready",
        },
    }

    history_fieldnames = [
        "case",
        "iteration",
        "nonlinear_residual_rms",
        "nonlinear_residual_max",
        "pressure_residual_absolute",
        "pressure_residual_relative",
        "divergence_rms",
        "divergence_max",
        "inlet_mass_flux",
        "outlet_mass_flux",
        "relative_mass_flux_mismatch",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=history_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(_history_rows("unobstructed", clear_diagnostics))
        writer.writerows(
            _history_rows("near_blockage", blockage_diagnostics)
        )

    clear_threshold = clear_report["convergence"]["threshold"]
    blockage_threshold = blockage_report["convergence"]["threshold"]
    checks = {
        "unobstructed_converged": clear_report["convergence"]["converged"],
        "unobstructed_residual_met": (
            clear_report["convergence"]["final_residual_rms"]
            <= clear_threshold * (1.0 + 1.0e-10)
        ),
        "unobstructed_mass_balance": (
            clear_report["conservation"]["relative_mass_flux_mismatch"]
            < 1.0e-10
        ),
        "near_blockage_converged": blockage_report["convergence"]["converged"],
        "near_blockage_residual_met": (
            blockage_report["convergence"]["final_residual_rms"]
            <= blockage_threshold * (1.0 + 1.0e-10)
        ),
        "near_blockage_mass_balance": (
            blockage_report["conservation"]["relative_mass_flux_mismatch"]
            < 1.0e-10
        ),
        "solid_penetration_below_two_percent": leakage_fraction < 0.02,
        "resolved_gap_carries_flux": resolved_gap_fraction > 0.95,
        "gradient_state_residual_below_1e-8": bool(
            gradient_metrics[0] < 1.0e-8
        ),
        "taylor_asymptotic_rate_above_1.8": float(jnp.min(rates[1:])) > 1.8,
    }
    status = "success" if all(checks.values()) else "failed"

    ended_utc = datetime.now(timezone.utc)
    selected_environment = {
        name: os.environ[name]
        for name in (
            "JAX_ENABLE_X64",
            "JAX_PLATFORMS",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
        if name in os.environ
    }
    branch = _git_text("branch", "--show-current") or None
    report = {
        "schema_version": "issue-4-verification-1.0",
        "scope": (
            "Lightweight CPU acceptance evidence for the conservative "
            "non-periodic Navier--Stokes--Brinkman development kernel."
        ),
        "eligibility": {
            "paper_production_result": False,
            "reason": (
                "Poiseuille, cavity, and cylinder validation in issue #5 and "
                "residual-based implicit differentiation in issue #6 remain open."
            ),
        },
        "status": status,
        "acceptance_checks": checks,
        "provenance": {
            "repository": _git_text("remote", "get-url", "origin"),
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": branch,
            "tracked_or_staged_tree_dirty_at_start": tracked_dirty_at_start,
            "untracked_paths_excluded_from_dirty_flag": True,
            "untracked_path_count_at_start": untracked_count_at_start,
            "command": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd()),
            "start_utc": started_utc.isoformat(),
            "end_utc": ended_utc.isoformat(),
            "wall_seconds": (ended_utc - started_utc).total_seconds(),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "backend": jax.default_backend(),
            "precision": "float64",
            "environment_variables": selected_environment,
            "peak_memory": {
                "host_bytes": None,
                "accelerator_bytes": None,
                "measurement_method": "not measured for lightweight CPU gate",
            },
        },
        "discretization": {
            "equations": "steady incompressible Navier--Stokes--Brinkman",
            "layout": "cell pressure with face-normal velocities on a 2-D MAC grid",
            "advection": "first-order donor-cell conservative momentum face flux",
            "diffusion": "second-order centred viscous Laplacian",
            "brinkman": "face-interpolated resistance treated by local implicit mobility",
            "projection": "compatible divergence D, mobility M, and gradient G",
            "pressure_operator": "matrix-free symmetric positive definite -D M G",
            "pressure_gauge": "zero pressure prescribed at the outlet face",
            "boundary_conditions": {
                "inlet": "prescribed discrete parabolic mass flux",
                "walls": "no penetration and no slip",
                "outlet": "zero pressure with velocity zero-gradient closure",
            },
        },
        "cases": {
            "unobstructed": clear_report,
            "near_blockage": blockage_report,
        },
        "gradient_verification": gradient_report,
        "artifacts": [
            {
                "path": csv_path.relative_to(root).as_posix(),
                "sha256": _sha256(csv_path),
                "media_type": "text/csv",
                "role": "residual-history",
            }
        ],
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    if status != "success":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit("Verification failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
