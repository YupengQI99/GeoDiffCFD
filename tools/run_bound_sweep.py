#!/usr/bin/env python3
"""Coefficient-bound sweep of the periodic Fourier-obstacle closed loop.

Repeats the closed-loop part of ``examples/31_steady_result_to_geometry.py``
with the same design map, implicit fixed-point ``value_and_grad`` and
``projected_gradient_descent`` settings (normalized projected steepest
descent, Armijo backtracking, initial step 0.04, step growth 2, relative
projected-gradient tolerance 1e-3, at most 60 iterations). Only the box bound
is changed, via ``dataclasses.replace(bench, coefficient_bound=b)``.

Each bound runs in its own process and is merged into the output file:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tools/run_bound_sweep.py --bound 0.3
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tools/run_bound_sweep.py --bound 0.2 ...

``--max-iterations N`` raises the iteration cap of the same run; such runs are
stored under the key ``"<b>_maxit<N>"``. The archived sweep is
``runs/periodic_bound_sweep/results.json``; the default output is a new file so
that the archive is not overwritten.

``--bound 0.3`` additionally compares the result with
``output/data/31_steady_result_to_geometry.json`` and exits with status 2 if
the reproduction is not within ``REPRO_RTOL``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jax_cad.optimization.projected_descent import (  # noqa: E402
    projected_gradient,
    projected_gradient_descent,
)
from jax_cad.unified import (  # noqa: E402
    DifferentiableProgram,
    FourierObstacleBenchmark,
    fourier_boundary_radius,
    fourier_shape_area,
)

DEFAULT_OUTPUT = ROOT / "output" / "bound_sweep_results.json"
ARCHIVED = ROOT / "output" / "data" / "31_steady_result_to_geometry.json"
REPRO_RTOL = 1e-6
SOURCE_FILES = (
    "jax_cad/unified/flagship.py",
    "jax_cad/unified/geometry.py",
    "jax_cad/unified/implicit.py",
    "jax_cad/unified/materials.py",
    "jax_cad/unified/problem.py",
    "jax_cad/differentiable/mac_periodic.py",
    "jax_cad/optimization/projected_descent.py",
    "examples/31_steady_result_to_geometry.py",
)


# --------------------------------------------------------------- provenance
def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance() -> dict[str, object]:
    import jaxlib
    import scipy

    return {
        "repository_git_commit": _git_commit(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
        "driver": "tools/run_bound_sweep.py",
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------- geometry
COEFFICIENT_NAMES = ("a2", "a3", "a4", "a5", "b2", "b3", "b4", "b5")


def _log_radius_slope(coefficients: np.ndarray, modes, n: int = 16384) -> tuple[float, float]:
    """max |r'/r| = max |f'(theta)| of the log-radius expansion on a dense angle grid."""

    theta = 2.0 * np.pi * (np.arange(n) + 0.5) / n
    k = len(modes)
    a, b = coefficients[:k], coefficients[k:]
    m = np.asarray(modes, dtype=np.float64)
    phase = theta[:, None] * m
    slope = np.sum(m * (-a * np.sin(phase) + b * np.cos(phase)), axis=1)
    i = int(np.argmax(np.abs(slope)))
    return float(abs(slope[i])), float(theta[i])


def _cardinal_radii(bench: FourierObstacleBenchmark, coefficients) -> list[float]:
    angles = jnp.array([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi], dtype=jnp.float64)
    radii = fourier_boundary_radius(
        angles, jnp.asarray(coefficients), base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096
    )
    return np.asarray(radii).tolist()


# --------------------------------------------------------------- one bound
def run_bound(bound: float, max_iterations: int | None = None) -> dict[str, object]:
    bench = dataclasses.replace(FourierObstacleBenchmark(), coefficient_bound=bound)
    if max_iterations is not None:
        bench = dataclasses.replace(bench, max_iterations=max_iterations)
    design_map = bench.design_map()
    implicit_solver, primal = bench.implicit_solver()
    implicit_program = DifferentiableProgram(design_map, implicit_solver, bench.objective)
    implicit_value_and_grad = jax.jit(implicit_program.value_and_grad)
    implicit_value = jax.jit(implicit_program.value)
    coefficients0 = bench.zero_coefficients()

    def value_and_grad_host(x):
        value, gradient = implicit_value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(gradient)

    def value_host(x):
        return float(implicit_value(jnp.asarray(x)))

    started = time.perf_counter()
    result = projected_gradient_descent(
        value_and_grad_host,
        value_host,
        np.asarray(coefficients0),
        lower=-bench.coefficient_bound,
        upper=bench.coefficient_bound,
        initial_step=bench.initial_step,
        max_iterations=bench.max_iterations,
        relative_gradient_tolerance=bench.relative_gradient_tolerance,
        step_growth=2.0,
    )
    loop_seconds = time.perf_counter() - started

    final = np.asarray(result.coefficients)
    value_f, grad_f = implicit_value_and_grad(jnp.asarray(final))
    value_f = float(value_f)
    grad_f = np.asarray(grad_f)
    pg_f = projected_gradient(final, grad_f, -bound, bound)
    first = result.trace[0]
    state0, diag0 = primal(design_map(coefficients0).momentum_resistance)
    state_f, diag_f = primal(design_map(jnp.asarray(final)).momentum_resistance)
    mean_u0 = float(jnp.mean(state0.u))
    mean_uf = float(jnp.mean(state_f.u))
    slope_max, slope_angle = _log_radius_slope(final, bench.modes)
    active = [
        {"name": n, "value": float(c), "at": "+b" if c > 0 else "-b"}
        for n, c in zip(COEFFICIENT_NAMES, final)
        if abs(c) >= bound - 1e-12
    ]
    area0 = float(fourier_shape_area(coefficients0, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096))
    areaf = float(
        fourier_shape_area(jnp.asarray(final), base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    )
    accepted = [item for item in result.trace if item["accepted"]]
    return {
        "coefficient_bound": bound,
        "configuration_changes": {"coefficient_bound": bound}
        | ({} if max_iterations is None else {"max_iterations": max_iterations}),
        "settings": {
            "initial_step": bench.initial_step,
            "max_iterations": bench.max_iterations,
            "relative_gradient_tolerance": bench.relative_gradient_tolerance,
            "step_growth": 2.0,
            "grid": [bench.nx, bench.ny],
            "primal_method": bench.primal_method,
        },
        "termination": result.termination,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "accepted_updates": len(accepted),
        "J_initial": float(first["objective"]),
        "J_final": value_f,
        "objective_reduction_percent": 100.0 * (float(first["objective"]) - value_f) / abs(float(first["objective"])),
        "initial_mean_streamwise_velocity": mean_u0,
        "final_mean_streamwise_velocity": mean_uf,
        "mean_flow_gain_percent": 100.0 * (mean_uf / mean_u0 - 1.0),
        "initial_projected_gradient_norm": float(result.initial_projected_gradient_norm),
        "final_projected_gradient_norm_loop": float(result.final_projected_gradient_norm),
        "final_projected_gradient_norm": float(np.linalg.norm(pg_f)),
        "final_projected_gradient_ratio": float(np.linalg.norm(pg_f) / result.initial_projected_gradient_norm),
        "initial_gradient_norm": float(first["gradient_norm"]),
        "final_gradient_norm": float(np.linalg.norm(grad_f)),
        "full_gradient_norm_ratio": float(np.linalg.norm(grad_f) / first["gradient_norm"]),
        "final_gradient": grad_f.tolist(),
        "final_coefficients": final.tolist(),
        "coefficient_names": list(COEFFICIENT_NAMES),
        "active_bounds": active,
        "active_bound_mask": [bool(abs(c) >= bound - 1e-12) for c in final],
        "cardinal_radii_0_pi2_pi_3pi2": _cardinal_radii(bench, final),
        "r0": _cardinal_radii(bench, final)[0],
        "r_pi_over_2": _cardinal_radii(bench, final)[1],
        "max_abs_rprime_over_r": slope_max,
        "max_abs_rprime_over_r_angle": slope_angle,
        "relative_area_error": abs(areaf - area0) / area0,
        "initial_primal_residual": float(diag0.residual_norm),
        "final_primal_residual": float(diag_f.residual_norm),
        "loop_wall_seconds": loop_seconds,
        "trace": [
            {k: item[k] for k in ("step", "objective", "gradient_norm", "projected_gradient_norm",
                                  "accepted", "step_length", "backtracks", "coefficients")}
            for item in result.trace
        ],
    }


def check_archived(entry: dict[str, object]) -> dict[str, object]:
    archived = json.loads(ARCHIVED.read_text())
    opt = archived["optimization"]
    ref = {
        "J_final": opt["final_objective"],
        "J_initial": opt["initial_objective"],
        "mean_flow_gain_percent": opt["mean_velocity_increase_percent"],
        "final_projected_gradient_norm": opt["final_projected_gradient_norm"],
        "final_gradient_norm": opt["final_gradient_norm"],
    }
    rel = {k: abs(entry[k] - v) / abs(v) for k, v in ref.items()}
    coeff_diff = float(np.max(np.abs(np.asarray(entry["final_coefficients"]) - np.asarray(opt["final_coefficients"]))))
    passed = (
        rel["J_final"] <= REPRO_RTOL
        and rel["mean_flow_gain_percent"] <= 10 * REPRO_RTOL
        and entry["accepted_updates"] == opt["accepted_updates"]
    )
    return {
        "archived_record": ARCHIVED.relative_to(ROOT).as_posix(),
        "archived_git_commit": archived["provenance"]["git_commit"],
        "archived_jax": archived["provenance"]["jax"],
        "archived": ref | {"accepted_updates": opt["accepted_updates"], "iterations": opt["iterations"]},
        "relative_differences": rel,
        "max_abs_coefficient_difference": coeff_diff,
        "tolerance": REPRO_RTOL,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    wall = time.perf_counter()
    entry = run_bound(args.bound, args.max_iterations)
    entry["wall_seconds"] = time.perf_counter() - wall
    entry["peak_host_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    entry["provenance"] = _provenance()
    if abs(args.bound - 0.3) < 1e-12 and args.max_iterations is None:
        entry["reproduction_check"] = check_archived(entry)

    record = json.loads(args.output.read_text()) if args.output.exists() else {}
    record.setdefault(
        "description",
        "Coefficient-bound sweep of the periodic Fourier-obstacle closed loop (32x20, implicit "
        "fixed-point backend, projected steepest descent with Armijo, relative projected-gradient "
        "tolerance 1e-3, initial step 0.04); only coefficient_bound differs from the archived run.",
    )
    record.setdefault("runs", {})
    key = f"{args.bound:.2f}" + ("" if args.max_iterations is None else f"_maxit{args.max_iterations}")
    record["runs"][key] = entry
    record["runs"] = dict(sorted(record["runs"].items()))
    args.output.write_text(json.dumps(record, indent=1) + "\n")

    summary = {k: entry[k] for k in (
        "coefficient_bound", "termination", "iterations", "accepted_updates", "J_initial", "J_final",
        "mean_flow_gain_percent", "final_projected_gradient_ratio", "full_gradient_norm_ratio",
        "active_bounds", "r0", "r_pi_over_2", "max_abs_rprime_over_r", "final_coefficients", "wall_seconds")}
    print(json.dumps(summary, indent=1))
    if "reproduction_check" in entry:
        print(json.dumps(entry["reproduction_check"], indent=1))
        if not entry["reproduction_check"]["passed"]:
            sys.exit(2)


if __name__ == "__main__":
    main()
