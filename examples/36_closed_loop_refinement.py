"""Closed-loop geometry optimization at fixed regularization on nested grids.

Record 31 optimizes the Fourier obstacle on the ``32 x 20`` grid with interface
half-width ``epsilon = 0.075`` and ``alpha_max = 2400``.  This study repeats the
same projected-gradient loop (same objective, bounds, step, tolerances) on the
``64 x 40`` and ``128 x 80`` grids with the *same physical* ``epsilon`` and
``alpha_max``, so that every level discretizes one regularized continuous
problem.  It records the optimized coefficients, the converged mean-flow gain
and the stationarity measure per level, and evaluates every level's optimum on
every other level, which quantifies how much of the design is a grid artefact.

Usage (repository root, ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``)::

    python examples/36_closed_loop_refinement.py --level coarse
    python examples/36_closed_loop_refinement.py --level medium
    python examples/36_closed_loop_refinement.py --level fine
    python examples/36_closed_loop_refinement.py --merge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import steady_residual, steady_residual_norms
from jax_cad.optimization.projected_descent import projected_gradient, projected_gradient_descent
from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import DifferentiableProgram, FourierObstacleBenchmark
from jax_cad.unified.geometry import fourier_boundary_radius, fourier_shape_area

LEVELS = {"coarse": (32, 20), "medium": (64, 40), "fine": (128, 80)}
EPSILON = 0.075
ALPHA_MAX = 2400.0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"


def _benchmark(level: str) -> FourierObstacleBenchmark:
    nx, ny = LEVELS[level]
    probe = FourierObstacleBenchmark(nx=nx, ny=ny)
    return FourierObstacleBenchmark(nx=nx, ny=ny, epsilon_over_h=EPSILON / probe.h, alpha_max=ALPHA_MAX, primal_method="gmres")


def _shape(bench: FourierObstacleBenchmark, coefficients) -> dict[str, object]:
    angles = jnp.array([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi], dtype=jnp.float64)
    radii = fourier_boundary_radius(angles, coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    area = fourier_shape_area(coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
    return {
        "cardinal_radii": np.asarray(radii).tolist(),
        "quadrature_area": float(area),
        "quadrature_area_relative_error": abs(float(area) - np.pi * bench.base_radius**2) / (np.pi * bench.base_radius**2),
    }


def run_level(level: str) -> Path:
    bench = _benchmark(level)
    started = provenance_record()["utc"]
    design_map = bench.design_map()
    solver, primal = bench.implicit_solver()
    program = DifferentiableProgram(design_map, solver, bench.objective)
    physical = DifferentiableProgram(design_map, solver, lambda s, f, c: bench.physical_loss(s))
    value_and_grad = jax.jit(program.value_and_grad)
    value = jax.jit(program.value)
    physical_value = jax.jit(physical.value)
    coefficients0 = np.asarray(bench.zero_coefficients())

    def value_and_grad_host(x):
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g)

    def value_host(x):
        return float(value(jnp.asarray(x)))

    mean_u0 = -float(physical_value(jnp.asarray(coefficients0)))
    print(f"{level}: eps/h={bench.epsilon_over_h:g}, initial mean u {mean_u0:.8f}", flush=True)
    t0 = time.perf_counter()
    result = projected_gradient_descent(
        value_and_grad_host,
        value_host,
        coefficients0,
        lower=-bench.coefficient_bound,
        upper=bench.coefficient_bound,
        initial_step=bench.initial_step,
        max_iterations=bench.max_iterations,
        relative_gradient_tolerance=bench.relative_gradient_tolerance,
        step_growth=2.0,
    )
    loop_seconds = time.perf_counter() - t0
    final = jnp.asarray(result.coefficients)
    fields = design_map(final)
    state, diagnostics = primal(fields.momentum_resistance)
    res_rms, res_max = steady_residual_norms(steady_residual(state, fields.momentum_resistance, bench.grid, bench.params()))
    mean_u1 = float(jnp.mean(state.u))
    v1, g1 = block_tree(value_and_grad(final))
    pg = projected_gradient(np.asarray(final), np.asarray(g1), -bench.coefficient_bound, bench.coefficient_bound)
    accepted_updates = sum(1 for item in result.trace if item.get("accepted"))
    v0, g0 = block_tree(value_and_grad(jnp.asarray(coefficients0)))
    trace = [
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else (np.asarray(v).tolist() if k == "coefficients" else v)) for k, v in item.items()}
        for item in result.trace
    ]
    record = {
        "level": level,
        "configuration": bench.to_dict(),
        "optimization": {
            "iterations": int(result.iterations),
            "accepted_updates": int(accepted_updates),
            "converged": bool(result.converged),
            "termination": result.termination,
            "objective_initial": float(v0),
            "objective_final": float(v1),
            "mean_u_initial": mean_u0,
            "mean_u_final": mean_u1,
            "mean_velocity_increase_percent": 100.0 * (mean_u1 / mean_u0 - 1.0),
            "initial_projected_gradient_norm": float(np.linalg.norm(np.asarray(g0))),
            "final_projected_gradient_norm": float(np.linalg.norm(pg)),
            "projected_gradient_ratio": float(np.linalg.norm(pg) / np.linalg.norm(np.asarray(g0))),
            "final_coefficients": np.asarray(final).tolist(),
            "active_bounds": [bool(abs(c) >= bench.coefficient_bound - 1e-12) for c in np.asarray(final)],
            "loop_seconds": loop_seconds,
            "trace": trace,
        },
        "final_shape": _shape(bench, final),
        "final_state": {
            "primal_residual": float(diagnostics.residual_norm),
            "steady_residual_rms": float(res_rms),
            "steady_residual_max": float(res_max),
        },
        "provenance": {**provenance_record(), "start_utc": started, "script": "examples/36_closed_loop_refinement.py"},
    }
    path = DATA_DIR / f"36_closed_loop_refinement_{level}.json"
    write_json(path, record)
    print(
        f"{level}: {result.iterations} iterations, {accepted_updates} accepted, gain {record['optimization']['mean_velocity_increase_percent']:.2f}%, "
        f"pg ratio {record['optimization']['projected_gradient_ratio']:.2e}, {loop_seconds:.0f}s",
        flush=True,
    )
    print(f"Saved: {path}")
    return path


def merge() -> Path:
    records = {}
    for level in LEVELS:
        path = DATA_DIR / f"36_closed_loop_refinement_{level}.json"
        if path.exists():
            records[level] = json.loads(path.read_text(encoding="utf-8"))
    levels = [level for level in LEVELS if level in records]
    # Cross-evaluation: every level's optimum evaluated (converged) on every level.
    cross = []
    for eval_level in levels:
        bench = _benchmark(eval_level)
        design_map = bench.design_map()
        solver, primal = bench.implicit_solver()
        physical_value = jax.jit(DifferentiableProgram(design_map, solver, lambda s, f, c: bench.physical_loss(s)).value)
        mean_u0 = -float(physical_value(bench.zero_coefficients()))
        for opt_level in levels:
            c = jnp.asarray(records[opt_level]["optimization"]["final_coefficients"], dtype=jnp.float64)
            mean_u = -float(physical_value(c))
            cross.append(
                {
                    "optimized_on": opt_level,
                    "evaluated_on": eval_level,
                    "mean_u": mean_u,
                    "gain_percent": 100.0 * (mean_u / mean_u0 - 1.0),
                }
            )
            print(f"optimum of {opt_level} evaluated on {eval_level}: gain {cross[-1]['gain_percent']:.3f}%", flush=True)
    coefficient_differences = []
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            ca = np.asarray(records[a]["optimization"]["final_coefficients"])
            cb = np.asarray(records[b]["optimization"]["final_coefficients"])
            coefficient_differences.append(
                {
                    "levels": [a, b],
                    "l2_difference": float(np.linalg.norm(ca - cb)),
                    "l2_difference_relative_to_bound": float(np.linalg.norm(ca - cb) / (0.30 * np.sqrt(len(ca)))),
                    "max_abs_difference": float(np.max(np.abs(ca - cb))),
                    "cardinal_radii_difference": (np.asarray(records[a]["final_shape"]["cardinal_radii"]) - np.asarray(records[b]["final_shape"]["cardinal_radii"])).tolist(),
                }
            )
    record = {
        "claim": (
            "The projected-gradient closed loop at fixed interface width and penalization reaches consistent optimized shapes "
            "on nested grids; each level's optimum is evaluated on every other level."
        ),
        "levels": {level: LEVELS[level] for level in levels},
        "epsilon": EPSILON,
        "alpha_max": ALPHA_MAX,
        "per_level": {
            level: {
                "epsilon_over_h": records[level]["configuration"]["epsilon_over_h"],
                "iterations": records[level]["optimization"]["iterations"],
                "accepted_updates": records[level]["optimization"]["accepted_updates"],
                "mean_velocity_increase_percent": records[level]["optimization"]["mean_velocity_increase_percent"],
                "projected_gradient_ratio": records[level]["optimization"]["projected_gradient_ratio"],
                "final_coefficients": records[level]["optimization"]["final_coefficients"],
                "active_bounds": records[level]["optimization"]["active_bounds"],
                "cardinal_radii": records[level]["final_shape"]["cardinal_radii"],
                "loop_seconds": records[level]["optimization"]["loop_seconds"],
            }
            for level in levels
        },
        "cross_evaluation": cross,
        "coefficient_differences": coefficient_differences,
        "provenance_by_level": {level: records[level]["provenance"] for level in levels},
        "merged_provenance": provenance_record(),
    }
    path = DATA_DIR / "36_closed_loop_refinement.json"
    write_json(path, record)
    print(f"Saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=list(LEVELS))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.level:
        run_level(args.level)
    if args.merge:
        merge()
    if not args.level and not args.merge:
        parser.error("specify --level or --merge")


if __name__ == "__main__":
    main()
