"""Joint grid / interface-width / penalization study of the geometry gradient.

For the committed Fourier-obstacle benchmark the converged steady objective and
its implicit derivative with respect to the native geometry coefficients are
computed on three nested grids, three interface widths ``epsilon/h`` and four
penalization strengths ``alpha_max``, for the reference circle and for a fixed
deformed test shape.  The record separates three questions that are often
conflated: whether the discrete derivative is correct (Taylor tests at every
level), whether objective and derivative converge under grid refinement at fixed
material parameters, and how sensitive both are to the material parameters.

Usage (repository root, ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``)::

    python examples/32_geometry_gradient_convergence.py --level coarse
    python examples/32_geometry_gradient_convergence.py --level medium
    python examples/32_geometry_gradient_convergence.py --level fine
    python examples/32_geometry_gradient_convergence.py --merge
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
from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    FourierObstacleBenchmark,
    directional_taylor_test,
    normalize_direction,
)

LEVELS = {"coarse": (32, 20), "medium": (64, 40), "fine": (128, 80)}
EPSILON_OVER_H = (1.0, 1.5, 2.0)
ALPHA_MAX = (600.0, 2400.0, 9600.0)
SHAPES = {
    "circle": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "deformed": (0.15, 0.0, 0.05, 0.0, 0.0, -0.03, 0.0, 0.02),
}
TAYLOR_DIRECTION = (0.30, -0.40, 0.20, 0.10, -0.25, 0.15, 0.12, -0.08)
TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)
TAYLOR_POINT = {"epsilon_over_h": 1.5, "alpha_max": 2400.0}
# The restarted-GMRES fixed-point primal stagnates on the fine grid at weak
# penalization (alpha_max = 600); the direct symmetric saddle-point solve of the
# same discrete system converges there in seconds, and vice versa at strong
# penalization.  Both are converged to 1e-12 and agree to 1e-9 in the state
# (record 31, direct_saddle_point_cross_check); the method is stored per entry.
PRIMAL_METHOD_BY_ALPHA = {600.0: "saddle", 2400.0: "gmres", 9600.0: "gmres"}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"


def _physical_objective(bench: FourierObstacleBenchmark):
    def objective(state, fields, coefficients):
        del fields, coefficients
        return bench.physical_loss(state)

    return objective


def run_level(level: str) -> Path:
    nx, ny = LEVELS[level]
    direction = normalize_direction(jnp.asarray(TAYLOR_DIRECTION, dtype=jnp.float64))
    started = provenance_record()["utc"]
    entries = []
    for epsilon_over_h in EPSILON_OVER_H:
        for alpha_max in ALPHA_MAX:
            bench = FourierObstacleBenchmark(
                nx=nx, ny=ny, epsilon_over_h=epsilon_over_h, alpha_max=alpha_max,
                primal_method=PRIMAL_METHOD_BY_ALPHA[alpha_max],
                primal_maxiter=4_000 if PRIMAL_METHOD_BY_ALPHA[alpha_max] == "saddle" else 40_000,
                primal_restart=200 if PRIMAL_METHOD_BY_ALPHA[alpha_max] == "saddle" else 60,
            )
            design_map = bench.design_map()
            solver, primal = bench.implicit_solver()
            program = DifferentiableProgram(design_map, solver, _physical_objective(bench))
            value_and_grad = jax.jit(program.value_and_grad)
            for shape, coefficients in SHAPES.items():
                c = jnp.asarray(coefficients, dtype=jnp.float64)
                t0 = time.perf_counter()
                value, gradient = block_tree(value_and_grad(c))
                first_call = time.perf_counter() - t0
                t0 = time.perf_counter()
                value, gradient = block_tree(value_and_grad(c))
                warm = time.perf_counter() - t0
                fields = design_map(c)
                state, diagnostics = primal(fields.momentum_resistance)
                res_rms, res_max = steady_residual_norms(
                    steady_residual(state, fields.momentum_resistance, bench.grid, bench.params())
                )
                chi = np.asarray(fields.fluid_fraction)
                u = np.asarray(state.u)
                u_cell = 0.5 * (u + np.roll(u, 1, axis=0))
                solid = chi < 0.5
                solid_area = float(np.sum(1.0 - chi) * bench.grid.dx * bench.grid.dy)
                entry = {
                    "level": level,
                    "nx": nx,
                    "ny": ny,
                    "h": bench.h,
                    "dt": bench.dt,
                    "epsilon_over_h": epsilon_over_h,
                    "epsilon": bench.epsilon,
                    "alpha_max": alpha_max,
                    "brinkman_layer_over_h": float((bench.viscosity / alpha_max) ** 0.5 / bench.h),
                    "shape": shape,
                    "coefficients": list(coefficients),
                    "objective": float(value),
                    "mean_streamwise_velocity": float(-value),
                    "gradient": np.asarray(gradient).tolist(),
                    "gradient_norm": float(jnp.linalg.norm(gradient)),
                    "directional_derivative": float(jnp.vdot(gradient, direction)),
                    "primal_method": bench.primal_method,
                    "primal_residual": float(diagnostics.residual_norm),
                    "primal_converged": bool(diagnostics.converged),
                    "steady_residual_rms": float(res_rms),
                    "steady_residual_max": float(res_max),
                    "solid_interior_mean_abs_u_over_mean_u": float(np.mean(np.abs(u_cell[solid])) / np.mean(u)) if solid.any() else None,
                    "solid_cells": int(solid.sum()),
                    "smoothed_solid_area": solid_area,
                    "smoothed_solid_area_relative_error": abs(solid_area - np.pi * bench.base_radius**2) / (np.pi * bench.base_radius**2),
                    "first_call_seconds_including_compile": first_call,
                    "warm_value_and_grad_seconds": warm,
                }
                if epsilon_over_h == TAYLOR_POINT["epsilon_over_h"] and alpha_max == TAYLOR_POINT["alpha_max"]:
                    taylor = directional_taylor_test(program.value, c, direction, epsilons=TAYLOR_EPSILONS)
                    entry["taylor_test"] = {
                        "epsilons": np.asarray(taylor.epsilons).tolist(),
                        "first_order_remainders": np.asarray(taylor.first_order_error).tolist(),
                        "first_order_rates": np.asarray(taylor.first_order_rate).tolist(),
                        "zero_order_rates": np.asarray(taylor.zero_order_rate).tolist(),
                    }
                entries.append(entry)
                print(
                    f"{level} eps/h={epsilon_over_h} alpha={alpha_max:g} {shape}: J={float(value):.10f} "
                    f"dJ.d={entry['directional_derivative']:.8e} "
                    f"res={float(res_rms):.1e} warm={warm:.2f}s",
                    flush=True,
                )
    provenance = provenance_record()
    provenance["start_utc"] = started
    provenance["script"] = "examples/32_geometry_gradient_convergence.py"
    path = DATA_DIR / f"32_geometry_gradient_convergence_{level}.json"
    write_json(path, {"level": level, "entries": entries, "provenance": provenance})
    print(f"Saved: {path}")
    return path


def _order(coarse: float, fine: float) -> float | None:
    if coarse == 0 or fine == 0:
        return None
    return float(np.log2(abs(coarse) / abs(fine)))


def merge() -> Path:
    levels = {}
    for level in LEVELS:
        path = DATA_DIR / f"32_geometry_gradient_convergence_{level}.json"
        levels[level] = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for level in LEVELS for e in levels[level]["entries"]]

    def find(level, eps, alpha, shape):
        for e in entries:
            if e["level"] == level and e["epsilon_over_h"] == eps and e["alpha_max"] == alpha and e["shape"] == shape:
                return e
        raise KeyError((level, eps, alpha, shape))

    convergence = []
    order = list(LEVELS)
    for eps in EPSILON_OVER_H:
        for alpha in ALPHA_MAX:
            for shape in SHAPES:
                rows = [find(level, eps, alpha, shape) for level in order]
                fine = rows[-1]
                objective_errors = [abs(r["objective"] - fine["objective"]) for r in rows[:-1]]
                derivative_errors = [abs(r["directional_derivative"] - fine["directional_derivative"]) for r in rows[:-1]]
                convergence.append(
                    {
                        "epsilon_over_h": eps,
                        "alpha_max": alpha,
                        "shape": shape,
                        "h": [r["h"] for r in rows],
                        "objective": [r["objective"] for r in rows],
                        "directional_derivative": [r["directional_derivative"] for r in rows],
                        "objective_change_coarse_to_medium": rows[1]["objective"] - rows[0]["objective"],
                        "objective_change_medium_to_fine": rows[2]["objective"] - rows[1]["objective"],
                        "derivative_change_coarse_to_medium": rows[1]["directional_derivative"] - rows[0]["directional_derivative"],
                        "derivative_change_medium_to_fine": rows[2]["directional_derivative"] - rows[1]["directional_derivative"],
                        "objective_observed_order_richardson": _order(
                            abs(rows[1]["objective"] - rows[0]["objective"]), abs(rows[2]["objective"] - rows[1]["objective"])
                        ),
                        "derivative_observed_order_richardson": _order(
                            abs(rows[1]["directional_derivative"] - rows[0]["directional_derivative"]),
                            abs(rows[2]["directional_derivative"] - rows[1]["directional_derivative"]),
                        ),
                        "objective_relative_difference_to_fine": [e / abs(fine["objective"]) for e in objective_errors],
                        "derivative_relative_difference_to_fine": [e / abs(fine["directional_derivative"]) for e in derivative_errors],
                    }
                )
    sensitivity = []
    for level in order:
        for shape in SHAPES:
            ref = find(level, 1.5, 2400.0, shape)
            for eps in EPSILON_OVER_H:
                for alpha in ALPHA_MAX:
                    e = find(level, eps, alpha, shape)
                    sensitivity.append(
                        {
                            "level": level,
                            "shape": shape,
                            "epsilon_over_h": eps,
                            "alpha_max": alpha,
                            "objective_relative_to_reference": e["objective"] / ref["objective"] - 1.0,
                            "derivative_relative_to_reference": e["directional_derivative"] / ref["directional_derivative"] - 1.0,
                        }
                    )
    record = {
        "claim": (
            "Objective and native-geometry derivative of the converged Stokes--Brinkman benchmark under joint "
            "grid, interface-width and penalization refinement; Taylor tests confirm the discrete derivative at every level."
        ),
        "levels": {level: LEVELS[level] for level in LEVELS},
        "epsilon_over_h": list(EPSILON_OVER_H),
        "alpha_max": list(ALPHA_MAX),
        "shapes": SHAPES,
        "taylor_point": TAYLOR_POINT,
        "entries": entries,
        "grid_convergence": convergence,
        "material_sensitivity": sensitivity,
        "provenance_by_level": {level: levels[level]["provenance"] for level in LEVELS},
        "merged_provenance": provenance_record(),
    }
    path = DATA_DIR / "32_geometry_gradient_convergence.json"
    write_json(path, record)
    print(f"Saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=list(LEVELS), help="run one grid level")
    parser.add_argument("--merge", action="store_true", help="merge per-level records")
    args = parser.parse_args()
    if args.level:
        run_level(args.level)
    if args.merge:
        merge()
    if not args.level and not args.merge:
        parser.error("specify --level or --merge")


if __name__ == "__main__":
    main()
