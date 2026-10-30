"""Grid refinement of the geometry derivative at a fixed regularized problem.

Record 32 refines the grid with ``epsilon/h`` fixed, so the regularized
continuous problem changes with every level and the discrete geometry
derivative cannot be expected to converge.  This study separates the two
limits.  (i) *Fixed regularization*: the physical interface half-width
``epsilon`` and the penalization ``alpha_max`` are held fixed while the grid is
refined, so the discrete objective and its native-geometry derivative must
converge to those of one well-defined continuous problem (expected order two
for the MAC discretization of smooth coefficients).  (ii) *Regularization
limit*: on the finest grid the interface width is halved twice at fixed
``alpha_max`` to estimate the order of the ``epsilon -> 0`` limit, and the
penalization is varied at fixed ``epsilon``.

Usage (repository root, ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``)::

    python examples/35_regularized_refinement.py --level coarse
    python examples/35_regularized_refinement.py --level medium
    python examples/35_regularized_refinement.py --level fine
    python examples/35_regularized_refinement.py --merge
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

LEVELS = {"coarse": (32, 20), "medium": (64, 40), "fine": (128, 80), "finest": (256, 160)}
# Fixed physical interface half-width: 1.5 cells on the coarse grid, 3 on the
# medium grid, 6 on the fine grid and 12 on the finest grid.
EPSILON = 0.075
ALPHA_MAX = (600.0, 2400.0, 9600.0)
# Regularization limit on the finest level at alpha_max = 2400.
EPSILON_SEQUENCE_FINE = (0.15, 0.075, 0.0375)
SHAPES = {
    "circle": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "deformed": (0.15, 0.0, 0.05, 0.0, 0.0, -0.03, 0.0, 0.02),
}
TAYLOR_DIRECTION = (0.30, -0.40, 0.20, 0.10, -0.25, 0.15, 0.12, -0.08)
TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)
TAYLOR_ALPHA = 2400.0
PRIMAL_METHOD_BY_ALPHA = {600.0: "saddle", 2400.0: "gmres", 9600.0: "gmres"}
# Taylor tests (four extra primal solves) are run at alpha_max = 2400 on every
# level except the finest, where the derivative is verified by agreement with
# the extrapolated sequence instead.
TAYLOR_LEVELS = ("coarse", "medium", "fine")
# On the finest grid the restarted-GMRES fixed-point primal needs hours per
# solve at alpha_max >= 2400, whereas the direct saddle-point solve at
# alpha_max = 600 takes about two minutes; the fourth level is therefore
# computed at alpha_max = 600 only.
ALPHA_BY_LEVEL = {"finest": (600.0,)}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "data"


def _physical_objective(bench: FourierObstacleBenchmark):
    def objective(state, fields, coefficients):
        del fields, coefficients
        return bench.physical_loss(state)

    return objective


def _benchmark(nx: int, ny: int, epsilon: float, alpha_max: float) -> FourierObstacleBenchmark:
    method = PRIMAL_METHOD_BY_ALPHA[alpha_max]
    probe = FourierObstacleBenchmark(nx=nx, ny=ny)
    return FourierObstacleBenchmark(
        nx=nx,
        ny=ny,
        epsilon_over_h=epsilon / probe.h,
        alpha_max=alpha_max,
        primal_method=method,
        primal_maxiter=4_000 if method == "saddle" else 40_000,
        primal_restart=200 if method == "saddle" else 60,
    )


def _evaluate(level: str, bench: FourierObstacleBenchmark, study: str, with_taylor: bool, direction) -> list[dict]:
    design_map = bench.design_map()
    solver, primal = bench.implicit_solver()
    program = DifferentiableProgram(design_map, solver, _physical_objective(bench))
    value_and_grad = jax.jit(program.value_and_grad)
    entries = []
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
        solid_area = float(np.sum(1.0 - chi) * bench.grid.dx * bench.grid.dy)
        entry = {
            "study": study,
            "level": level,
            "nx": bench.nx,
            "ny": bench.ny,
            "h": bench.h,
            "dt": bench.dt,
            "epsilon": bench.epsilon,
            "epsilon_over_h": bench.epsilon_over_h,
            "alpha_max": bench.alpha_max,
            "brinkman_layer_over_h": float((bench.viscosity / bench.alpha_max) ** 0.5 / bench.h),
            "brinkman_layer_over_epsilon": float((bench.viscosity / bench.alpha_max) ** 0.5 / bench.epsilon),
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
            "smoothed_solid_area": solid_area,
            "smoothed_solid_area_relative_error": abs(solid_area - np.pi * bench.base_radius**2) / (np.pi * bench.base_radius**2),
            "first_call_seconds_including_compile": first_call,
            "warm_value_and_grad_seconds": warm,
        }
        if with_taylor:
            taylor = directional_taylor_test(program.value, c, direction, epsilons=TAYLOR_EPSILONS)
            entry["taylor_test"] = {
                "epsilons": np.asarray(taylor.epsilons).tolist(),
                "first_order_remainders": np.asarray(taylor.first_order_error).tolist(),
                "first_order_rates": np.asarray(taylor.first_order_rate).tolist(),
                "zero_order_rates": np.asarray(taylor.zero_order_rate).tolist(),
            }
        entries.append(entry)
        print(
            f"{level} [{study}] eps={bench.epsilon:g} (eps/h={bench.epsilon_over_h:g}) alpha={bench.alpha_max:g} {shape}: "
            f"J={float(value):.10f} dJ.d={entry['directional_derivative']:.8e} res={float(res_rms):.1e} warm={warm:.2f}s",
            flush=True,
        )
    return entries


def run_level(level: str) -> Path:
    nx, ny = LEVELS[level]
    direction = normalize_direction(jnp.asarray(TAYLOR_DIRECTION, dtype=jnp.float64))
    started = provenance_record()["utc"]
    entries = []
    for alpha_max in ALPHA_BY_LEVEL.get(level, ALPHA_MAX):
        bench = _benchmark(nx, ny, EPSILON, alpha_max)
        entries += _evaluate(level, bench, "fixed_regularization", alpha_max == TAYLOR_ALPHA and level in TAYLOR_LEVELS, direction)
    if level == "fine":
        for epsilon in EPSILON_SEQUENCE_FINE:
            if epsilon == EPSILON:
                continue  # already computed above at alpha_max = TAYLOR_ALPHA
            bench = _benchmark(nx, ny, epsilon, TAYLOR_ALPHA)
            entries += _evaluate(level, bench, "regularization_limit", False, direction)
    provenance = provenance_record()
    provenance["start_utc"] = started
    provenance["script"] = "examples/35_regularized_refinement.py"
    path = DATA_DIR / f"35_regularized_refinement_{level}.json"
    write_json(path, {"level": level, "entries": entries, "provenance": provenance})
    print(f"Saved: {path}")
    return path


def _order(d1: float, d2: float) -> float | None:
    if d1 == 0 or d2 == 0:
        return None
    return float(np.log2(abs(d1) / abs(d2)))


def merge() -> Path:
    levels = {}
    for level in LEVELS:
        path = DATA_DIR / f"35_regularized_refinement_{level}.json"
        levels[level] = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for level in LEVELS for e in levels[level]["entries"]]

    def find(**keys):
        for e in entries:
            if all(abs(e[k] - v) < 1e-12 if isinstance(v, float) else e[k] == v for k, v in keys.items()):
                return e
        raise KeyError(keys)

    grid_convergence = []
    for alpha_max in ALPHA_MAX:
        for shape in SHAPES:
            used_levels = [level for level in LEVELS if alpha_max in ALPHA_BY_LEVEL.get(level, ALPHA_MAX)]
            rows = [find(level=level, study="fixed_regularization", alpha_max=alpha_max, shape=shape) for level in used_levels]
            J = [r["objective"] for r in rows]
            D = [r["directional_derivative"] for r in rows]
            dJ = [J[k + 1] - J[k] for k in range(len(J) - 1)]
            dD = [D[k + 1] - D[k] for k in range(len(D) - 1)]
            # Observed orders from consecutive triples; the last triple uses the
            # three finest grids and drives the Richardson extrapolation.
            pJ = [_order(dJ[k], dJ[k + 1]) for k in range(len(dJ) - 1)]
            pD = [_order(dD[k], dD[k + 1]) for k in range(len(dD) - 1)]
            grid_convergence.append(
                {
                    "alpha_max": alpha_max,
                    "epsilon": EPSILON,
                    "shape": shape,
                    "levels": used_levels,
                    "h": [r["h"] for r in rows],
                    "epsilon_over_h": [r["epsilon_over_h"] for r in rows],
                    "objective": J,
                    "directional_derivative": D,
                    "objective_changes": dJ,
                    "derivative_changes": dD,
                    "objective_observed_orders": pJ,
                    "derivative_observed_orders": pD,
                    "objective_extrapolated": None if pJ[-1] is None else J[-1] + dJ[-1] / (2 ** pJ[-1] - 1),
                    "derivative_extrapolated": None if pD[-1] is None else D[-1] + dD[-1] / (2 ** pD[-1] - 1),
                    "derivative_relative_changes": [abs(d) / abs(D[-1]) for d in dD],
                    "min_taylor_first_order_rate": [min(r["taylor_test"]["first_order_rates"]) if "taylor_test" in r else None for r in rows],
                }
            )
    regularization_limit = []
    for shape in SHAPES:
        rows = [find(level="fine", epsilon=eps, alpha_max=TAYLOR_ALPHA, shape=shape) for eps in EPSILON_SEQUENCE_FINE]
        J = [r["objective"] for r in rows]
        D = [r["directional_derivative"] for r in rows]
        regularization_limit.append(
            {
                "level": "fine",
                "alpha_max": TAYLOR_ALPHA,
                "shape": shape,
                "epsilon": list(EPSILON_SEQUENCE_FINE),
                "epsilon_over_h": [r["epsilon_over_h"] for r in rows],
                "objective": J,
                "directional_derivative": D,
                "objective_observed_order_in_epsilon": _order(J[1] - J[0], J[2] - J[1]),
                "derivative_observed_order_in_epsilon": _order(D[1] - D[0], D[2] - D[1]),
            }
        )
    penalization = []
    for level in LEVELS:
        for shape in SHAPES:
            alphas = ALPHA_BY_LEVEL.get(level, ALPHA_MAX)
            rows = [find(level=level, study="fixed_regularization", alpha_max=a, shape=shape) for a in alphas]
            penalization.append(
                {
                    "level": level,
                    "shape": shape,
                    "alpha_max": list(alphas),
                    "objective": [r["objective"] for r in rows],
                    "directional_derivative": [r["directional_derivative"] for r in rows],
                }
            )
    record = {
        "claim": (
            "At fixed interface width and penalization the converged objective and its native-geometry derivative "
            "converge under grid refinement; the regularization limit is estimated separately on the finest grid."
        ),
        "levels": {level: LEVELS[level] for level in LEVELS},
        "epsilon": EPSILON,
        "alpha_max": list(ALPHA_MAX),
        "alpha_max_by_level": {level: list(ALPHA_BY_LEVEL.get(level, ALPHA_MAX)) for level in LEVELS},
        "epsilon_sequence_fine": list(EPSILON_SEQUENCE_FINE),
        "shapes": SHAPES,
        "taylor_direction": list(TAYLOR_DIRECTION),
        "entries": entries,
        "grid_convergence": grid_convergence,
        "regularization_limit": regularization_limit,
        "penalization_dependence": penalization,
        "provenance_by_level": {level: levels[level]["provenance"] for level in LEVELS},
        "merged_provenance": provenance_record(),
    }
    path = DATA_DIR / "35_regularized_refinement.json"
    write_json(path, record)
    print(f"Saved: {path}")
    for g in grid_convergence:
        print(
            f"alpha={g['alpha_max']:g} {g['shape']}: J orders {g['objective_observed_orders']}, "
            f"dJ.d orders {g['derivative_observed_orders']}, dJ.d={g['directional_derivative']}, rel changes {g['derivative_relative_changes']}"
        )
    for r in regularization_limit:
        print(f"eps-limit {r['shape']}: dJ.d={r['directional_derivative']} order {r['derivative_observed_order_in_epsilon']}")
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
