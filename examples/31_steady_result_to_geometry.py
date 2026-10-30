"""Steady-state closed loop: converged result -> implicit gradient -> geometry.

This is the flagship verification of the manuscript.  The physical objective is
evaluated at the converged fixed point of the mobility-weighted periodic MAC
Stokes--Brinkman kernel, so that the result does not depend on an arbitrary
finite horizon, and the derivative with respect to the eight native Fourier
geometry coefficients is obtained by implicit differentiation of that fixed
point.  The record contains:

1. the horizon and time-step diagnostics that motivate the steady objective and
   the mobility-weighted projection;
2. a cross-backend comparison (implicit fixed point versus long-horizon
   unrolled reverse mode) with synchronized timings, host memory and adjoint
   residuals;
3. double-precision directional Taylor tests of both backends;
4. a projected-gradient closed loop at exactly fixed area with an explicit
   stationarity measure;
5. full provenance (commit, environment, precision, timestamps).

Run from the repository root with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true``.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import (
    divergence_norms,
    simulate,
    steady_adjoint_residual,
    steady_residual,
    steady_residual_norms,
    steady_stokes_brinkman,
    step,
)
from jax_cad.optimization.projected_descent import projected_gradient_descent
from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    FourierObstacleBenchmark,
    directional_taylor_test,
    fixed_point_adjoint,
    fourier_boundary_radius,
    fourier_shape_area,
    make_implicit_fixed_point_solver,
)

TAYLOR_DIRECTION = (0.30, -0.40, 0.20, 0.10, -0.25, 0.15, 0.12, -0.08)
TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)
UNROLLED_STEPS = 12_800
FINAL_UNROLLED_STEPS = (12_800, 25_600, 51_200)
HORIZON_STEPS = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
PREDICTION_STEPS = (0.04, 0.02, 0.01, 0.005, 0.0025)
TIMING_SAMPLES = 5


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _timed(function, argument, samples: int):
    started = time.perf_counter()
    output = block_tree(function(argument))
    compile_seconds = time.perf_counter() - started
    durations = []
    for _ in range(samples):
        started = time.perf_counter()
        output = block_tree(function(argument))
        durations.append(time.perf_counter() - started)
    return output, compile_seconds, durations


def _taylor_record(result) -> dict[str, object]:
    return {
        "direction": list(TAYLOR_DIRECTION),
        "direction_normalized": True,
        "epsilons": np.asarray(result.epsilons).tolist(),
        "zero_order_errors": np.asarray(result.zero_order_error).tolist(),
        "first_order_remainders": np.asarray(result.first_order_error).tolist(),
        "zero_order_rates": np.asarray(result.zero_order_rate).tolist(),
        "first_order_rates": np.asarray(result.first_order_rate).tolist(),
        "directional_derivative": float(result.directional_derivative),
    }


def _shape_diagnostics(bench: FourierObstacleBenchmark, coefficients, state, fields) -> dict[str, object]:
    grid = bench.grid
    chi = np.asarray(fields.fluid_fraction)
    u_cell = 0.5 * (np.asarray(state.u) + np.roll(np.asarray(state.u), 1, axis=0))
    solid = chi < 0.5
    mean_u = float(np.mean(np.asarray(state.u)))
    solid_area = float(np.sum(1.0 - chi) * grid.dx * grid.dy)
    exact_area = float(np.pi * bench.base_radius**2)
    div_rms, div_max = divergence_norms(state, grid)
    residual_rms, residual_max = steady_residual_norms(
        steady_residual(state, fields.momentum_resistance, grid, bench.params())
    )
    angles = jnp.array([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi], dtype=jnp.float64)
    radii = fourier_boundary_radius(
        angles, coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096
    )
    return {
        "mean_streamwise_velocity": mean_u,
        "max_streamwise_velocity": float(np.max(np.asarray(state.u))),
        "solid_cells": int(solid.sum()),
        "solid_interior_mean_abs_u_over_mean_u": float(np.mean(np.abs(u_cell[solid])) / mean_u),
        "smoothed_solid_area": solid_area,
        "exact_area": exact_area,
        "smoothed_solid_area_relative_error": abs(solid_area - exact_area) / exact_area,
        "quadrature_area": float(
            fourier_shape_area(coefficients, base_radius=bench.base_radius, modes=bench.modes, n_quadrature=4096)
        ),
        "cardinal_radii": np.asarray(radii).tolist(),
        "divergence_rms": float(div_rms),
        "divergence_max": float(div_max),
        "steady_residual_rms": float(residual_rms),
        "steady_residual_max": float(residual_max),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "output" / "data" / "31_steady_result_to_geometry.json"
    started_utc = provenance_record()["utc"]
    wall_started = time.perf_counter()

    bench = FourierObstacleBenchmark()
    grid = bench.grid
    params = bench.params()
    design_map = bench.design_map()
    state0 = bench.initial_state()
    implicit_solver, primal = bench.implicit_solver()
    unrolled_solver = bench.transient_solver(UNROLLED_STEPS)

    implicit_program = DifferentiableProgram(design_map, implicit_solver, bench.objective)
    unrolled_program = DifferentiableProgram(design_map, unrolled_solver, bench.objective)
    implicit_value_and_grad = jax.jit(implicit_program.value_and_grad)
    implicit_value = jax.jit(implicit_program.value)
    unrolled_value_and_grad = jax.jit(unrolled_program.value_and_grad)

    coefficients0 = bench.zero_coefficients()

    # ------------------------------------------------------------------ 1. horizon
    print("[1/5] horizon and time-step diagnostics", flush=True)
    horizon: dict[str, object] = {"steps": list(HORIZON_STEPS), "dt": bench.dt}
    alpha0 = design_map(coefficients0).momentum_resistance
    horizon["circle_mean_u_by_steps"] = [
        float(jnp.mean(simulate(state0, alpha0, grid, params, n_steps=n).u)) for n in HORIZON_STEPS
    ]
    horizon["circle_final_time_by_steps"] = [n * bench.dt for n in HORIZON_STEPS]
    steady_circle, diag_circle = primal(alpha0)
    horizon["circle_steady_mean_u"] = float(jnp.mean(steady_circle.u))
    horizon["circle_primal_residual"] = float(diag_circle.residual_norm)
    horizon["circle_primal_method"] = bench.primal_method
    # Independent cross-check: direct preconditioned solve of the symmetric
    # steady saddle-point system versus the Krylov fixed point of the split step.
    direct_circle = steady_stokes_brinkman(
        alpha0, grid, params, tol=1e-12, atol=0.0, maxiter=4000, restart=50
    )
    direct_rms, direct_max = steady_residual_norms(steady_residual(direct_circle, alpha0, grid, params))
    horizon["direct_saddle_point_cross_check"] = {
        "mean_u_direct": float(jnp.mean(direct_circle.u)),
        "mean_u_fixed_point": float(jnp.mean(steady_circle.u)),
        "max_abs_velocity_difference": float(
            jnp.maximum(
                jnp.max(jnp.abs(direct_circle.u - steady_circle.u)),
                jnp.max(jnp.abs(direct_circle.v - steady_circle.v)),
            )
        ),
        "max_abs_pressure_difference": float(jnp.max(jnp.abs(direct_circle.p - steady_circle.p))),
        "direct_steady_residual_rms": float(direct_rms),
        "direct_steady_residual_max": float(direct_max),
    }

    dt_study = {"dt_values": [], "weighted": [], "constant_mobility": []}
    for k in range(3):
        dt = bench.dt / 2**k
        dt_study["dt_values"].append(dt)
        for key, weighted in (("weighted", True), ("constant_mobility", False)):
            p_k = bench.params(dt=dt, mobility_weighted=weighted)
            _, primal_k = make_implicit_fixed_point_solver(
                lambda s, a, p_k=p_k: step(s, a, grid, p_k),
                max_iterations=bench.fixed_point_max_iterations,
                atol=bench.fixed_point_atol,
                rtol=bench.fixed_point_rtol,
            )
            s_k, d_k = primal_k(alpha0, state0)
            r_rms, r_max = steady_residual_norms(steady_residual(s_k, alpha0, grid, p_k))
            dt_study[key].append(
                {
                    "dt": dt,
                    "mean_u": float(jnp.mean(s_k.u)),
                    "fixed_point_iterations": int(d_k.iterations),
                    "fixed_point_residual": float(d_k.residual_norm),
                    "converged": bool(d_k.converged),
                    "steady_residual_rms": float(r_rms),
                    "steady_residual_max": float(r_max),
                    "max_dt_alpha_face": float(dt * jnp.max(alpha0)),
                }
            )
    horizon["fixed_point_dt_study_circle"] = dt_study

    # ------------------------------------------------------------ 2. backends (initial)
    print("[2/5] cross-backend comparison at the initial circle", flush=True)
    rss_before = _rss_bytes()
    (value_i, grad_i), compile_i, samples_i = _timed(implicit_value_and_grad, coefficients0, TIMING_SAMPLES)
    rss_implicit = _rss_bytes()
    (value_u, grad_u), compile_u, samples_u = _timed(unrolled_value_and_grad, coefficients0, TIMING_SAMPLES)
    rss_unrolled = _rss_bytes()
    adjoint_cotangent = jax.grad(lambda s: bench.objective(s, None, coefficients0))(steady_circle)
    _, adjoint_residual = steady_adjoint_residual(
        steady_circle, alpha0, grid, params, adjoint_cotangent,
        tol=bench.primal_tol, atol=bench.primal_atol, maxiter=bench.primal_maxiter, restart=bench.primal_restart,
    )
    cotangent_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(adjoint_cotangent)))
    _, fixed_point_adjoint_residual, _ = fixed_point_adjoint(
        lambda s, a: step(s, a, grid, params),
        steady_circle,
        alpha0,
        adjoint_cotangent,
        linear_tol=bench.adjoint_tol,
        linear_atol=bench.adjoint_atol,
        linear_maxiter=bench.adjoint_maxiter,
        linear_restart=bench.adjoint_restart,
    )
    backends_initial = {
        "implicit_fixed_point": {
            "objective": float(value_i),
            "gradient": np.asarray(grad_i).tolist(),
            "compile_seconds": compile_i,
            "warm_samples_seconds": samples_i,
            "median_warm_seconds": float(np.median(samples_i)),
            "primal_method": bench.primal_method,
            "primal_residual": float(diag_circle.residual_norm),
            "primal_converged": bool(diag_circle.converged),
            "adjoint_relative_residual": float(adjoint_residual),
            "adjoint_formulation": "residual-based: symmetric steady saddle-point operator A^T lambda = dJ/ds",
            "fixed_point_adjoint_relative_residual": float(fixed_point_adjoint_residual),
            "adjoint_cotangent_norm": float(cotangent_norm),
            "peak_host_rss_bytes_after": rss_implicit,
        },
        "unrolled_reverse_mode": {
            "n_steps": UNROLLED_STEPS,
            "final_time": UNROLLED_STEPS * bench.dt,
            "objective": float(value_u),
            "gradient": np.asarray(grad_u).tolist(),
            "compile_seconds": compile_u,
            "warm_samples_seconds": samples_u,
            "median_warm_seconds": float(np.median(samples_u)),
            "peak_host_rss_bytes_after": rss_unrolled,
        },
        "peak_host_rss_bytes_before": rss_before,
        "memory_measurement_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss sampled after each backend; process-level upper bound",
        "relative_gradient_difference": float(jnp.linalg.norm(grad_i - grad_u) / jnp.linalg.norm(grad_u)),
        "relative_objective_difference": float(abs(value_i - value_u) / abs(value_u)),
    }

    # ---------------------------------------------------------------- 3. Taylor
    print("[3/5] directional Taylor tests", flush=True)
    direction = jnp.asarray(TAYLOR_DIRECTION, dtype=jnp.float64)
    taylor_implicit = directional_taylor_test(implicit_program.value, coefficients0, direction, epsilons=TAYLOR_EPSILONS)
    taylor_unrolled = directional_taylor_test(unrolled_program.value, coefficients0, direction, epsilons=TAYLOR_EPSILONS)

    # ------------------------------------------------------------ 4. closed loop
    print("[4/5] projected-gradient closed loop (implicit backend)", flush=True)

    def value_and_grad_host(x):
        value, gradient = implicit_value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(gradient)

    def value_host(x):
        return float(implicit_value(jnp.asarray(x)))

    loop_started = time.perf_counter()
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
    loop_seconds = time.perf_counter() - loop_started
    for item in result.trace:
        alpha_k = design_map(jnp.asarray(item["coefficients"])).momentum_resistance
        state_k, diag_k = primal(alpha_k)
        item["mean_streamwise_velocity"] = float(jnp.mean(state_k.u))
        item["primal_residual"] = float(diag_k.residual_norm)
    final_coefficients = jnp.asarray(result.coefficients)
    final_fields = design_map(final_coefficients)
    final_state, final_diag = primal(final_fields.momentum_resistance)
    initial_fields = design_map(coefficients0)
    accepted = [item for item in result.trace if item["accepted"]]
    first = result.trace[0]

    # ------------------------------------------------------- 5. backends (final)
    print("[5/5] cross-backend comparison at the final geometry", flush=True)
    value_if, grad_if = block_tree(implicit_value_and_grad(final_coefficients))
    unrolled_by_horizon = []
    for n_steps in FINAL_UNROLLED_STEPS:
        program_n = DifferentiableProgram(design_map, bench.transient_solver(n_steps), bench.objective)
        value_n, grad_n = block_tree(jax.jit(program_n.value_and_grad)(final_coefficients))
        unrolled_by_horizon.append(
            {
                "n_steps": n_steps,
                "final_time": n_steps * bench.dt,
                "objective": float(value_n),
                "relative_objective_difference": float(abs(value_if - value_n) / abs(value_n)),
                "relative_gradient_difference": float(jnp.linalg.norm(grad_if - grad_n) / jnp.linalg.norm(grad_n)),
            }
        )
    longest = unrolled_by_horizon[-1]
    backends_final = {
        "implicit_objective": float(value_if),
        "implicit_gradient": np.asarray(grad_if).tolist(),
        "unrolled_by_horizon": unrolled_by_horizon,
        "unrolled_objective": longest["objective"],
        "relative_gradient_difference": longest["relative_gradient_difference"],
        "relative_objective_difference": longest["relative_objective_difference"],
        "note": "the unrolled reference must itself be converged in time; the horizon sweep shows the residual transient error of each finite horizon",
    }
    # first-order prediction quality versus step length at the initial circle
    value0, grad0 = block_tree(implicit_value_and_grad(coefficients0))
    pg0 = np.asarray(grad0)
    direction0 = -pg0 / np.linalg.norm(pg0)
    prediction_study = []
    for step_length in PREDICTION_STEPS:
        candidate = np.clip(np.asarray(coefficients0) + step_length * direction0, -bench.coefficient_bound, bench.coefficient_bound)
        realised = float(implicit_value(jnp.asarray(candidate))) - float(value0)
        predicted = float(pg0 @ (candidate - np.asarray(coefficients0)))
        prediction_study.append({"step_length": step_length, "linear_prediction": predicted, "realised_change": realised, "ratio": realised / predicted})
    initial_diag = _shape_diagnostics(bench, coefficients0, steady_circle, initial_fields)
    final_diagnostics = _shape_diagnostics(bench, final_coefficients, final_state, final_fields)

    def _field_arrays(state, fields) -> dict[str, object]:
        u = np.asarray(state.u)
        v = np.asarray(state.v)
        return {
            "layout": "cell-centred arrays indexed [i, j] with x = (i + 1/2) dx, y = (j + 1/2) dy; u and v are averaged from the MAC faces",
            "alpha": np.asarray(fields.momentum_resistance).tolist(),
            "fluid_fraction": np.asarray(fields.fluid_fraction).tolist(),
            "level_set": np.asarray(fields.level_set).tolist(),
            "u_cell": (0.5 * (u + np.roll(u, 1, axis=0))).tolist(),
            "v_cell": (0.5 * (v + np.roll(v, 1, axis=1))).tolist(),
            "pressure": np.asarray(state.p).tolist(),
        }

    archived_fields = {
        "initial": _field_arrays(steady_circle, initial_fields),
        "final": _field_arrays(final_state, final_fields),
    }
    transient_final = float(jnp.mean(simulate(state0, final_fields.momentum_resistance, grid, params, n_steps=100).u))

    provenance = provenance_record()
    provenance["start_utc"] = started_utc
    provenance["wall_seconds"] = time.perf_counter() - wall_started
    provenance["peak_host_rss_bytes"] = _rss_bytes()
    provenance["script"] = "examples/31_steady_result_to_geometry.py"

    record = {
        "claim": (
            "A converged periodic Stokes--Brinkman result is differentiated implicitly to eight "
            "native, exactly area-preserving Fourier geometry coefficients; the implicit gradient "
            "agrees with long-horizon reverse mode, passes a double-precision Taylor test, and drives "
            "a projected-gradient descent with an explicit stationarity measure."
        ),
        "configuration": bench.to_dict(),
        "horizon_and_time_step_diagnostics": horizon,
        "cross_backend_initial": backends_initial,
        "derivative_verification": {
            "implicit_fixed_point": _taylor_record(taylor_implicit),
            "unrolled_reverse_mode": _taylor_record(taylor_unrolled),
        },
        "first_geometry_step": {
            "step_length": first["step_length"],
            "linear_prediction": first["linear_prediction"],
            "realised_change": first["realised_change"],
            "ratio": first["realised_change"] / first["linear_prediction"] if first["linear_prediction"] else None,
            "prediction_quality_by_step_length": prediction_study,
        },
        "optimization": {
            "algorithm": "normalized projected steepest descent with box projection and Armijo backtracking",
            "termination": result.termination,
            "converged": result.converged,
            "iterations": result.iterations,
            "accepted_updates": len(accepted),
            "initial_objective": first["objective"],
            "final_objective": float(value_if),
            "objective_reduction_percent": 100.0 * (first["objective"] - float(value_if)) / abs(first["objective"]),
            "initial_mean_streamwise_velocity": initial_diag["mean_streamwise_velocity"],
            "final_mean_streamwise_velocity": final_diagnostics["mean_streamwise_velocity"],
            "mean_velocity_increase_percent": 100.0
            * (final_diagnostics["mean_streamwise_velocity"] / initial_diag["mean_streamwise_velocity"] - 1.0),
            "initial_projected_gradient_norm": result.initial_projected_gradient_norm,
            "final_projected_gradient_norm": result.final_projected_gradient_norm,
            "final_gradient_norm": float(jnp.linalg.norm(grad_if)),
            "active_bounds": [bool(abs(c) >= bench.coefficient_bound - 1e-12) for c in np.asarray(final_coefficients)],
            "initial_area": initial_diag["quadrature_area"],
            "final_area": final_diagnostics["quadrature_area"],
            "relative_area_error": abs(final_diagnostics["quadrature_area"] - initial_diag["quadrature_area"]) / initial_diag["quadrature_area"],
            "final_coefficients": np.asarray(final_coefficients).tolist(),
            "final_cardinal_radii": final_diagnostics["cardinal_radii"],
            "final_primal_residual": float(final_diag.residual_norm),
            "loop_wall_seconds": loop_seconds,
            "final_shape_transient_100_step_mean_u": transient_final,
        },
        "initial_shape_diagnostics": initial_diag,
        "final_shape_diagnostics": final_diagnostics,
        "cross_backend_final": backends_final,
        "fields": archived_fields,
        "trace": result.trace,
        "provenance": provenance,
    }
    write_json(data_path, record)
    summary = {k: record[k] for k in ("first_geometry_step", "optimization", "cross_backend_initial", "cross_backend_final")}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {data_path}")


if __name__ == "__main__":
    main()
