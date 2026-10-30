"""Device check for the geometry gradient: CPU versus accelerator, both backends.

The manuscript reports CPU float64 gradients only, because an earlier unrolled
reverse pass through the non-periodic kernel failed its Taylor test on the
RTX 4090.  This script archives the
corresponding evidence for the periodic benchmark of the manuscript: on the
default JAX device it runs the directional Taylor test of the flagship
objective with the implicit fixed-point backend and with the unrolled backend,
and compares state, objective and gradient with a CPU evaluation of the same
program.  It is a diagnostic record; it does not change any manuscript number.

Run with ``JAX_ENABLE_X64=true`` and, for the accelerator case,
``JAX_PLATFORMS=cuda,cpu`` so that the CPU reference is available in-process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.run_artifacts import block_tree, provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    FourierObstacleBenchmark,
    directional_taylor_test,
)

TAYLOR_DIRECTION = (0.30, -0.40, 0.20, 0.10, -0.25, 0.15, 0.12, -0.08)
TAYLOR_EPSILONS = (8e-3, 4e-3, 2e-3, 1e-3)
UNROLLED_STEPS = 12_800


def _taylor(result) -> dict[str, object]:
    return {
        "epsilons": np.asarray(result.epsilons).tolist(),
        "first_order_remainders": np.asarray(result.first_order_error).tolist(),
        "first_order_rates": np.asarray(result.first_order_rate).tolist(),
        "zero_order_rates": np.asarray(result.zero_order_rate).tolist(),
        "directional_derivative": float(result.directional_derivative),
        "minimum_asymptotic_rate": float(np.min(np.asarray(result.first_order_rate)[1:])),
    }


def evaluate(device) -> dict[str, object]:
    bench = FourierObstacleBenchmark()
    with jax.default_device(device):
        design_map = bench.design_map()
        implicit_solver, primal = bench.implicit_solver()
        unrolled_solver = bench.transient_solver(UNROLLED_STEPS)
        implicit = DifferentiableProgram(design_map, implicit_solver, bench.objective)
        unrolled = DifferentiableProgram(design_map, unrolled_solver, bench.objective)
        c0 = jax.device_put(bench.zero_coefficients(), device)
        direction = jax.device_put(jnp.asarray(TAYLOR_DIRECTION, dtype=jnp.float64), device)
        out: dict[str, object] = {"device": str(device), "platform": device.platform}
        for name, program in (("implicit_fixed_point", implicit), ("unrolled_reverse_mode", unrolled)):
            value_and_grad = jax.jit(program.value_and_grad)
            started = time.perf_counter()
            value, gradient = block_tree(value_and_grad(c0))
            compile_seconds = time.perf_counter() - started
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                value, gradient = block_tree(value_and_grad(c0))
                samples.append(time.perf_counter() - started)
            taylor = directional_taylor_test(program.value, c0, direction, epsilons=TAYLOR_EPSILONS)
            out[name] = {
                "objective": float(value),
                "gradient": np.asarray(gradient).tolist(),
                "compile_seconds": compile_seconds,
                "warm_samples_seconds": samples,
                "taylor_test": _taylor(taylor),
            }
        state, diagnostics = primal(design_map(c0).momentum_resistance)
        out["state"] = {
            "mean_streamwise_velocity": float(jnp.mean(state.u)),
            "primal_residual": float(diagnostics.residual_norm),
            "converged": bool(diagnostics.converged),
            "u": np.asarray(state.u).tolist(),
        }
        return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    devices = {d.platform: d for d in jax.devices()}
    cpu = jax.devices("cpu")[0]
    accelerator = next((d for p, d in devices.items() if p != "cpu"), None)

    report: dict[str, object] = {
        "claim": "Device dependence of the flagship geometry gradient for both sensitivity backends.",
        "configuration": FourierObstacleBenchmark().to_dict(),
        "unrolled_steps": UNROLLED_STEPS,
        "cpu": evaluate(cpu),
    }
    if accelerator is not None:
        acc = evaluate(accelerator)
        report["accelerator"] = acc
        u_cpu = np.asarray(report["cpu"]["state"]["u"])
        u_acc = np.asarray(acc["state"]["u"])
        comparison = {"state_relative_l2_difference": float(np.linalg.norm(u_cpu - u_acc) / np.linalg.norm(u_cpu))}
        for name in ("implicit_fixed_point", "unrolled_reverse_mode"):
            g_cpu = np.asarray(report["cpu"][name]["gradient"])
            g_acc = np.asarray(acc[name]["gradient"])
            comparison[name] = {
                "relative_objective_difference": abs(report["cpu"][name]["objective"] - acc[name]["objective"]) / abs(report["cpu"][name]["objective"]),
                "relative_gradient_difference": float(np.linalg.norm(g_cpu - g_acc) / np.linalg.norm(g_cpu)),
                "accelerator_minimum_taylor_rate": acc[name]["taylor_test"]["minimum_asymptotic_rate"],
                "accelerator_taylor_passes_1p8": acc[name]["taylor_test"]["minimum_asymptotic_rate"] > 1.8,
            }
        report["comparison"] = comparison
    else:
        report["accelerator"] = None
        report["note"] = "no accelerator visible; CPU-only record"
    for key in ("cpu", "accelerator"):
        if isinstance(report.get(key), dict):
            report[key]["state"].pop("u")
    report["provenance"] = {**provenance_record(), "script": "examples/33_backend_device_check.py"}
    path = root / "output" / "data" / "33_backend_device_check.json"
    write_json(path, report)
    summary = {k: v for k, v in report.items() if k in ("comparison", "note")}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
