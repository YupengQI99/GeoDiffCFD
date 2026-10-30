#!/usr/bin/env python3
"""Bounded CPU-only verification of the opt-in normal-constraint prototype.

This records development tests, not a schema-1 cylinder experiment. A clean
source run can be requested for final evidence; dirty development runs remain
labelled as such. No arbitrary commands or large CFD cases can be launched.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "multibody-offset": ["tests/test_multibody_offset.py", "tests/test_wall_shift_prerequisites.py", "tests/test_wall_shift_registration.py", "tests/test_multibody_timecheck.py", "tests/test_wake_guard.py"],
    "dfg2d1-wall-shift": ["tests/test_dfg2d1_wall_shift.py", "tests/test_wall_shift_prerequisites.py", "tests/test_wall_shift_registration.py", "tests/test_brinkman_mask_calibration.py", "tests/test_dfg2d2_pairing.py", "tests/test_wake_guard.py"],
    "wall-shift-unit": ["tests/test_wall_shift_prerequisites.py", "tests/test_wall_shift_registration.py", "tests/test_brinkman_mask_calibration.py", "tests/test_dfg2d2_pairing.py"],
    "dfg2d2-pairing": ["tests/test_dfg2d2_pairing.py", "tests/test_multibody_penalty.py", "tests/test_multibody_timecheck.py", "tests/test_wake_guard.py"],
    "multibody-timecheck": ["tests/test_multibody_timecheck.py", "tests/test_multibody_penalty.py", "tests/test_wake_guard.py", "tests/test_multibody_wake_config.py"],
    "multibody-penalty": ["tests/test_multibody_penalty.py", "tests/test_wake_guard.py", "tests/test_multibody_wake_config.py"],
    "multibody-final-summary": ["tests/test_multibody_final_summary.py"],
    "vascular-density-restart": ["tests/test_vascular_density_step_restart.py", "tests/test_vascular_density_final.py", "tests/test_vascular_density_optimization.py", "tests/test_vascular_density_audit.py", "tests/test_vascular_density_driver.py", "tests/test_projected_descent.py"],
    "vascular-refinement-preflight": ["tests/test_vascular_refinement_preflight.py", "tests/test_vascular_flow.py"],
    "vascular-density-final": ["tests/test_vascular_density_final.py", "tests/test_vascular_connectivity.py", "tests/test_vascular_density_optimization.py", "tests/test_vascular_density_audit.py", "tests/test_vascular_density_driver.py", "tests/test_projected_descent.py"],
    "vascular-density-optimization": ["tests/test_vascular_density_optimization.py", "tests/test_vascular_density_driver.py", "tests/test_projected_descent.py"],
    "vascular-density-audit": ["tests/test_vascular_density_audit.py", "tests/test_vascular_density_driver.py"],
    "vascular-density": ["tests/test_vascular_routes.py", "tests/test_vascular_density_driver.py"],
    "vascular-audit": ["tests/test_vascular_array_audit.py"],
    "vascular-implicit": ["tests/test_vascular_implicit.py"],
    "vascular-routes": ["tests/test_vascular_routes.py", "tests/test_vascular_route_driver.py"],
    "vascular-volume": ["tests/test_vascular_volume.py"],
    "vascular-flow": ["tests/test_vascular_flow.py"],
    "vascular-geometry": ["tests/test_vascular_plate.py"],
    "multiroute-material": ["tests/test_multiroute_cassette.py"],
    "multiroute-flow": ["tests/test_multiroute_flow.py"],
    "multibody-continuation": ["tests/test_multibody_wake_continuation.py", "tests/test_multibody_wake_config.py", "tests/test_projected_descent.py", "tests/test_wake_guard.py"],
    "multibody-wake": ["tests/test_multibody_wake.py", "tests/test_multibody_wake_config.py", "tests/test_wake_guard.py"],
    "wake-design": ["tests/test_wake_design.py", "tests/test_wake_guard.py"],
    "fixed-kernel": ["tests/test_fixed_kernel_response.py", "-k", "not geometry_gradient"],
    "fixed-kernel-gradient": ["tests/test_fixed_kernel_response.py", "-k", "geometry_gradient"],
    "fixed-model-driver": ["tests/test_fixed_model_driver.py", "tests/test_normal_cylinder_driver.py", "tests/test_cylinder_guard.py"],
    "fixed-model-archives": ["tests/test_fixed_model_archives.py"],
    "fixed-model-paper": ["tests/test_manuscript_evidence.py", "tests/test_fixed_model_archives.py"],
    "reaction-pressure": ["tests/test_reaction_pressure_trace.py", "tests/test_pressure_trace_diagnostic.py", "tests/test_dolfinx_brinkman.py"],
    "reaction-pressure-archives": ["tests/test_reaction_pressure_archives.py"],
    "refinement-audit": ["tests/test_normal_refinement_audit.py"],
    "prototype": ["tests/test_mac_normal_constraint.py"],
    "response": ["tests/test_mac_normal_response.py", "-k", "not reverse_geometry and not chunked_response"],
    "response-gradient": ["tests/test_mac_normal_response.py::test_reverse_geometry_derivative_includes_response_rebuild"],
    "response-chunk": ["tests/test_mac_normal_response.py::test_chunked_response_matches_separate_steps"],
    "response-driver": ["tests/test_normal_cylinder_driver.py", "tests/test_cylinder_guard.py", "tests/test_run_cylinder_verification_captured.py"],
    "normal-fe": ["tests/test_fe_normal_constraint.py", "tests/test_dolfinx_brinkman.py"],
    "normal-calibration": ["tests/test_planar_mask_limit.py", "tests/test_continuum_mask_calibration.py"],
    "native-pressure": ["tests/test_native_pressure_trace.py", "tests/test_pressure_trace_diagnostic.py"],
    "native-pressure-archives": ["tests/test_native_pressure_archives.py"],
    "continuum-gradient": ["tests/test_continuum_normal_gradient.py"],
    "pressure-preconditioner": ["tests/test_mac_pressure_preconditioner.py"],
    "pressure-trace": ["tests/test_pressure_trace_diagnostic.py", "tests/test_dolfinx_brinkman.py", "tests/test_normal_cylinder_driver.py"],
    "regression": ["tests/test_native_face_brinkman.py", "tests/test_brinkman_pressure_response.py",
                   "tests/test_run_cylinder_verification_captured.py", "tests/test_cylinder_guard.py"],
}


def snapshot():
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                python_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in paths if p.endswith(".py") and (ROOT/p).is_file()})


def child(args):
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only supervisor required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("single-thread numerical libraries required")
    for key, limit in ((resource.RLIMIT_AS, 4*1024**3), (resource.RLIMIT_CPU, 60)):
        soft, hard = resource.getrlimit(key)
        cap = min([limit]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(key, (cap, cap))
    affinity = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, affinity)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, str(ROOT))
    # This mixed CPU/CUDA installation eagerly imports accelerator DSOs even
    # with JAX_PLATFORMS=cpu, exhausting a 4 GiB *virtual* address-space cap.
    # Hide optional accelerator packages ONLY in this isolated CPU child;
    # do not modify site-packages, JAX solvers, installed drivers or the host.
    hidden = ["jax_cuda12_plugin", "jax_cuda13_plugin", "jax_rocm60_plugin",
              "jax_plugins.xla_cuda12", "jax_plugins.xla_cuda13", "jax_plugins.xla_rocm60"]
    for name in hidden:
        sys.modules[name] = None
    # Scientific imports happen only after the hard resource limits.
    import pytest
    start = time.monotonic()
    code = pytest.main(["-q", "-s", *SUITES[args.suite], "--junitxml", str(args.output_dir/"junit.xml")])
    import jax
    import numpy
    import scipy
    record = dict(exit_code=int(code), wall_seconds=time.monotonic()-start,
                  peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  affinity=affinity, address_space_bytes=4*1024**3, cpu_seconds=60,
                  hidden_accelerator_packages=hidden,
                  software=dict(python=sys.version, platform=platform.platform(), jax=jax.__version__,
                                numpy=numpy.__version__, scipy=scipy.__version__, pytest=pytest.__version__,
                                devices=[str(d) for d in jax.devices()], backend=jax.default_backend(), precision="float64"),
                  paper_eligible=False, experiment_kind="bounded-unit-verification-not-cylinder")
    (args.output_dir/"child.json").write_text(json.dumps(record, indent=2)+"\n")
    return int(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="prototype")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.child:
        return child(args)
    from guard_cylinder_run import sample, rejection, interrupt_group
    output = args.output_dir
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("need a new external output directory")
    before = snapshot()
    if args.require_clean and before["status"]:
        raise RuntimeError("clean source required")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--suite", args.suite, "--output-dir", str(output)]
    process, reason = None, None
    started = time.monotonic()
    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    with (ROOT.parent/".pressure-response-cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = sample(output.parent)
        problem = rejection(initial, startup=True, low_disk_exception=True)
        if problem:
            raise RuntimeError(problem)
        output.mkdir(exist_ok=False)
        (output/"source.json").write_text(json.dumps(before, indent=2)+"\n")
        try:
            with (output/"hardware.csv").open("x", newline="") as stream, (output/"stdout.log").open("xb") as stdout, (output/"stderr.log").open("xb") as stderr:
                writer = csv.DictWriter(stream, list(initial)); writer.writeheader(); writer.writerow(initial); stream.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                while process.poll() is None:
                    values = sample(output.parent)
                    writer.writerow(values); stream.flush()
                    reason = rejection(values, startup=False, low_disk_exception=True)
                    if time.monotonic()-started > 120:
                        reason = "120 second wall timeout"
                    if reason:
                        interrupt_group(process)
                        break
                    time.sleep(2)
        except BaseException as error:
            reason = f"{type(error).__name__}: {error}"
            if process is not None:
                interrupt_group(process)
            raise
        finally:
            after = snapshot()
            if after != before:
                reason = reason or "source changed during tests"
            record = dict(command=command, initial=initial, low_disk_exception=True,
                          address_space_bytes=4*1024**3, cpu_seconds=60, wall_seconds_limit=120,
                          stop_reason=reason, source_unchanged=after == before,
                          elapsed_seconds=time.monotonic()-started,
                          exit_code=None if process is None else process.poll())
            (output/"guard.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps(record, indent=2), flush=True)
        return 2 if reason else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
