#!/usr/bin/env python3
"""Bounded CPU Docker inner runner; invoke via run_cylinder_verification_captured.py.

No GPU access, dependency download, existing-data deletion, or hardware setting
changes. Hard 8 GiB container memory/no swap, two CPU quota, 20 minute timeout.
The explicitly user-approved disk exception keeps at least 30 GiB available.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from dolfinx_brinkman_cylinder import ROOT, load_model
from guard_cylinder_run import sample, rejection


def resolve_solver(config_path):
    config = json.loads(Path(config_path).read_text())
    if config.get("schema_version") == "dolfinx-finite-brinkman-config-1.0":
        config, _, _, _ = load_model(config_path)
        return config, config["container"]["image"], ROOT/"tools/dolfinx_brinkman_cylinder.py"
    if "pressure_trace_diagnostic" in config:
        import dolfinx_dfg2d1 as evidence
        from pressure_trace_diagnostic import validate
        config = evidence._load_config(config_path)
        validate(config)
        return config, "dolfinx/dolfinx@"+evidence.IMAGE_DIGEST, ROOT/"tools/dolfinx_dfg2d1.py"
    raise ValueError("only bounded finite-model or pinned sharp-trace diagnostics allowed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_dir.resolve()
    config, image, solver_script = resolve_solver(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing nonempty output")
    # Output outside the source tree avoids dirtying source during the solve.
    if output.is_relative_to(ROOT):
        raise ValueError("CPU FE evidence must be generated outside repository")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    inspection = json.loads(subprocess.check_output(["docker", "image", "inspect", image], timeout=20))
    output.mkdir(parents=True, exist_ok=True)
    guard = output.with_name(output.name + ".guard")
    guard.mkdir(exist_ok=False)
    name = "brinkman-diagnostic-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--pull=never", "--network=none",
               "--memory=8g", "--memory-swap=8g", "--cpus=2", "--pids-limit=128",
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
               "--mount", f"type=bind,src={output},dst={output}", "--workdir", str(ROOT),
               "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
               "--env", "CUDA_VISIBLE_DEVICES=", "--env", "XDG_CACHE_HOME=/tmp/cache",
               "--env", "XDG_CONFIG_HOME=/tmp/config", image, "python3",
               str(solver_script), "--config", str(config_path),
               "--output-dir", str(output)]
    (guard / "launch.json").write_text(json.dumps(dict(command=command, image_inspection=inspection,
        user_approved_disk_exception=True, timeout_seconds=1200), indent=2)+"\n")
    process = None
    reason = None
    started = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (ROOT.parent / ".brinkman-cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = sample(output)
        problem = rejection(initial, startup=True, low_disk_exception=True)
        if problem:
            raise RuntimeError(problem)
        try:
            with (guard / "hardware.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(initial))
                writer.writeheader()
                writer.writerow(initial)
                stream.flush()
                print(json.dumps(dict(docker_command=command)), flush=True)
                process = subprocess.Popen(command)
                while process.poll() is None:
                    current = sample(output)
                    writer.writerow(current)
                    stream.flush()
                    reason = rejection(current, startup=False, low_disk_exception=True)
                    if time.monotonic()-started > 1200:
                        reason = "20 minute CPU timeout"
                    if reason:
                        break
                    time.sleep(2)
        except BaseException as error:
            reason = f"{type(error).__name__}: {error}"
        finally:
            # Stopping the Docker CLI alone can orphan the solver. Address ONLY
            # this launch's unique container, and let Docker enforce the deadline.
            if process is not None and process.poll() is None:
                subprocess.run(["docker", "stop", "--time", "15", name], timeout=30, check=False)
                process.wait(timeout=30)
            record = dict(stop_reason=reason, elapsed_seconds=time.monotonic()-started,
                          exit_code=None if process is None else process.returncode)
            (guard / "guard.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        return 2 if reason else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
