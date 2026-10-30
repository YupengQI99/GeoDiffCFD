#!/usr/bin/env python3
"""Supervise ONE captured cylinder run; never changes GPU settings or deletes data.

The ordinary low-disk exception is limited to <=440x82 development runs;
each specifically authorized 880x164 case needs its own matching fine flag.
Hardware samples and supervisor logs live beside, not inside, the captured run.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def gpu_sample():
    raw = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    memory, temperature, power = map(float, raw.strip().split(","))
    if not all(math.isfinite(v) for v in (memory, temperature, power)):
        raise ValueError("nonfinite GPU telemetry")
    return dict(gpu_memory_mib=memory, temperature_c=temperature, power_w=power)


def gpu_pids():
    raw = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    return [int(line) for line in raw.splitlines() if line.strip()]


def sample(path):
    values = gpu_sample()
    temperatures = {"cpu_temperature_c": [], "nvme_temperature_c": []}
    for device in Path("/sys/class/hwmon").glob("hwmon*"):
        name_path = device / "name"
        if not name_path.is_file():
            continue
        name = name_path.read_text().strip()
        key = ("cpu_temperature_c" if name in ("k10temp", "coretemp") else
               "nvme_temperature_c" if name == "nvme" else None)
        if key:
            for sensor in device.glob("temp*_input"):
                value = float(sensor.read_text()) / 1000.0
                if not math.isfinite(value):
                    raise ValueError("nonfinite host temperature telemetry")
                temperatures[key].append(value)
    values.update({key: max(readings) if readings else None for key, readings in temperatures.items()})
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0]) * 1024
    disk = shutil.disk_usage(path)
    values.update(
        unix_time=time.time(), available_ram_fraction=mem["MemAvailable"] / mem["MemTotal"],
        disk_free_gib=disk.free / GIB,
        disk_used_fraction=disk.used / (disk.used + disk.free),
    )
    return values


def rejection(values, *, startup, low_disk_exception):
    minimum_disk = 30 if low_disk_exception else (100 if startup else 10)
    if values["disk_free_gib"] < minimum_disk:
        return f"disk below {minimum_disk} GiB reserve"
    if startup and not low_disk_exception and values["disk_used_fraction"] > 0.92:
        return "filesystem over 92% used"
    if values["available_ram_fraction"] < (0.20 if startup else 0.10):
        return "insufficient available host RAM"
    if values["gpu_memory_mib"] > 22000:
        return "GPU memory over 22000 MiB"
    # More conservative than the handoff's 82 C for 30 seconds.
    if values["temperature_c"] >= (75 if startup else 82):
        return "GPU temperature limit"
    if values.get("cpu_temperature_c") is not None and values["cpu_temperature_c"] >= (85 if startup else 90):
        return "CPU temperature limit"
    if values.get("nvme_temperature_c") is not None and values["nvme_temperature_c"] >= (70 if startup else 75):
        return "NVMe temperature limit"
    return None


def check_scope(config, low_disk_exception, *, fine_approved=False, fixed_model_fine_approved=False):
    has_fine = "fine_authorization" in config
    has_fixed_fine = "fixed_model_fine_authorization" in config
    if fine_approved and fixed_model_fine_approved:
        raise ValueError("fine approvals refer to separate cases and cannot be combined")
    if has_fine != fine_approved:
        raise ValueError("fine configuration and explicit user-approved-fine flag must match")
    if has_fixed_fine != fixed_model_fine_approved:
        raise ValueError("fixed-model fine configuration and explicit approval flag must match")
    if fixed_model_fine_approved:
        from run_normal_cylinder import validate_fixed_model_fine
        validate_fixed_model_fine(config)
        if not low_disk_exception:
            raise ValueError("fixed-model fine control retains the explicit 30 GiB disk reserve")
    if fine_approved:
        from run_normal_cylinder import FINE_AUTHORIZATION
        if config.get("fine_authorization") != FINE_AUTHORIZATION or config.get("grid") != dict(nx=880,ny=164,lx=2.2,ly=.41):
            raise ValueError("fine approval is limited to one declared 880x164 verification")
        if not low_disk_exception:
            raise ValueError("fine run on this disk requires the explicit retained 30 GiB reserve")
    if low_disk_exception:
        if config.get("study", {}).get("paper_eligible") is not False:
            raise ValueError("low-disk exception requires a development-only configuration")
        if config.get("study", {}).get("development_only") is not True:
            raise ValueError("low-disk exception requires development_only=true")
        grid = config["grid"]
        if (grid["nx"] > 440 or grid["ny"] > 82) and not (fine_approved or fixed_model_fine_approved):
            raise ValueError("low-disk exception does not authorize fine grids")


def interrupt_group(process):
    """Signal only our new process group; allow evidence finalization."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--user-approved-low-disk", action="store_true")
    parser.add_argument("--user-approved-fine", action="store_true", help="one explicitly authorized segmented 880x164 verification; all hardware gates retained")
    parser.add_argument("--user-approved-fixed-model-fine", action="store_true", help="separately approved single 880x164 control retaining the medium physical model and 48 markers")
    parser.add_argument("--normal-constraint", action="store_true", help="use the opt-in total-reaction cylinder diagnostic")
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("timeout must be in (0, 3600] seconds; no unattended long runs")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if args.normal_constraint:
        sys.path.insert(0, str(ROOT))
        from run_normal_cylinder import load_config
        load_config(config_path)
    elif config.get("schema_version") == "normal-cylinder-diagnostic-config-1.0":
        raise ValueError("normal-constraint configuration requires its explicit runner")
    if (args.user_approved_fine or args.user_approved_fixed_model_fine) and not args.normal_constraint:
        raise ValueError("fine approval applies only to the declared normal-constraint diagnostic")
    if args.user_approved_fixed_model_fine and args.timeout_seconds > 1800:
        raise ValueError("fixed-model fine control requires a segment of at most 1800 seconds")
    check_scope(config, args.user_approved_low_disk, fine_approved=args.user_approved_fine,
                fixed_model_fine_approved=args.user_approved_fixed_model_fine)
    # Refuse unidentified source, including staged changes. Untracked outputs
    # are permitted by the repository's compute evidence contract.
    for command in (["git", "diff", "--exit-code"], ["git", "diff", "--cached", "--exit-code"]):
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "ls-files", "--error-unmatch", str(config_path)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    guard_dir = output.with_name(output.name + ".guard")
    # Cooperating launches share a lock, even when their artifact disks differ.
    with (ROOT / "runs" / ".cylinder-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = sample(output.parent)
        problem = rejection(initial, startup=True, low_disk_exception=args.user_approved_low_disk)
        if problem or gpu_pids():
            raise RuntimeError(problem or "GPU already has a compute process")
        guard_dir.mkdir(exist_ok=False)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="0", JAX_ENABLE_X64="true", JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="true", XLA_PYTHON_CLIENT_MEM_FRACTION="0.70",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        )
        # Do not inherit an allocator override that could defeat the memory cap.
        environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
        command = [sys.executable, str(ROOT / "tools/run_cylinder_verification_captured.py"),
                   "--config", str(config_path), "--output-dir", str(output)]
        if args.normal_constraint:
            command += ["--inner-runner", str(ROOT / "tools/run_normal_cylinder.py")]
        started = time.monotonic()
        process = None
        reason = None
        peak = initial["gpu_memory_mib"]
        max_temperature = initial["temperature_c"]
        try:
            with (guard_dir / "hardware.csv").open("x", newline="") as hardware, \
                 (guard_dir / "stdout.log").open("xb") as stdout, \
                 (guard_dir / "stderr.log").open("xb") as stderr:
                writer = csv.DictWriter(hardware, fieldnames=list(initial))
                writer.writeheader()
                writer.writerow(initial)
                hardware.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                while process.poll() is None:
                    current = sample(output.parent)
                    writer.writerow(current)
                    hardware.flush()
                    peak = max(peak, current["gpu_memory_mib"])
                    max_temperature = max(max_temperature, current["temperature_c"])
                    reason = rejection(current, startup=False, low_disk_exception=args.user_approved_low_disk)
                    for pid in gpu_pids():
                        try:
                            if os.getpgid(pid) != process.pid:
                                reason = "another GPU compute process appeared"
                        except ProcessLookupError:
                            pass
                    if time.monotonic() - started > args.timeout_seconds:
                        reason = "supervisor timeout"
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
            record = dict(command=command, low_disk_exception=args.user_approved_low_disk,
                          initial=initial, sampled_peak_gpu_mib=peak,
                          sampled_max_temperature_c=max_temperature, stop_reason=reason,
                          elapsed_seconds=time.monotonic() - started,
                          exit_code=None if process is None else process.poll())
            (guard_dir / "guard.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        return 2 if reason else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
