#!/usr/bin/env python3
"""Capture a portable compute-environment manifest for experiment handoff."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_ENVIRONMENT_VARIABLES = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "JAX_COMPILATION_CACHE_DIR",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "XLA_FLAGS",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _tracked_tree_is_dirty() -> bool | None:
    """Ignore untracked run artifacts but reject source/index modifications."""

    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], check=False
        ).returncode
    except FileNotFoundError:
        return None
    return unstaged != 0 or staged != 0


def _nvidia_inventory() -> list[dict[str, Any]]:
    """Return visible NVIDIA hardware metadata when nvidia-smi is available."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    inventory = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory_mib, driver = parts
        try:
            memory_bytes = int(float(memory_mib) * 1024 * 1024)
        except ValueError:
            memory_bytes = None
        inventory.append(
            {
                "platform": "cuda",
                "name": name,
                "count": 1,
                "memory_bytes": memory_bytes,
                "driver": driver or None,
            }
        )
    return inventory


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _cuda_toolkit_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _jax_devices() -> tuple[str | None, str | None, list[dict[str, Any]]]:
    try:
        import jax
    except Exception:
        return None, None, []
    devices = list(jax.devices())
    platform_version = None
    if devices:
        platform_version = str(
            getattr(getattr(devices[0], "client", None), "platform_version", None)
        )
        if platform_version == "None":
            platform_version = None
    grouped: dict[tuple[str, str], int] = {}
    for device in devices:
        key = (str(device.platform), str(getattr(device, "device_kind", device)))
        grouped[key] = grouped.get(key, 0) + 1
    accelerators = [
        {
            "platform": platform_name,
            "name": name,
            "count": count,
            "memory_bytes": None,
            "driver": None,
        }
        for (platform_name, name), count in sorted(grouped.items())
    ]
    nvidia = _nvidia_inventory()
    if devices and devices[0].platform == "gpu" and nvidia:
        accelerators = nvidia
    return (
        devices[0].platform if devices else None,
        platform_version,
        accelerators,
    )


def capture(precision: str) -> dict[str, Any]:
    backend, backend_platform_version, accelerators = _jax_devices()
    return {
        "git": {
            "repository": _run_git("config", "--get", "remote.origin.url"),
            "commit": _run_git("rev-parse", "HEAD"),
            "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": _tracked_tree_is_dirty(),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "process_cpu_affinity": _cpu_affinity(),
            "memory_bytes": _memory_bytes(),
            "accelerators": accelerators,
            "device_visibility": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "unset_policy": "runtime default; all enumerated devices may be visible",
            },
        },
        "software": {
            "python": sys.version.split()[0],
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "jax_cuda_plugin": _package_version("jax-cuda12-plugin"),
            "jax_cuda_pjrt": _package_version("jax-cuda12-pjrt"),
            "backend": backend,
            "backend_platform_version": backend_platform_version,
            "cuda_toolkit_nvcc": _cuda_toolkit_version(),
            "precision": precision,
            "environment_variables": {
                name: os.environ[name]
                for name in TRACKED_ENVIRONMENT_VARIABLES
                if name in os.environ
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=("float32", "float64", "mixed", "other"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(capture(args.precision), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
