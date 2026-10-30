"""Helpers for auditable CPU/GPU experiment run directories."""

from __future__ import annotations

import hashlib
import json
import resource
from pathlib import Path
from typing import Any, Callable

import jax


def block_tree(tree: Any) -> Any:
    """Synchronize every JAX leaf and return the original PyTree."""

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def compile_and_sample(
    function: Callable[[], Any], samples: int
) -> tuple[Any, float, list[float]]:
    """Compile a nullary JAX function, then record synchronized executions."""

    import time

    if samples < 1:
        raise ValueError("samples must be at least one")
    started = time.perf_counter()
    compiled = jax.jit(function).lower().compile()
    compile_seconds = time.perf_counter() - started
    durations: list[float] = []
    result = None
    for _ in range(samples):
        started = time.perf_counter()
        result = block_tree(compiled())
        durations.append(time.perf_counter() - started)
    return result, compile_seconds, durations


def compile_and_sample_args(
    function: Callable[..., Any],
    args: tuple,
    samples: int,
    *,
    static_argnums: tuple[int, ...] = (),
) -> tuple[Any, float, list[float]]:
    """Compile ``function`` with ``args`` passed as arguments, then time it.

    Unlike :func:`compile_and_sample`, arrays are *not* closed over as
    constants.  On the RTX 4090 with JAX 0.7.0 a compiled program whose
    880 x 164 float64 operands were embedded as constants produced a wrong
    projection step (archived runs cff6489 and 9e71539,
    development-joint-q1-fine): the same step with the arrays passed as
    arguments agrees with the CPU to round-off.  Every GPU forward run must
    therefore use this helper.  Entries of ``static_argnums`` are Python
    scalars baked into the executable and are not passed at call time.
    """

    import time

    if samples < 1:
        raise ValueError("samples must be at least one")
    jitted = jax.jit(function, static_argnums=static_argnums)
    started = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    compile_seconds = time.perf_counter() - started
    dynamic = tuple(a for i, a in enumerate(args) if i not in static_argnums)
    durations: list[float] = []
    result = None
    for _ in range(samples):
        started = time.perf_counter()
        result = block_tree(compiled(*dynamic))
        durations.append(time.perf_counter() - started)
    return result, compile_seconds, durations


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path, base: Path, *, media_type: str, role: str
) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(base.resolve()).as_posix(),
        "sha256": sha256(path),
        "media_type": media_type,
        "role": role,
    }


def peak_memory_record() -> tuple[int | None, int | None, str]:
    """Read process RSS and JAX allocator peak after synchronized execution."""

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. This repository's production
    # paths are Linux, but retain a conservative platform check.
    import sys

    peak_host = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    peak_accelerator = None
    method = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    devices = jax.devices()
    if devices:
        try:
            stats = devices[0].memory_stats()
        except Exception:
            stats = None
        if stats:
            for key in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
                value = stats.get(key)
                if value is not None:
                    peak_accelerator = int(value)
                    method += f"; JAX device.memory_stats()['{key}']"
                    break
    return peak_host, peak_accelerator, method


def no_gradient_record(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "backend": "none",
        "parameterization": None,
        "parameter_names": [],
        "values": [],
        "norm": None,
        "adjoint_residual": None,
        "message": message,
        "taylor_test": None,
    }


def provenance_record(*, precision: str = "float64") -> dict[str, Any]:
    """Git, software, hardware and timestamp metadata for an archived record.

    The record is intentionally self-contained so that a JSON file under
    ``output/data`` can be audited without the surrounding run directory.
    ``git_dirty`` reports tracked or staged modifications of *source* files
    only: untracked files and regenerated records under ``output/data`` (the
    outputs of the very experiments being run) do not change the source state.
    """

    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    import jaxlib
    import numpy
    import scipy

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    dirty_probe = git(
        "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)output/data"
    )
    devices = jax.devices()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if dirty_probe is None else bool(dirty_probe),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kind": devices[0].device_kind if devices else None,
        "precision": precision,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
