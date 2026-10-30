"""Run every included test file in an isolated process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("JAX_ENABLE_X64", "true")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    failures = []
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        print(f"\n=== {test.name} ===", flush=True)
        command = [sys.executable, "-m", "pytest", "-q", str(test)]
        if args.report_dir:
            command += ["--junitxml", str(args.report_dir.resolve() / (test.stem + ".xml"))]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            failures.append(test.name)
    if failures:
        print("Failed test files: " + ", ".join(failures))
        return 1
    print("All included test files passed (device-dependent skips may apply).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
