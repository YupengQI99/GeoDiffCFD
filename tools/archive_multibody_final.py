#!/usr/bin/env python3
"""Archive final-design checks byte-for-byte, retaining the interrupted attempt."""
import argparse
import hashlib
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    root = Path(__file__).resolve().parents[1]
    if dst.exists() or not dst.is_relative_to(root / "runs"):
        raise ValueError("new directory under repository runs/ required")
    runs = ["gradient_final_v1", "gradient_final_retry_v1", "medium_final_v1",
            "half_dt_final_v1", "late_window_final_v1"]
    names = runs + [name + ".guard" for name in runs] + [
        "cpu_continuation_clean_v1", "cpu_flow_clean_v1", "cpu_final_summary_clean_v1",
        "raw_audit_v1.json", "summary.json", "summary.csv", "figures", "verification_figures",
    ]
    files = []
    for name in names:
        path = src / name
        if not path.exists():
            raise FileNotFoundError(path)
        for entry in ([path] if path.is_file() else path.rglob("*")):
            if entry.is_symlink():
                raise ValueError("no symlink archive inputs")
            if entry.is_file():
                files.append(entry)
    total = sum(path.stat().st_size for path in files)
    if total > 512 * 1024**2 or shutil.disk_usage(root).free-total < 30 * 1024**3:
        raise ValueError("archive size or 30 GiB reserve would be exceeded")
    dst.mkdir(parents=True, exist_ok=False)
    for name in names:
        path = src / name
        if path.is_dir():
            shutil.copytree(path, dst / name)
        else:
            shutil.copy2(path, dst / name)
    sums = []
    for path in sorted(files):
        relative = path.relative_to(src)
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        copied = hashlib.sha256((dst / relative).read_bytes()).hexdigest()
        if original != copied:
            raise RuntimeError(f"copy changed bytes: {relative}")
        sums.append(f"{copied}  {relative}")
    (dst / "SHA256SUMS").write_text("\n".join(sums)+"\n")
    print(f"Copied and verified {len(files)} artifacts ({total} bytes) to {dst}")


if __name__ == "__main__":
    main()
