"""Rewrite ``output/data/SHA256SUMS`` so that it covers every archived record.

Usage::

    python tools/update_data_checksums.py          # rewrite the manifest
    python tools/update_data_checksums.py --check  # exit 1 if the manifest is stale

Paths inside the manifest are relative to the repository root so that
``sha256sum -c output/data/SHA256SUMS`` works from there.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "data"
MANIFEST = DATA / "SHA256SUMS"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_text() -> str:
    lines = []
    for path in sorted(DATA.iterdir()):
        if path.name == MANIFEST.name or path.is_dir():
            continue
        lines.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = manifest_text()
    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if current != text:
            print("output/data/SHA256SUMS is stale; run tools/update_data_checksums.py", file=sys.stderr)
            return 1
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(text.splitlines())} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
