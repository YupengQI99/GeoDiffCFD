"""Authenticate segmented optimisation without resetting its original criterion.

No JAX import, no writes, and no geometry/physics changes. A new configuration
pins its parent's result bytes; the captured guard authenticates companion
configuration and evidence. Failed or hardware-stopped runs are not resumed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from multibody_wake_config import MODEL_KEYS, validate_config


def _read_pinned(record, root):
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("changed pinned optimisation/gradient record")
    return path, json.loads(raw)


def optimization_context(c, root, _seen=()):
    """Recover the original gradient reference and exact next design step."""
    root = Path(root)
    validate_config(c)
    context = dict(initial_theta=c["theta"], reference_gradient_norm=None,
                   accepted_steps_before=0, initial_step=.60, parent=None)
    if "continuation" in c:
        path, data = _read_pinned(c["continuation"], root)
        if path in _seen or len(_seen) >= 100:
            raise ValueError("cyclic or excessive continuation chain")
        previous = json.loads((path.parent / "config.json").read_text())
        if any(previous[k] != c[k] for k in MODEL_KEYS):
            raise ValueError("continuation changed the fixed physical/discrete model")
        guard = path.parent.with_name(path.parent.name + ".guard")
        safety = json.loads((guard / "guard.json").read_text())
        if not (safety["source_unchanged"] and safety["exit_code"] == 0 and safety["stop_reason"] is None):
            raise ValueError("cannot resume failed or hardware-stopped segment")
        hashes = json.loads((guard / "artifacts.json").read_text())
        for required in (path, path.parent / "config.json", guard / "guard.json"):
            name = str(required.relative_to(path.parent.parent))
            if name not in hashes:
                raise ValueError("parent capture is missing required checksums")
        for name, digest in hashes.items():
            item = (path.parent.parent / name).resolve()
            if not (item.is_relative_to(path.parent) or item.is_relative_to(guard)):
                raise ValueError("parent checksum target outside captured directories")
            if hashlib.sha256(item.read_bytes()).hexdigest() != digest:
                raise ValueError("parent capture checksum mismatch")
        opt = data["optimization"]
        if data["paper_eligible"] is not False or data["mode"] != "optimize":
            raise ValueError("parent must be a development optimisation")
        if opt["converged"] or opt["termination"] != "max_iterations":
            raise ValueError("resume only a successful nonconverged budget segment")
        if c["theta"] != opt["coefficients"]:
            raise ValueError("continuation must start at the exact parent candidate")
        if c["gradient_gate_records"] != previous["gradient_gate_records"]:
            raise ValueError("continuation must preserve original gradient evidence")
        context = optimization_context(previous, root, (*_seen, path))
        accepted = [t for t in opt["trace"] if t["accepted"]]
        if not accepted:
            raise ValueError("parent budget segment made no accepted progress")
        ref = context["reference_gradient_norm"]
        if ref is None:
            ref = opt["trace"][0]["projected_gradient_norm"]
        if not math.isfinite(ref) or ref <= 0:
            raise ValueError("invalid original projected-gradient reference")
        if "reference_gradient_norm" in opt and opt["reference_gradient_norm"] != ref:
            raise ValueError("parent changed the original convergence reference")
        final = data["evaluations"][-1]
        if (final["theta"] != opt["coefficients"] or final["gradient"] != opt["final_gradient"]
                or final["objective"] != opt["final_value"]):
            raise ValueError("parent terminal gradient evidence does not match candidate")
        context = dict(initial_theta=context["initial_theta"], reference_gradient_norm=ref,
                       accepted_steps_before=context["accepted_steps_before"] + len(accepted),
                       initial_step=accepted[-1]["step_length"], parent=dict(
                           record=c["continuation"], value=opt["final_value"],
                           gradient=opt["final_gradient"]))

    gates = c.get("gradient_gate_records", [])
    if len(gates) < 2:
        raise ValueError("initial and distinct-state gradient gates required")
    shapes = []
    for gate in gates:
        path, data = _read_pinned(gate, root)
        previous = json.loads((path.parent / "config.json").read_text())
        if any(previous[k] != c[k] for k in MODEL_KEYS):
            raise ValueError("different gradient-gate model")
        audit = data.get("gradient_audit", {})
        if audit.get("passed") is not True or len(audit["finite_difference_checks"]) != 4:
            raise ValueError("passed full-trajectory per-body FD evidence required")
        shapes.append(previous["theta"])
    initial = context["initial_theta"]
    if not any(t == initial for t in shapes) or not any(t != initial for t in shapes):
        raise ValueError("gates must include ORIGINAL start and a distinct state")
    return context
