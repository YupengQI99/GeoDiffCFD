import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from multibody_wake_config import validate_config
from multibody_wake_continuation import optimization_context


def store(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def records(tmp_path):
    c = json.loads((ROOT / "configs/multibody_wake/coarse_optimization.json").read_text())
    gates = []
    for index in range(2):
        gate_config = copy.deepcopy(c)
        gate_config["theta"] = [index * .1] * 12
        path = tmp_path / f"gradient{index}" / "result.json"
        gates.append(store(path, dict(gradient_audit=dict(passed=True, finite_difference_checks=[{}] * 4))))
        store(path.parent / "config.json", gate_config)
    c["gradient_gate_records"] = gates
    path = tmp_path / "parent" / "result.json"
    theta = [.1] * 12; gradient = [.2] * 12
    opt = dict(coefficients=theta, final_gradient=gradient, final_value=1.,
               converged=False, termination="max_iterations",
               trace=[dict(accepted=True, step_length=.3, projected_gradient_norm=2.)])
    pinned = store(path, dict(mode="optimize", paper_eligible=False, optimization=opt,
                              evaluations=[dict(theta=theta, gradient=gradient, objective=1.)]))
    store(path.parent / "config.json", c)
    guard = tmp_path / "parent.guard"
    store(guard / "guard.json", dict(source_unchanged=True, exit_code=0, stop_reason=None))
    def manifest():
        files = [path, path.parent / "config.json", guard / "guard.json"]
        store(guard / "artifacts.json", {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    manifest()
    new = dict(c, theta=theta, max_optimization_steps=4, continuation=pinned)
    return new, path, guard, manifest


def test_resume_recovers_original_reference_and_last_accepted_step(records):
    c, path, guard, manifest = records
    context = optimization_context(c, ROOT)
    assert context["reference_gradient_norm"] == 2.
    assert context["accepted_steps_before"] == 1
    assert context["initial_theta"] == [0] * 12
    assert context["initial_step"] == .3
    assert context["parent"]["value"] == 1.


@pytest.mark.parametrize("change", ["model", "theta", "hash", "hardware", "config", "gates"])
def test_resume_rejects_inconsistent_or_unsafe_parent(records, change):
    c, path, guard, manifest = records
    if change == "model": c["alpha_max"] = 161
    elif change == "theta": c["theta"] = [.2] * 12
    elif change == "hash": c["continuation"]["sha256"] = "0" * 64
    elif change == "hardware":
        store(guard / "guard.json", dict(source_unchanged=True, exit_code=0, stop_reason="GPU temperature limit"))
        manifest()
    elif change == "config":
        previous = json.loads((path.parent / "config.json").read_text())
        previous["theta"] = [.2] * 12
        store(path.parent / "config.json", previous)
    elif change == "gates": c["gradient_gate_records"] = c["gradient_gate_records"][:1]
    with pytest.raises(ValueError): optimization_context(c, ROOT)


def test_continuation_segment_is_smaller_but_not_new_global_budget(records):
    c, *_ = records
    validate_config(c)
    for change in (dict(max_optimization_steps=5), dict(continuation={}), dict(theta=[1.01]*12)):
        with pytest.raises(ValueError): validate_config(dict(c, **change))
