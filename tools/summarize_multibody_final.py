#!/usr/bin/env python3
"""Rebuild audited final-design comparisons without modifying manuscript files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODEL_KEYS = (
    "geometry", "grid", "viscosity", "dt", "n_steps", "chunk_steps",
    "window_start_step", "mean_inlet", "reference_diameter", "alpha_max",
    "epsilon", "advection_scheme",
)
PHYSICS = (
    "jax_cad/differentiable/mac_channel.py",
    "jax_cad/differentiable/wake_design.py",
    "jax_cad/differentiable/multibody_wake.py",
    "jax_cad/unified/geometry.py", "jax_cad/unified/materials.py",
    "jax_cad/verification.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pg_norm(theta, gradient):
    return math.sqrt(sum((z - min(1., max(-1., z-g)))**2
                         for z, g in zip(theta, gradient, strict=True)))


def check_pair(initial, final, final_theta):
    require(all(initial[k] == final[k] for k in MODEL_KEYS), "paired models differ")
    require(initial["theta"] == [0.] * 12, "initial geometry is not the original")
    require(final["theta"] == final_theta, "candidate is not the final design")


def criteria_for(comparisons, gradient_passed, stationary):
    gain = comparisons[0]["initial_objective"] - comparisons[0]["candidate_objective"]
    require(gain > 0, "nonpositive original absolute improvement")
    for row in comparisons:
        row["changes_relative_to_original_gain"] = {
            k: abs(row[k+"_objective"] - comparisons[0][k+"_objective"]) / gain
            for k in ("initial", "candidate")
        }
    return dict(
        positive_gain_all_comparisons=all(row["reduction_percent"] > 0 for row in comparisons),
        time_and_window_changes_below_tenth_of_gain=all(
            max(row["changes_relative_to_original_gain"].values()) < .1
            for row in comparisons[2:]),
        final_full_trajectory_gradient_gate=gradient_passed,
        original_projected_gradient_stop=stationary,
    )


def build(repo, new, audit_path=None):
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def captured(directory):
        guard = directory.with_name(directory.name + ".guard")
        safety = read(guard / "guard.json")
        require(safety["exit_code"] == 0 and safety["stop_reason"] is None
                and safety["source_unchanged"], f"unsuccessful guard: {directory}")
        for name, digest in read(guard / "artifacts.json").items():
            path = (directory.parent / name).resolve()
            require(path.is_relative_to(directory.resolve()) or path.is_relative_to(guard.resolve()),
                    "capture path outside run")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"capture mismatch: {name}")
        result, config = read(directory / "result.json"), read(directory / "config.json")
        source = read(guard / "source.json")
        require(result["paper_eligible"] is False and result["provenance"]["backend"] == "gpu",
                "expected development GPU evidence")
        require(not source["status"], "dirty captured source")
        for key in PHYSICS:
            actual = hashlib.sha256((repo / key).read_bytes()).hexdigest()
            require(source["python_sha256"][key] == actual, f"changed physics: {key}")
            hashes[str((repo / key).resolve())] = actual
        for evaluation in result["evaluations"]:
            require(read(directory / evaluation["directory"] / "metrics.json") == evaluation,
                    "result and per-evaluation metrics disagree")
        return config, result, safety

    old = repo / "runs/multibody_wake_diagnostics"
    terminal = repo / "runs/multibody_wake_continuation_20260921/continuation_01_retry"
    c_old, r_old, _ = captured(old / "optimization_v1")
    c_final, r_final, _ = captured(terminal)
    opt = r_final["optimization"]
    theta = opt["coefficients"]
    require(all(c_old[k] == c_final[k] for k in MODEL_KEYS), "continuation changed the model")
    require(c_final["theta"] == r_old["optimization"]["coefficients"], "wrong continuation parent")
    parent_path = repo / c_final["continuation"]["path"]
    require(parent_path.resolve() == (old / "optimization_v1/result.json").resolve(), "wrong parent path")
    require(hashlib.sha256(parent_path.read_bytes()).hexdigest() == c_final["continuation"]["sha256"],
            "parent pin mismatch")
    original_ref = r_old["optimization"]["trace"][0]["projected_gradient_norm"]
    require(opt["reference_gradient_norm"] == original_ref, "reset projected-gradient reference")
    pg = pg_norm(theta, opt["final_gradient"])
    require(math.isclose(pg, opt["final_projected_gradient_norm"], rel_tol=1e-12), "wrong final PG")
    stationary = bool(opt["converged"] and opt["termination"] == "projected_gradient_tolerance"
                      and pg <= .02 * original_ref)
    require(stationary and opt["total_accepted_steps"] == 10, "expected the original ten-update stop")

    cg, rg, sg = captured(new / "gradient_final_retry_v1")
    require(cg["theta"] == theta and all(cg[k] == c_final[k] for k in MODEL_KEYS),
            "gradient check is not on the final design and model")
    ga = rg["gradient_audit"]
    require(math.isclose(ga["value"], opt["final_value"], rel_tol=1e-11, abs_tol=1e-11),
            "final objective did not reproduce")
    require(all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
                for a, b in zip(ga["gradient"], opt["final_gradient"], strict=True)),
            "final gradient did not reproduce")
    gradient_passed = bool(ga["passed"] and len(ga["finite_difference_checks"]) == 4
        and min(ga["rates"][-2:]) > 1.8
        and max(t["relative_difference"] for t in ga["finite_difference_checks"]) < .005)
    c0, r0, _ = captured(old / "coarse_initial_v1")
    pairs = [("Design mesh", c0, r0["evaluations"][0], cg, rg["evaluations"][0])]
    safety_records = [dict(name="gradient_final_retry_v1", **sg)]
    allowed = {"medium": {"grid"}, "half_dt": {"dt", "n_steps", "window_start_step"},
               "late_window": {"n_steps", "window_start_step"}}
    for kind, label in (("medium", "Refined mesh"), ("half_dt", "Half time step"),
                        ("late_window", "Later window")):
        ci, ri, si = captured(old / f"{kind}_initial_v1")
        cf, rf, sf = captured(new / f"{kind}_final_v1")
        require({k for k in MODEL_KEYS if cf[k] != c_final[k]} == allowed[kind],
                f"unexpected model change in {kind}")
        for key in ("jax", "jaxlib", "precision", "backend", "device_kind"):
            require(ri["provenance"][key] == rf["provenance"][key], f"baseline environment changed: {key}")
        pairs.append((label, ci, ri["evaluations"][0], cf, rf["evaluations"][0]))
        safety_records.append(dict(name=f"{kind}_final_v1", **sf))
    comparisons = []
    for label, ci, mi, cf, mf in pairs:
        check_pair(ci, cf, theta)
        require(mi["theta"] == ci["theta"] and mf["theta"] == cf["theta"], "metrics geometry mismatch")
        require(mi["reference_areas"] == mf["reference_areas"], "changed reference body area")
        require(all(math.isclose(area, ref, rel_tol=1e-12, abs_tol=1e-14)
                    for m in (mi, mf) for area, ref in zip(m["geometric_areas"], m["reference_areas"], strict=True)),
                "native body area not preserved")
        comparisons.append(dict(label=label, grid=ci["grid"], dt=ci["dt"],
            window=[ci["window_start_step"]*ci["dt"], ci["n_steps"]*ci["dt"]],
            initial_objective=mi["objective"], candidate_objective=mf["objective"],
            reduction_percent=100*(1-mf["objective"]/mi["objective"]),
            initial_areas=mi["geometric_areas"], candidate_areas=mf["geometric_areas"],
            initial_diffuse_area=mi["diffuse_area"], candidate_diffuse_area=mf["diffuse_area"]))
    criteria = criteria_for(comparisons, gradient_passed, stationary)
    history = []
    for result in (r_old, r_final):
        o = result["optimization"]
        states = o["trace"] + [dict(coefficients=o["coefficients"], objective=o["final_value"],
                                    projected_gradient_norm=o["final_projected_gradient_norm"])]
        for state in states:
            if history and state["coefficients"] == history[-1]["theta"]:
                require(math.isclose(state["objective"], history[-1]["objective"], rel_tol=1e-12),
                        "continuation seam changed objective")
                continue
            history.append(dict(update=len(history), theta=state["coefficients"], objective=state["objective"],
                projected_gradient_ratio=state["projected_gradient_norm"]/original_ref))
    require(len(history) == 11 and history[-1]["theta"] == theta, "incomplete ten-update history")
    raw_audit = read(audit_path or new / "raw_audit_v1.json")
    require(raw_audit["passed"] and all(r["passed"] for r in raw_audit["records"]), "raw audits failed")
    required = {str((new / name).resolve()) for name in
                ("gradient_final_retry_v1", "medium_final_v1", "half_dt_final_v1", "late_window_final_v1")}
    required |= {str((old / name).resolve()) for name in
                 ("coarse_initial_v1", "medium_initial_v1", "half_dt_initial_v1", "late_window_initial_v1")}
    require(required <= {str(Path(r["directory"]).resolve()) for r in raw_audit["records"]},
            "missing new-design or reused-baseline raw audit")
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(schema="multibody-final-rechecks-1", paper_eligible=False, passed=all(criteria.values()),
        scope="fixed final ten-update geometry; finite-Brinkman finite-time model; no sharp-wall or gradient-grid claim",
        accepted_updates=10, final_theta=theta, final_projected_gradient_norm=pg,
        original_projected_gradient_norm=original_ref, final_projected_gradient_ratio=pg/original_ref,
        gradient_audit=ga, comparisons=comparisons, criteria=criteria, optimization_history=history,
        hardware=safety_records, source_sha256=hashes,
        retained_interruption="gradient_final_v1: supervisor SIGTERM; incomplete, not counted as passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--new-runs", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Fresh raw-audit report after relocating an archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    csv_path = args.output.with_suffix(".csv")
    require(not args.output.exists() and not csv_path.exists(), "new summary paths required")
    report = build(args.repo.resolve(), args.new_runs.resolve(), args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "nx", "ny", "dt", "window_start", "window_end", "J0", "J_final", "gain_percent"])
        for row in report["comparisons"]:
            writer.writerow([row["label"], row["grid"]["nx"], row["grid"]["ny"], row["dt"],
                             *row["window"], row["initial_objective"], row["candidate_objective"], row["reduction_percent"]])
    print(json.dumps({"passed": report["passed"], "criteria": report["criteria"],
                      "comparisons": report["comparisons"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
