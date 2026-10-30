"""Frozen observer of a sharp-wall solution; never changes the PDE or primary dp."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = "configs/forward_verification/dolfinx_dfg2d1_reference.json"
PARENT_HASH = "cdb4515047728d0c8f465274974376aceac2f66f9779b073d5b78c353119497d"
OBSERVER = dict(reference_h=[.01, .005, .0025],
                offsets_over_h=[4.5, 4.8, 5.1, 5.4, 5.7], degree=1,
                development_only=True, paper_eligible=False)
NATIVE_OBSERVER = dict(reference_h=[.01, .005, .0025],
                       offsets_over_h=[3.5, 4.5, 5.5, 6.5, 7.5], degree=2,
                       sampling="native-force-free-axis-columns",
                       development_only=True, paper_eligible=False)
NEAR_OBSERVER = dict(reference_h=[.01, .005, .0025],
                    offsets_over_h=[.5,1.5,2.5,3.5,4.5],degree=2,
                    sampling="native-reaction-corrected-near-columns",
                    development_only=True,paper_eligible=False)


def validate(config):
    if config.get("source_config") != PARENT or config.get("source_config_sha256") != PARENT_HASH:
        raise ValueError("sharp trace diagnostic must pin the canonical reference")
    parent_path = ROOT/PARENT
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != PARENT_HASH:
        raise ValueError("canonical reference configuration hash mismatch")
    parent = json.loads(parent_path.read_text())
    for key in ("schema_version", "case", "container", "geometry", "physics", "mesh",
                "finite_elements", "boundary_conditions", "force_measurement", "solver", "reference"):
        if config.get(key) != parent[key]:
            raise ValueError(f"sharp trace diagnostic must preserve canonical {key}")
    if config.get("study") != {**parent["study"], "development_only": True, "paper_eligible": False}:
        raise ValueError("sharp trace diagnostic cannot claim paper eligibility")
    if config.get("pressure_trace_diagnostic") not in (OBSERVER, NATIVE_OBSERVER, NEAR_OBSERVER):
        raise ValueError("pressure observer must match the preregistered estimator")
    return parent_path


def linear_trace(offsets, values):
    """Same degree-one least-squares intercept as the finite-mask measurement."""
    import numpy as np
    offsets, values = np.asarray(offsets, dtype=float), np.asarray(values, dtype=float)
    if offsets.ndim != 1 or len(offsets) < 2 or values.shape != offsets.shape:
        raise ValueError("paired one-dimensional samples required")
    if not np.isfinite(offsets).all() or not np.isfinite(values).all() or np.min(offsets) <= 0:
        raise ValueError("finite positive offsets and finite samples required")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("distinct increasing offsets required")
    # Normalisation only improves conditioning; intercept is unchanged.
    design = np.polynomial.polynomial.polyvander(offsets/offsets[-1], 1)
    weights = np.linalg.pinv(design)[0]
    intercept = float(np.polynomial.polynomial.polyfit(offsets, values, 1)[0])
    return dict(intercept=intercept, weights=weights.tolist(),
                weight_l1=float(np.abs(weights).sum()), condition=float(np.linalg.cond(design)))


def observe(config, probe, pressure_front, pressure_rear):
    """probe(xy) is the independent sharp FE field, not a fitted pressure model."""
    validate(config)
    settings = config["pressure_trace_diagnostic"]
    centre = config["geometry"]["cylinder_centre"]
    radius = config["geometry"]["cylinder_radius"]
    direct = pressure_front-pressure_rear
    if settings == NEAR_OBSERVER:
        import numpy as np
        import sys
        sys.path.insert(0,str(ROOT))
        from jax_cad.reaction_pressure_trace import observe_continuous
        records=[]
        for h in settings["reference_h"]:
            observed=observe_continuous(probe,points=np.empty((0,2)),normals=np.empty((0,2)),
                integrated_reaction=np.empty(0),h=h,centre=centre,radius=radius)
            records.append(dict(reference_h=h,**observed,
                bias_vs_same_solution_direct=observed["pressure_difference"]-direct,
                relative_bias_vs_same_solution_direct=(observed["pressure_difference"]-direct)/abs(direct)))
        return dict(protocol="native-near-trace-on-sharp-reference-v1",settings=settings,
            direct_pressure_difference=direct,traces=records,paper_eligible=False,
            modifies_pde=False,selects_best_fit=False)
    records = []
    for h in settings["reference_h"]:
        offsets = [v*h for v in settings["offsets_over_h"]]
        sides = []
        for sign in (-1, 1):
            points = [[centre[0]+sign*(radius+s), centre[1]] for s in offsets]
            values = [float(probe(point)) for point in points]
            if settings == NATIVE_OBSERVER:
                import sys
                sys.path.insert(0, str(ROOT))
                from jax_cad.native_pressure_trace import polynomial_intercept
                fit = polynomial_intercept(offsets, values, degree=2)
            else:
                fit = linear_trace(offsets, values)
            sides.append(dict(points=points, samples=values, **fit))
        trace = sides[0]["intercept"]-sides[1]["intercept"]
        records.append(dict(reference_h=h, offsets=offsets, front=sides[0], rear=sides[1],
                            pressure_difference_extrapolated=trace,
                            bias_vs_same_solution_direct=trace-direct,
                            relative_bias_vs_same_solution_direct=(trace-direct)/abs(direct)))
    protocol = ("native-quadratic-trace-on-sharp-reference-v1" if settings == NATIVE_OBSERVER
                else "frozen-linear-exterior-trace-on-sharp-reference-v1")
    return dict(protocol=protocol, settings=settings,
                direct_pressure_front=pressure_front, direct_pressure_rear=pressure_rear,
                direct_pressure_difference=direct, traces=records,
                paper_eligible=False, modifies_pde=False, selects_best_fit=False)
