from __future__ import annotations

import pytest

from tools.summarize_forward_verification import (
    ValidationProblem,
    _convergence_rows,
    _flatten_metrics,
)


def _row(h: float, objective: float, name: str, run_id: str) -> dict:
    return {
        "case": "dfg-cylinder-2d1-re20",
        "config_schema_version": "test-v1",
        "study": "grid-refinement",
        "objective_name": name,
        "objective": objective,
        "h": h,
        "run_id": run_id,
        "nx": round(2.2 / h),
        "ny": round(0.41 / h),
    }


def test_convergence_series_do_not_mix_measurement_objectives() -> None:
    rows = [
        _row(0.04, 0.16, "brinkman-error", "old-coarse"),
        _row(0.02, 0.08, "brinkman-error", "old-fine"),
        _row(0.04, 0.09, "surface-error", "new-coarse"),
        _row(0.02, 0.03, "surface-error", "new-fine"),
    ]
    convergence = _convergence_rows(rows)
    assert len(convergence) == 4
    by_name = {}
    for row in convergence:
        by_name.setdefault(row["objective_name"], []).append(row)
    assert set(by_name) == {"brinkman-error", "surface-error"}
    assert by_name["brinkman-error"][1]["observed_order_from_previous"] == 1.0
    assert by_name["surface-error"][1]["observed_order_from_previous"] == pytest.approx(
        1.5849625007211563
    )


def test_duplicate_spacing_within_one_objective_is_rejected() -> None:
    rows = [
        _row(0.04, 0.16, "surface-error", "coarse-a"),
        _row(0.04, 0.12, "surface-error", "coarse-b"),
    ]
    with pytest.raises(ValidationProblem, match="duplicate grid spacing"):
        _convergence_rows(rows)


@pytest.mark.parametrize("scheme", [None, "central"])
def test_cylinder_summary_preserves_scheme_penalty_and_primary_force(scheme):
    metrics = {
        "immersed_boundary": {"epsilon_over_h": 1.5, "alpha_max": 40,
                              "delta_b_over_h": 1, "mask_midpoint_shift_over_h": 0.78},
        "numerical": {"brinkman_drag_coefficient": 6.0, "brinkman_lift_coefficient": 0.004,
                      "surface_drag_coefficient": 4.5, "surface_lift_coefficient": 0.001,
                      "pressure_difference": 0.11, "recirculation_length_from_cylinder_rear": 0.08},
        "relative_errors": {"brinkman_drag_coefficient": 0.08, "pressure_difference": 0.05},
    }
    if scheme is not None:
        metrics["advection_scheme"] = scheme
    flat = _flatten_metrics({"case": "dfg-cylinder-2d1-re20",
                             "experiment_id": "jax_diagnostic", "metrics": metrics})
    assert flat["advection_scheme"] == (scheme or "donor-cell")
    assert flat["delta_b_over_h"] == 1
    assert flat["mask_midpoint_shift_over_h"] == 0.78
    assert flat["brinkman_drag_coefficient"] == 6.0
    assert flat["surface_drag_coefficient"] == 4.5


def test_finite_fe_summary_keeps_direct_and_extrapolated_pressure_distinct():
    metrics = dict(metrics=dict(drag_coefficient=6., lift_coefficient=.001,
                                pressure_difference=.10, pressure_difference_radial_extrapolated=.112),
                   formulation=dict(model=dict(alpha=40, epsilon=.0075, shift=.0039), quadrature_degree=10),
                   mesh=dict(cells=100, mixed_dofs=500, diagonal="right", local_refinements=2))
    flat = _flatten_metrics(dict(case="dfg-cylinder-2d1-finite-brinkman", metrics=metrics))
    assert flat["pressure_difference_direct"] == .10
    assert flat["pressure_difference"] == .112
    assert flat["brinkman_drag_coefficient"] == 6.
    assert "surface_drag_coefficient" not in flat
    assert flat["local_refinements"] == 2
