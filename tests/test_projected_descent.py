from __future__ import annotations

import numpy as np
import pytest

from jax_cad.optimization.projected_descent import (
    projected_gradient,
    projected_gradient_descent,
)


def _quadratic(center):
    def value(x):
        return float(0.5 * np.sum((x - center) ** 2))

    def value_and_grad(x):
        return value(x), x - center

    return value, value_and_grad


def test_projected_gradient_zeroes_outward_components_at_bounds():
    x = np.array([0.3, -0.3, 0.0])
    g = np.array([-1.0, 2.0, 0.5])
    projected = projected_gradient(x, g, -0.3, 0.3)
    np.testing.assert_allclose(projected, [0.0, 0.0, 0.5])
    # Components pointing inward are retained.
    projected = projected_gradient(x, -g, -0.3, 0.3)
    np.testing.assert_allclose(projected, [1.0, -2.0, -0.5])


def test_descent_converges_to_interior_minimizer():
    value, value_and_grad = _quadratic(np.array([0.1, -0.05, 0.2]))
    result = projected_gradient_descent(
        value_and_grad, value, np.zeros(3), lower=-0.3, upper=0.3,
        initial_step=0.05, max_iterations=200, relative_gradient_tolerance=1e-6,
    )
    assert result.converged
    assert result.termination == "projected_gradient_tolerance"
    np.testing.assert_allclose(result.coefficients, [0.1, -0.05, 0.2], atol=1e-6)
    objectives = [item["objective"] for item in result.trace]
    assert all(b <= a for a, b in zip(objectives, objectives[1:]))
    assert all(item["coefficient_change_norm"] <= item["step_length"] + 1e-14 for item in result.trace)


def test_descent_stops_at_active_bound_with_zero_projected_gradient():
    value, value_and_grad = _quadratic(np.array([0.9, 0.0]))
    result = projected_gradient_descent(
        value_and_grad, value, np.zeros(2), lower=-0.3, upper=0.3,
        initial_step=0.1, max_iterations=100, relative_gradient_tolerance=1e-8,
    )
    assert result.converged
    np.testing.assert_allclose(result.coefficients, [0.3, 0.0], atol=1e-8)
    assert result.final_projected_gradient_norm <= 1e-8 * result.initial_projected_gradient_norm
    first = result.trace[0]
    assert first["accepted"] and first["realised_change"] < 0.0
    # For a quadratic the realised change exceeds the linear prediction by the curvature term.
    assert first["realised_change"] > first["linear_prediction"]


def test_continuation_keeps_original_threshold_not_restart_norm():
    value, vg = _quadratic(np.zeros(1))
    result = projected_gradient_descent(vg, value, np.array([.03]), lower=-1., upper=1.,
        initial_step=.1, max_iterations=4, relative_gradient_tolerance=.02,
        reference_gradient_norm=2.)
    assert result.converged and not any(t["accepted"] for t in result.trace)
    assert result.initial_projected_gradient_norm == .03
    assert result.final_projected_gradient_norm <= .02 * 2.


def test_segmented_descent_matches_unsplit_trajectory():
    value, vg = _quadratic(np.zeros(2))
    initial = np.array([.9, -.7])
    reference = np.linalg.norm(initial)
    whole = projected_gradient_descent(vg, value, initial, lower=-1., upper=1.,
        initial_step=.45, max_iterations=80, relative_gradient_tolerance=.02)
    x = initial.copy(); step = .45; accepted = 0
    for _ in range(40):
        part = projected_gradient_descent(vg, value, x, lower=-1., upper=1.,
            initial_step=step, max_iterations=2, relative_gradient_tolerance=.02,
            reference_gradient_norm=reference)
        records = [r for r in part.trace if r["accepted"]]
        accepted += len(records)
        if records: step = records[-1]["step_length"]
        x = part.coefficients
        if part.converged: break
    assert part.converged and whole.converged
    np.testing.assert_array_equal(x, whole.coefficients)
    assert accepted == sum(t["accepted"] for t in whole.trace)


@pytest.mark.parametrize("reference", [0., -1., float("nan"), float("inf")])
def test_continuation_rejects_invalid_reference(reference):
    value, vg = _quadratic(np.zeros(1))
    with pytest.raises(ValueError, match="reference gradient"):
        projected_gradient_descent(vg, value, np.ones(1), lower=-1., upper=1.,
            initial_step=.1, max_iterations=1, reference_gradient_norm=reference)
