"""Final-design summaries must reject stale candidates and changed paired models."""
import copy
import math

import pytest

from tools.summarize_multibody_final import MODEL_KEYS, check_pair, criteria_for, pg_norm


def paired():
    initial = dict.fromkeys(MODEL_KEYS, 1)
    initial["theta"] = [0.] * 12
    final = copy.deepcopy(initial)
    theta = [1.] * 12
    final["theta"] = theta
    return initial, final, theta


def comparisons():
    return [dict(initial_objective=j0, candidate_objective=jf,
                 reduction_percent=100*(1-jf/j0))
            for j0, jf in [(16., 11.), (16.2, 11.2), (16.01, 11.01), (15.99, 10.99)]]


def test_identical_final_design_and_model_pass():
    check_pair(*paired())


def test_old_candidate_rejected():
    initial, final, theta = paired()
    final["theta"] = [.8] * 12
    with pytest.raises(ValueError, match="final design"):
        check_pair(initial, final, theta)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_changed_paired_model_rejected(key):
    initial, final, theta = paired()
    final[key] = 2
    with pytest.raises(ValueError, match="paired models"):
        check_pair(initial, final, theta)


def test_wrong_initial_geometry_rejected():
    initial, final, theta = paired()
    initial["theta"] = [.1] * 12
    with pytest.raises(ValueError, match="original"):
        check_pair(initial, final, theta)


def test_projected_gradient_respects_active_bounds():
    assert pg_norm([1., -1., 0.], [-2., 2., .3]) == pytest.approx(.3)
    assert pg_norm([1., -1.], [2., -2.]) == pytest.approx(math.sqrt(8))


def test_original_criteria_pass():
    assert all(criteria_for(comparisons(), True, True).values())


def test_time_sensitivity_not_hidden_by_positive_improvement():
    rows = comparisons()
    rows[2]["candidate_objective"] = 12.
    assert not criteria_for(rows, True, True)["time_and_window_changes_below_tenth_of_gain"]


def test_nonpositive_refined_gain_fails():
    rows = comparisons()
    rows[1]["reduction_percent"] = -1.
    assert not criteria_for(rows, True, True)["positive_gain_all_comparisons"]


@pytest.mark.parametrize("gradient,stationary", [(False, True), (True, False), (False, False)])
def test_no_pass_without_gradient_and_original_stop(gradient, stationary):
    assert not all(criteria_for(comparisons(), gradient, stationary).values())
