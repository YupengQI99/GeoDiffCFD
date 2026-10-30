"""Independent algebraic signs/rank checks before the bounded FE CPU runs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("fe_normal_constraint", Path(__file__).resolve().parents[1]/"tools/fe_normal_constraint.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_augmented_newton_schur_step_matches_full_saddle_solve(nonsymmetric):
    random = np.random.default_rng(422)
    Q = random.normal(size=(30, 30))
    A = Q.T @ Q+np.eye(30)
    if nonsymmetric:
        A += .2*random.normal(size=A.shape)
    B = random.normal(size=(6, 30))
    momentum = random.normal(size=30)
    constraint = random.normal(size=6)
    Y = np.linalg.solve(A, B.T)
    inverse_residual = np.linalg.solve(A, momentum)
    dlambda, ratio = helper.multiplier_correction(B @ Y, constraint, B @ inverse_residual)
    dz = -inverse_residual-Y @ dlambda
    exact = np.linalg.solve(np.block([[A, B.T], [B, np.zeros((6, 6))]]), -np.r_[momentum, constraint])
    np.testing.assert_allclose(np.r_[dz, dlambda], exact, rtol=1.e-12, atol=1.e-12)
    assert ratio > 1.e-10
    np.testing.assert_allclose(A @ dz+B.T @ dlambda, -momentum, atol=1.e-12)
    np.testing.assert_allclose(B @ dz, -constraint, atol=1.e-12)


def test_duplicate_fe_constraints_do_not_get_silent_regularisation():
    with pytest.raises(RuntimeError, match="rank-deficient"):
        helper.multiplier_correction(np.ones((2, 2)), np.ones(2), np.zeros(2))
