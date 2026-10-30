import json

import numpy as np
import pytest

from jax_cad.brinkman_mask_calibration import solve_planar_brinkman_shear
from jax_cad.continuum_mask_calibration import continuum_planar_displacement


def test_continuum_calibration_agrees_across_integrators_and_with_refined_fd():
    results = [continuum_planar_displacement(.5, 1.5, method=method, rtol=tol, atol=tol/100)
               for method, tol in (("DOP853", 1e-9), ("DOP853", 1e-11), ("RK45", 1e-11))]
    roots = [r["zero_displacement_shift_over_reference_h"] for r in results]
    assert max(roots)-min(roots) < 1e-9
    errors = []
    for n in (16, 32, 64):
        solved = solve_planar_brinkman_shear(alpha_max=4, viscosity=1, grid_spacing=1/n,
            epsilon_over_h=1.5*n, solid_cells=16*n, fluid_cells=16*n, far_field_fit_cells=4*n)
        errors.append(abs(solved.displacement_length-roots[1]))
    assert np.all(np.asarray(errors[1:]) < .26*np.asarray(errors[:-1]))
    assert errors[-1] < 1e-5
    print(json.dumps(dict(check="grid_free_planar_calibration", integrations=results,
                          refined_fd_errors=errors, reference_data="none")), flush=True)


@pytest.mark.parametrize("q", [.25, .5, 1., 2.])
def test_sharp_halfspace_has_exact_displacement_q(q):
    assert continuum_planar_displacement(q, 0)["zero_displacement_shift_over_reference_h"] == q


@pytest.mark.parametrize("q,epsilon", [(0, 1), (.5, -1), (float("nan"), 1), (5, 1)])
def test_rejects_invalid_or_unbounded_inputs(q, epsilon):
    with pytest.raises(ValueError):
        continuum_planar_displacement(q, epsilon)
