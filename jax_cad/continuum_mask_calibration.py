"""Grid-free planar tangential Brinkman displacement, never fitted to CFD data.

Development-only independent calibration. This does not replace the existing
single-phase MAC calibration. The continuous half-space equation is
u'' = alpha(s) u, alpha_max=1/q**2, with dimensionless reference h=nu=1.
Integrate r=u'/u via r'=alpha-r**2; in the constant solid r=1/q exactly.
Beyond the compact transition u=C(s+L), hence L=1/r(epsilon)-epsilon.
Translation invariance makes the zero-displacement mask shift exactly L.
"""
from __future__ import annotations

import math


def continuum_planar_displacement(q, epsilon, *, method="DOP853", rtol=1e-11, atol=1e-13):
    import numpy as np
    from scipy.integrate import solve_ivp

    if not (math.isfinite(q) and .125 <= q <= 4):
        raise ValueError("q must be finite in [.125,4] for this bounded diagnostic")
    if not (math.isfinite(epsilon) and 0 <= epsilon <= 4):
        raise ValueError("epsilon must be finite in [0,4]")
    if method not in ("DOP853", "RK45"):
        raise ValueError("audited explicit integrators only")
    if not (1e-13 <= rtol <= 1e-7 and 1e-15 <= atol <= 1e-9):
        raise ValueError("diagnostic integration tolerances outside audited range")
    if epsilon == 0:
        endpoint, displacement, evaluations = 1/q, q, 0
    else:
        def rhs(s, ratio):
            t = float(np.clip(s/epsilon, -1., 1.))
            alpha = .5*(1-t-math.sin(math.pi*t)/math.pi)/(q*q)
            return alpha-ratio*ratio
        result = solve_ivp(rhs, (-epsilon, epsilon), [1/q], method=method,
                           rtol=rtol, atol=atol, max_step=epsilon/8)
        if not result.success or not np.isfinite(result.y).all() or np.min(result.y) <= 0:
            raise RuntimeError("continuous planar logarithmic derivative failed")
        endpoint = float(result.y[0, -1])
        displacement = 1/endpoint-epsilon
        evaluations = result.nfev
    return dict(schema_version="continuum-planar-displacement-1.0", q=q,
                epsilon_over_reference_h=epsilon, method=method, rtol=rtol, atol=atol,
                endpoint_logarithmic_derivative=endpoint,
                unshifted_displacement_over_reference_h=displacement,
                zero_displacement_shift_over_reference_h=displacement,
                function_evaluations=evaluations, reference_data="none",
                definition="continuous compact-cosine half-space tangential shear; not MAC calibration",
                paper_eligible=False)
