from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import (
    steady_residual,
    steady_residual_norms,
    steady_stokes_brinkman,
)
from jax_cad.unified import DifferentiableProgram, FourierObstacleBenchmark


def test_benchmark_configuration_is_self_consistent():
    bench = FourierObstacleBenchmark(nx=16, ny=10)
    record = bench.to_dict()
    assert record["h"] == min(record["dx"], record["dy"])
    assert abs(record["epsilon"] - bench.epsilon_over_h * record["h"]) < 1e-15
    diffusion = bench.viscosity * bench.dt * (1.0 / bench.grid.dx**2 + 1.0 / bench.grid.dy**2)
    assert abs(diffusion - bench.diffusion_number) < 1e-12
    assert record["n_geometry_parameters"] == 2 * len(bench.modes)


def test_krylov_fixed_point_matches_direct_saddle_solve_and_is_differentiable():
    bench = FourierObstacleBenchmark(nx=16, ny=10)
    design_map = bench.design_map()
    solver, primal = bench.implicit_solver()
    alpha = design_map(bench.zero_coefficients()).momentum_resistance

    state, diagnostics = primal(alpha)
    assert bool(diagnostics.converged)
    rms, _ = steady_residual_norms(steady_residual(state, alpha, bench.grid, bench.params()))
    assert float(rms) < 1e-9

    direct = steady_stokes_brinkman(alpha, bench.grid, bench.params(), tol=1e-13, maxiter=2000)
    np.testing.assert_allclose(np.asarray(direct.u), np.asarray(state.u), atol=1e-9)

    program = DifferentiableProgram(design_map, solver, bench.objective)
    value, gradient = jax.jit(program.value_and_grad)(bench.zero_coefficients())
    assert np.isfinite(float(value))
    assert gradient.shape == (bench.n_design,)
    assert np.all(np.isfinite(np.asarray(gradient)))
    # Elongating the obstacle along the flow (positive a_2) increases the mean flow.
    assert float(gradient[0]) < 0.0
