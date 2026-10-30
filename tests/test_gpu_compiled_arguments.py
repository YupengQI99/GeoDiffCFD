"""Accelerator regression test: compiled forward solves must pass arrays as arguments.

On the RTX 4090 with JAX 0.7.0, a compiled program that embedded the 880 x 164
float64 operands of the DFG cylinder case as constants (a nullary ``jax.jit``)
returned a wrong projection step, and the pseudo-time iteration converged to a
state with steady residual 0.207 (archived runs cff6489 and 9e71539,
development-joint-q1-fine).  The same step with the arrays passed as arguments
agrees with the eager and CPU results to round-off.  These tests run only when
an accelerator is present; on CPU the two compilation modes agree and the
tests are skipped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    create_channel_grid,
    create_initial_state,
    parabolic_inlet,
    solve_steady,
    step,
)
from jax_cad.run_artifacts import compile_and_sample_args

ACCELERATOR = any(d.platform != "cpu" for d in jax.devices())


def _fine_cylinder_like():
    grid = create_channel_grid(880, 164, 2.2, 0.41)
    inlet = parabolic_inlet(grid, 0.2, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    x = (jnp.arange(grid.nx, dtype=jnp.float64)[:, None] + 0.5) * grid.dx
    y = (jnp.arange(grid.ny, dtype=jnp.float64)[None, :] + 0.5) * grid.dy
    phi = jnp.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2) - 0.05
    alpha = 160.0 * 0.5 * (1.0 - jnp.tanh(phi / 0.004))
    params = ChannelParams(
        viscosity=0.001,
        dt=0.000625,
        pressure_tolerance=1e-10,
        pressure_atol=1e-12,
        pressure_maxiter=4000,
        include_advection=True,
    )
    return grid, inlet, state0, alpha, params


@pytest.mark.skipif(not ACCELERATOR, reason="accelerator-only regression test")
def test_compiled_solve_with_array_arguments_matches_eager_step() -> None:
    grid, inlet, state0, alpha, params = _fine_cylinder_like()
    eager, _ = step(state0, alpha, inlet, grid, params)
    (compiled, _), _, _ = compile_and_sample_args(
        lambda s0, a, i, steps: solve_steady(s0, a, i, grid, params, steps),
        (state0, alpha, inlet, 1),
        1,
        static_argnums=(3,),
    )
    np.testing.assert_allclose(np.asarray(compiled.u), np.asarray(eager.u), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(compiled.v), np.asarray(eager.v), rtol=0.0, atol=1e-10)


@pytest.mark.skipif(not ACCELERATOR, reason="accelerator-only regression test")
@pytest.mark.xfail(strict=False, reason="documents the CUDA constant-embedding miscompilation of JAX 0.7.0")
def test_compiled_solve_with_embedded_constants_matches_eager_step() -> None:
    grid, inlet, state0, alpha, params = _fine_cylinder_like()
    eager, _ = step(state0, alpha, inlet, grid, params)
    compiled, _ = jax.jit(lambda: step(state0, alpha, inlet, grid, params)).lower().compile()()
    np.testing.assert_allclose(np.asarray(compiled.u), np.asarray(eager.u), rtol=0.0, atol=1e-10)
