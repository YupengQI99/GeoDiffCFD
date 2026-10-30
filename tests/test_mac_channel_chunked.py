"""Chunked steady solve of the non-periodic kernel matches the single-program solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_channel import (
    ChannelParams,
    create_channel_grid,
    create_initial_state,
    parabolic_inlet,
    solve_steady,
    solve_steady_chunked,
)


def _problem():
    grid = create_channel_grid(24, 10, 2.4, 1.0)
    inlet = parabolic_inlet(grid, 0.2, dtype=jnp.float64)
    state0 = create_initial_state(grid, inlet, dtype=jnp.float64)
    x = jnp.linspace(0.0, 1.0, grid.nx, dtype=jnp.float64)[:, None]
    y = jnp.linspace(0.0, 1.0, grid.ny, dtype=jnp.float64)[None, :]
    alpha = 40.0 * jnp.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / 0.02)
    params = ChannelParams(
        viscosity=0.05,
        dt=0.004,
        pressure_tolerance=1.0e-12,
        pressure_atol=1.0e-14,
        pressure_maxiter=2000,
        include_advection=True,
        nonlinear_rtol=5.0e-3,
        nonlinear_atol=0.0,
        nonlinear_min_steps=5,
    )
    return grid, inlet, state0, alpha, params


def test_chunked_solve_reproduces_single_program_solve() -> None:
    grid, inlet, state0, alpha, params = _problem()
    reference_state, reference = solve_steady(state0, alpha, inlet, grid, params, 1500)
    chunked_state, chunked, report = solve_steady_chunked(
        state0, alpha, inlet, grid, params, 1500, 137
    )
    assert bool(reference.converged) and bool(chunked.converged)
    assert int(reference.iterations) == int(chunked.iterations)
    for a, b in zip(reference_state, chunked_state, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-11, atol=1e-13)
    n = int(reference.iterations)
    np.testing.assert_allclose(
        np.asarray(chunked.nonlinear_residual_rms_history[:n]),
        np.asarray(reference.nonlinear_residual_rms_history[:n]),
        rtol=1e-9,
        atol=1e-14,
    )
    assert bool(np.all(np.asarray(chunked.active_history[:n])))
    assert not bool(np.any(np.asarray(chunked.active_history[n:])))
    assert report["chunk_steps"] == 137
    assert report["chunk_count"] == -(-n // 137)
    assert report["frozen_chunk_restarts"] == 0
    assert all(not chunk["frozen"] for chunk in report["chunks"])
    assert sum(chunk["iterations"] for chunk in report["chunks"]) == n
    np.testing.assert_allclose(report["threshold"], params.nonlinear_rtol * float(reference.initial_residual_rms))


def test_chunked_solve_reports_iteration_cap() -> None:
    grid, inlet, state0, alpha, params = _problem()
    tight = params._replace(nonlinear_rtol=1e-14)
    _, diagnostics, report = solve_steady_chunked(state0, alpha, inlet, grid, tight, 50, 20)
    assert not bool(diagnostics.converged)
    assert int(diagnostics.iterations) == 50
    assert [c["iterations"] for c in report["chunks"]] == [20, 20, 10]
    assert report["frozen_chunk_restarts"] == 0
