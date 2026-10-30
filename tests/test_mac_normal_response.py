"""Scalable block elimination must reproduce the verified dense joint solve."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_cad.differentiable.mac_channel import (
    ChannelParams, create_channel_grid, create_initial_state, divergence,
    parabolic_inlet,
)
from jax_cad.differentiable.mac_normal_constraint import (
    NormalMarkers, dense_mac_layout, ellipse_markers, normal_interpolation,
    project_velocity_normal, spread_normal_reaction, require_valid, step_normal,
)
from jax_cad.differentiable.mac_normal_response import (
    normal_stencil, normal_sample, spread_impulse, build_normal_response,
    project_with_normal_response, step_with_normal_response,
    normal_control_volume_balance,
    solve_normal_chunk,
)

jax.config.update("jax_enable_x64", True)


def setup(ly=1.2):
    grid = create_channel_grid(16, 12, 1.6, ly)
    markers = ellipse_markers(jnp.array([.78, .5*ly]), jnp.array([.21, .14]), .17, 12)
    params = ChannelParams(.01, .003, pressure_tolerance=1.e-13, pressure_atol=1.e-15, pressure_maxiter=2000)
    u = .1*jnp.sin(jnp.arange(204).reshape((17, 12))*.13)
    v = .08*jnp.cos(jnp.arange(208).reshape((16, 13))*.11)
    mu, mv = .7+.1*jnp.cos(u), .6+.1*jnp.sin(v)
    return grid, markers, params, u, v, mu, mv


def test_compact_interpolation_and_adjoint_equal_dense():
    g, markers, _, u, v, *_ = setup()
    layout = dense_mac_layout(g)
    b, valid = normal_interpolation(markers, layout, g)
    stencil = normal_stencil(markers, g)
    assert valid and stencil.valid
    np.testing.assert_allclose(normal_sample(u, v, stencil), b @ jnp.r_[u.ravel(), v.ravel()], atol=1.e-15)
    amplitude = jnp.cos(jnp.arange(12))
    su, sv = spread_impulse(amplitude, stencil, g)
    ru, rv = spread_normal_reaction(amplitude*g.dx*g.dy/markers.weights, markers, layout, g)
    np.testing.assert_allclose(su, ru, atol=2.e-15)
    np.testing.assert_allclose(sv, rv, atol=2.e-15)


@pytest.mark.parametrize("ly", [1.2, .96])
def test_response_projection_matches_dense_including_nonzero_inlet(ly):
    g, markers, params, u, v, mu, mv = setup(ly)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    layout = dense_mac_layout(g)
    dense = require_valid(project_velocity_normal(u, v, mu, mv, inlet, g, params, layout, markers))
    response = build_normal_response(markers, mu, mv, g, params)
    candidate = require_valid(project_with_normal_response(u, v, inlet, g, params, response))
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(getattr(candidate.state, name), getattr(dense.state, name), rtol=1.e-10, atol=2.e-9)
    np.testing.assert_allclose(candidate.normal_traction, dense.normal_traction, rtol=1.e-10, atol=2.e-9)
    assert np.max(np.abs(divergence(candidate.state.u, candidate.state.v, g))) < 2.e-11
    assert np.max(np.abs(candidate.normal_velocity)) < 2.e-13
    assert candidate.stationarity_max < 2.e-13
    print(json.dumps(dict(check="response_vs_dense", ly=ly, pressure_max_error=float(jnp.max(jnp.abs(candidate.state.p-dense.state.p))),
          traction_max_error=float(jnp.max(jnp.abs(candidate.normal_traction-dense.normal_traction))),
          minimum_relative_eigenvalue=float(response.relative_minimum_eigenvalue), basis_divergence=float(response.basis_divergence_max),
          asymmetry=float(response.relative_asymmetry))), flush=True)


def test_response_nonlinear_steps_match_dense():
    g, markers, params, *_ = setup()
    params = params._replace(include_advection=True, advection_scheme="central")
    alpha = jnp.full((g.nx, g.ny), 3., dtype=jnp.float64)
    mu, mv = jnp.full((g.nx+1, g.ny), 1/(1+3*params.dt)), jnp.full((g.nx, g.ny+1), 1/(1+3*params.dt))
    response = build_normal_response(markers, mu, mv, g, params)
    layout = dense_mac_layout(g)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    a = b = create_initial_state(g, inlet, dtype=jnp.float64)
    for _ in range(3):
        dense = require_valid(step_normal(a, alpha, inlet, g, params, layout, markers))
        candidate = require_valid(step_with_normal_response(b, inlet, g, params, response))
        np.testing.assert_allclose(candidate.state.u, dense.state.u, atol=2.e-12)
        np.testing.assert_allclose(candidate.state.p, dense.state.p, atol=2.e-9)
        np.testing.assert_allclose(candidate.normal_traction, dense.normal_traction, atol=2.e-9)
        from jax_cad.verification import MACStaggeredControlVolume
        cv = MACStaggeredControlVolume(2, 13, 2, 9, 2, 13, 2, 9)
        balance = normal_control_volume_balance(candidate, alpha, inlet, g, params, cv)
        assert np.max(np.abs(balance["algebraic_closure"])) < 1.e-12
        assert np.linalg.norm(balance["captured_constraint"]) > 1.e-4
        a, b = dense.state, candidate.state


def test_redundant_response_is_rejected_without_regularisation():
    g, markers, params, u, v, mu, mv = setup()
    markers = NormalMarkers(*(jnp.concatenate((x, x[:1])) for x in markers))
    response = build_normal_response(markers, mu, mv, g, params)
    assert not response.valid
    with pytest.raises(ValueError, match="rank-deficient"):
        require_valid(project_with_normal_response(u, v, jnp.zeros(g.ny), g, params, response))


@pytest.mark.parametrize("ly", [1.2, .96])
def test_sequential_response_columns_reproduce_batched_equations(ly):
    g, markers, params, u, v, mu, mv = setup(ly)
    batched = build_normal_response(markers, mu, mv, g, params)
    sequential = jax.jit(lambda m, a, b: build_normal_response(m, a, b, g, params, column_batching="map"))(markers, mu, mv)
    assert batched.valid and sequential.valid
    for name in ("velocity_u", "velocity_v", "pressure_impulse", "schur"):
        np.testing.assert_allclose(getattr(sequential, name), getattr(batched, name), rtol=1e-10, atol=2e-11)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    a = require_valid(project_with_normal_response(u, v, inlet, g, params, batched))
    b = require_valid(project_with_normal_response(u, v, inlet, g, params, sequential))
    np.testing.assert_allclose(a.state.u, b.state.u, rtol=1e-10, atol=2e-11)
    np.testing.assert_allclose(a.state.p, b.state.p, rtol=1e-10, atol=2e-9)
    assert np.max(np.abs(divergence(b.state.u, b.state.v, g))) < 2e-11
    assert np.max(np.abs(b.normal_velocity)) < 2e-13


def test_large_grid_has_no_dense_pressure_matrix():
    g = create_channel_grid(44, 16, 2.2, .8)
    params = ChannelParams(.01, .003, pressure_tolerance=1.e-12, pressure_atol=1.e-14, pressure_maxiter=3000)
    markers = ellipse_markers(jnp.array([.7, .4]), jnp.array([.13, .13]), 0., 16)
    mu, mv = jnp.ones((45, 16)), jnp.ones((44, 17))
    response = build_normal_response(markers, mu, mv, g, params)
    assert response.valid
    assert response.schur.shape == (16, 16)
    assert response.velocity_u.shape == (16, 45, 16)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    initial = create_initial_state(g, inlet, dtype=jnp.float64)
    result = require_valid(project_with_normal_response(initial.u, initial.v, inlet, g, params, response))
    assert np.max(np.abs(divergence(result.state.u, result.state.v, g))) < 2.e-10
    assert np.max(np.abs(result.normal_velocity)) < 2.e-12


def test_reverse_geometry_derivative_includes_response_rebuild():
    g, _, params, u, v, mu, mv = setup()
    inlet = jnp.zeros(g.ny)
    def objective(theta):
        markers = ellipse_markers(theta[:2], theta[2:4], theta[4], 12)
        response = build_normal_response(markers, mu, mv, g, params)
        result = project_with_normal_response(u, v, inlet, g, params, response)
        return .01*jnp.sum(result.state.u**2)+.001*jnp.sum(markers.weights*result.normal_traction**2)
    f = jax.jit(jax.value_and_grad(objective))
    theta = jnp.array([.78, .60, .21, .14, .17])
    value, gradient = f(theta)
    differences = []
    for i in range(5):
        delta = jnp.eye(5)[i]*2.e-6
        finite_difference = (f(theta+delta)[0]-f(theta-delta)[0])/4.e-6
        np.testing.assert_allclose(gradient[i], finite_difference, rtol=2.e-5, atol=2.e-8)
        differences.append(float(abs(gradient[i]-finite_difference)/max(abs(finite_difference), 1.e-12)))
    print(json.dumps(dict(check="response_geometry_gradient", objective=float(value), gradient=np.asarray(gradient).tolist(), finite_difference_relative_errors=differences)), flush=True)


def test_chunked_response_matches_separate_steps():
    g, markers, params, *_ = setup()
    alpha = jnp.full((g.nx, g.ny), 3., dtype=jnp.float64)
    mu, mv = jnp.full((g.nx+1, g.ny), 1/(1+3*params.dt)), jnp.full((g.nx, g.ny+1), 1/(1+3*params.dt))
    response = build_normal_response(markers, mu, mv, g, params)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    initial_state = create_initial_state(g, inlet, dtype=jnp.float64)
    initial = require_valid(project_with_normal_response(initial_state.u, initial_state.v, inlet, g, params, response))
    expected = initial
    for _ in range(3):
        expected = require_valid(step_with_normal_response(expected.state, inlet, g, params, response))
    kernel = jax.jit(lambda start, a, i, r: solve_normal_chunk(start, a, i, g, params, r, 3, 0., 0))
    actual, history, count, metrics = kernel(initial, alpha, inlet, response)
    require_valid(actual)
    assert int(count) == 3 and np.isfinite(history).all()
    np.testing.assert_allclose(actual.state.u, expected.state.u, atol=3.e-12)
    np.testing.assert_allclose(actual.state.p, expected.state.p, atol=3.e-9)
    assert metrics[3] < 1.e-10 and metrics[4] < 1.e-12
