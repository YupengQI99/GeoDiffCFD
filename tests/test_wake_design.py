"""Tiny CPU/CUDA audits; no large unsteady CFD is launched by these tests."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_cad.differentiable.mac_channel import ChannelParams, create_channel_grid, parabolic_inlet, step
from jax_cad.differentiable.wake_design import (
    shape_fields, initial_flow, make_wake_kernels, evaluate_trajectory,
    objective_weights, validate_history,
)
from jax_cad.unified.geometry import fourier_shape_area
from jax_cad.verification import brinkman_force

jax.config.update("jax_enable_x64", True)


def problem():
    grid = create_channel_grid(12, 8, 1.2, .8)
    params = ChannelParams(.03, .002, include_advection=True, advection_scheme="central",
                           pressure_tolerance=1e-13, pressure_atol=1e-15,
                           pressure_maxiter=1000, pressure_preconditioner="constant-poisson-dct")
    inlet = parabolic_inlet(grid, .2, dtype=jnp.float64)
    material = lambda theta: shape_fields(theta, grid, center=(.4,.39), radius=.14,
                                         alpha_max=30., epsilon=.12).momentum_resistance
    return grid, params, inlet, material


def test_exact_area_and_material_contract():
    grid, _, _, material = problem()
    for theta in ([0.,0.], [.18,-.05], [-.12,.07]):
        coefficients = jnp.array([*theta,0.,0.])
        area = fourier_shape_area(coefficients, base_radius=.14, modes=(2,4), n_quadrature=4096)
        np.testing.assert_allclose(area, np.pi*.14**2, rtol=1e-13)
        q = np.asarray(material(jnp.asarray(theta)))
        assert q.shape == (grid.nx, grid.ny) and np.min(q)>=0 and np.max(q)<=30


def test_window_and_numerical_gates():
    np.testing.assert_array_equal(objective_weights(5,3), [0,0,0,.5,.5])
    with pytest.raises(ValueError): objective_weights(5,5)
    for column in range(2,7):
        h = np.zeros((2,7)); h[1,column] = 10
        with pytest.raises(RuntimeError): validate_history(h)
    with pytest.raises(RuntimeError): validate_history(np.full((1,7),np.nan))


def test_transient_material_adjoint_matches_serial_ad_and_taylor():
    grid, params, inlet, material = problem()
    state0 = initial_flow(grid,inlet)
    kernels = make_wake_kernels(grid,params,chunk_steps=4,force_scale=1.)
    nsteps, start = 11, 3  # tests a partial last chunk and startup differentiation
    theta = jnp.array([.04,-.02])
    alpha, pullback = jax.vjp(material,theta)
    result = evaluate_trajectory(alpha,inlet,state0,kernels,n_steps=nsteps,chunk_steps=4,
                                 window_start_step=start,gradient=True)
    gradient = pullback(result["material_gradient"])[0]
    # Separate serial execution, not the checkpointed program or lax.scan.
    one = jax.jit(lambda s,a,i: step(s,a,i,grid,params)[0])
    def reference(t):
        a=material(t); s=state0; value=0.
        for k in range(nsteps):
            s=one(s,a,inlet)
            if k>=start: value=value+brinkman_force(s,a,grid)[0]/(nsteps-start)
        return value
    ref, ref_gradient=jax.value_and_grad(reference)(theta)
    np.testing.assert_allclose(result["value"],ref,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(gradient,ref_gradient,rtol=1e-8,atol=1e-11)
    direction=jnp.array([.8,-.6])
    def value(t):
        return evaluate_trajectory(material(t),inlet,state0,kernels,n_steps=nsteps,chunk_steps=4,
                                    window_start_step=start)["value"]
    epsilons=np.array([.008,.004,.002,.001])
    errors=[abs(value(theta+e*direction)-result["value"]-e*float(gradient@direction)) for e in epsilons]
    rates=np.log2(np.array(errors[:-1])/errors[1:])
    fd=(value(theta+1e-5*direction)-value(theta-1e-5*direction))/2e-5
    np.testing.assert_allclose(gradient@direction,fd,rtol=2e-6,atol=2e-10)
    assert np.min(rates[-2:])>1.9
    print(json.dumps(dict(check="wake_full_transient_discrete_adjoint",backend=jax.default_backend(),
                         value=result["value"],gradient=np.asarray(gradient).tolist(),
                         taylor_rates=rates.tolist(),finite_difference=fd,
                         max_diagnostics=result["history"].max(axis=0).tolist(),
                         paper_eligible=False)),flush=True)
