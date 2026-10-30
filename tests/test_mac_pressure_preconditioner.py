import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_cad.differentiable.mac_channel import create_channel_grid, weighted_pressure_operator, _forward_conjugate_gradient
from jax_cad.differentiable.mac_pressure_preconditioner import constant_pressure_inverse, preconditioned_cg

jax.config.update("jax_enable_x64",True)


@pytest.mark.parametrize("nx,ny,lx,ly",[(12,8,1.2,.8),(11,9,1.4,.7),(44,16,2.2,.8)])
def test_dct_inverse_preserves_mixed_half_cell_boundary_operator(nx,ny,lx,ly):
    grid=create_channel_grid(nx,ny,lx,ly)
    mu,mv=jnp.ones((nx+1,ny)),jnp.ones((nx,ny+1))
    rhs=jnp.cos(jnp.arange(nx*ny).reshape((nx,ny))*.137)+.2
    inverse=jax.jit(lambda b:constant_pressure_inverse(b,grid))
    p=inverse(rhs)
    residual=weighted_pressure_operator(p,mu,mv,grid)-rhs
    assert float(jnp.linalg.norm(residual)/jnp.linalg.norm(rhs))<1e-11
    other=jnp.sin(jnp.arange(nx*ny).reshape((nx,ny))*.37)
    np.testing.assert_allclose(jnp.vdot(rhs,inverse(other)),jnp.vdot(inverse(rhs),other),atol=1e-12,rtol=1e-12)
    assert jnp.vdot(rhs,p)>0


def test_variable_mobility_solution_matches_unpreconditioned_equations():
    grid=create_channel_grid(32,16,1.6,.8)
    mu=.9+.07*jnp.sin(jnp.arange(33*16).reshape((33,16))*.17)
    mv=.9+.07*jnp.cos(jnp.arange(32*17).reshape((32,17))*.23)
    rhs=jnp.cos(jnp.arange(32*16).reshape((32,16))*.13)
    operator=lambda p:weighted_pressure_operator(p,mu,mv,grid)
    original=_forward_conjugate_gradient(operator,rhs,tolerance=1e-12,atol=1e-14,maxiter=4000)
    candidate,k,recursive,fresh,relative=jax.jit(lambda b:preconditioned_cg(operator,b,grid,tolerance=1e-12,atol=1e-14,maxiter=4000))(rhs)
    np.testing.assert_allclose(candidate,original,atol=1e-11,rtol=1e-10)
    assert int(k)<=20 and float(relative)<1e-11
    print(json.dumps(dict(check="preconditioned_same_pressure_equation",iterations=int(k),
                         recursive_residual=float(recursive),fresh_residual=float(fresh),
                         relative_residual=float(relative),max_solution_difference=float(jnp.max(jnp.abs(candidate-original))))),flush=True)


def test_zero_rhs_and_iteration_cap_are_not_hidden():
    grid=create_channel_grid(12,8,1.2,.8)
    mu,mv=jnp.ones((13,8)),jnp.ones((12,9))
    operator=lambda p:weighted_pressure_operator(p,mu,mv,grid)
    p,k,_,fresh,_=preconditioned_cg(operator,jnp.zeros((12,8)),grid,tolerance=1e-12,atol=1e-14,maxiter=4000)
    assert int(k)==0 and float(fresh)==0 and np.all(np.asarray(p)==0)
    _,k,_,fresh,_=preconditioned_cg(operator,jnp.ones((12,8)),grid,tolerance=1e-12,atol=1e-14,maxiter=0)
    assert int(k)==0 and float(fresh)>1


def test_pressure_vjp_matches_old_solve_and_central_difference():
    from jax_cad.differentiable.mac_channel import ChannelParams,solve_pressure
    grid=create_channel_grid(12,8,1.2,.8)
    rhs=jnp.cos(jnp.arange(96).reshape((12,8))*.17)
    def objective(theta, strategy):
        mu=.9+theta*.05*jnp.sin(jnp.arange(104).reshape((13,8))*.13)
        mv=.9+theta*.04*jnp.cos(jnp.arange(108).reshape((12,9))*.21)
        params=ChannelParams(.01,.003,pressure_tolerance=1e-13,pressure_atol=1e-15,pressure_maxiter=4000,pressure_preconditioner=strategy)
        p,_,_=solve_pressure(rhs,mu,mv,grid,params)
        return .5*jnp.sum(p*p)
    fast=jax.jit(jax.value_and_grad(lambda t:objective(t,"constant-poisson-dct")))
    old=jax.jit(jax.value_and_grad(lambda t:objective(t,"none")))
    value,gradient=fast(.7)
    old_value,old_gradient=old(.7)
    np.testing.assert_allclose([value,gradient],[old_value,old_gradient],rtol=1e-9,atol=1e-12)
    difference=(fast(.70001)[0]-fast(.69999)[0])/.00002
    np.testing.assert_allclose(gradient,difference,rtol=2e-6,atol=1e-11)


def test_nested_manufactured_preflight_checks_changing_right_hand_sides():
    from jax_cad.differentiable.mac_channel import ChannelParams
    from jax_cad.differentiable.mac_pressure_preconditioner import manufactured_preflight
    grid=create_channel_grid(16,12,1.6,1.2)
    params=ChannelParams(.01,.003,pressure_tolerance=1e-12,pressure_atol=1e-14,pressure_maxiter=4000,pressure_preconditioner="constant-poisson-dct")
    mu=jnp.ones((17,12))*.87
    mv=jnp.ones((16,13))*.91
    error,residual=jax.jit(lambda a,b:manufactured_preflight(a,b,grid,params))(mu,mv)
    assert error<1e-10 and residual<1e-11


def test_fine_response_scope_is_opt_in_and_marker_bounded():
    from jax_cad.differentiable.mac_normal_constraint import ellipse_markers
    from jax_cad.differentiable.mac_normal_response import normal_stencil
    grid=create_channel_grid(880,164,2.2,.41)
    with pytest.raises(ValueError): ellipse_markers(jnp.array([.2,.2]),jnp.array([.05,.05]),0.,96)
    markers=ellipse_markers(jnp.array([.2,.2]),jnp.array([.05,.05]),0.,96,allow_fine=True)
    with pytest.raises(ValueError): normal_stencil(markers,grid)
    assert normal_stencil(markers,grid,allow_fine=True).valid
