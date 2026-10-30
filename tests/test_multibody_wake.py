"""Small CPU/CUDA tests of the twelve-variable material-only transient chain."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax_cad.differentiable.mac_channel import ChannelParams,create_channel_grid,parabolic_inlet,step
from jax_cad.differentiable.multibody_wake import geometry_fields,body_level_set_cotangent
from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels,evaluate_trajectory
from jax_cad.unified.geometry import fourier_shape_area
from jax_cad.verification import brinkman_force

jax.config.update("jax_enable_x64",True)
ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"configs/multibody_wake/coarse_initial.json").read_text())


def analytic_pullback(z, cotangent, grid, c):
    q,phis,fractions,coeff=geometry_fields(z,grid,c)
    lam=np.asarray(body_level_set_cotangent(cotangent,phis,fractions,c))
    g=c["geometry"]; angles=2*np.pi*(np.arange(1024)+.5)/1024
    basis=np.stack([np.cos(m*angles) for m in g["modes"]]+[np.sin(m*angles) for m in g["modes"]])
    x,y=np.meshgrid((np.arange(grid.nx)+.5)*grid.dx,(np.arange(grid.ny)+.5)*grid.dy,indexing="ij")
    derivatives=[]
    for i,b in enumerate(g["bodies"]):
        a=np.arctan2(y-b["center"][1],x-b["center"][0])
        local=np.stack([np.cos(m*a) for m in g["modes"]]+[np.sin(m*a) for m in g["modes"]])
        exponent=np.exp(2*np.asarray(coeff[i])@basis); normal=.5*np.log(exponent.mean())
        dc=(basis*exponent).mean(axis=1)/exponent.mean()
        radius=b["radius"]*np.exp(np.einsum("i,ijk->jk",np.asarray(coeff[i]),local)-normal)
        full=-radius*(local-dc[:,None,None])
        derivatives.extend(np.sum(full[[0,3,1,4]]*lam[i],axis=(1,2))*g["design_scales"])
    return np.array(derivatives)


def test_native_areas_and_material_union():
    c=config(); grid=create_channel_grid(220,41,2.2,.41)
    for z in (jnp.zeros(12),jnp.linspace(-1,1,12),jnp.ones(12),-jnp.ones(12)):
        q,phi,chi,coef=geometry_fields(z,grid,c)
        areas=[float(fourier_shape_area(coef[i],base_radius=b["radius"],modes=(2,3,4),n_quadrature=4096))
               for i,b in enumerate(c["geometry"]["bodies"])]
        np.testing.assert_allclose(areas,np.pi*.05**2,rtol=1e-12)
        np.testing.assert_allclose(q.momentum_resistance,c["alpha_max"]*jnp.sum(1-chi,axis=0),atol=1e-12)
        assert np.max(np.sum(np.asarray(phi)<c["epsilon"],axis=0))==1
        assert np.min(q.momentum_resistance)>=0 and np.max(q.momentum_resistance)<=c["alpha_max"]


def test_independent_analytic_geometry_pullback():
    c=config(); grid=create_channel_grid(110,41,2.2,.41); z=jnp.linspace(-.2,.2,12)
    fn=lambda x:geometry_fields(x,grid,c)[0].momentum_resistance
    alpha,pullback=jax.vjp(fn,z)
    cot=jnp.sin(jnp.arange(alpha.size,dtype=jnp.float64).reshape(alpha.shape)*.1)
    analytic=analytic_pullback(z,cot,grid,c)
    np.testing.assert_allclose(pullback(cot)[0],analytic,rtol=1e-10,atol=1e-10)
    assert np.all(np.abs(analytic)>1e-8)


def test_full_transient_multibody_adjoint_against_serial_ad():
    c=config(); c["alpha_max"]=40.
    grid=create_channel_grid(44,16,2.2,.41)
    params=ChannelParams(.003,.002,include_advection=True,advection_scheme="central",
        pressure_tolerance=1e-13,pressure_atol=1e-15,pressure_maxiter=1000,pressure_preconditioner="constant-poisson-dct")
    inlet=parabolic_inlet(grid,.5,dtype=jnp.float64); initial=initial_flow(grid,inlet)
    material=lambda z:geometry_fields(z,grid,c)[0].momentum_resistance
    z=jnp.linspace(-.12,.12,12); alpha,pullback=jax.vjp(material,z)
    kernels=make_wake_kernels(grid,params,chunk_steps=4,force_scale=1.)
    def trajectory(t,gradient=False):
        return evaluate_trajectory(material(t),inlet,initial,kernels,n_steps=9,chunk_steps=4,window_start_step=3,gradient=gradient)
    result=trajectory(z,True); grad=pullback(result["material_gradient"])[0]
    single=jax.jit(lambda s,a:step(s,a,inlet,grid,params)[0])
    def serial(t):
        a=material(t); s=initial; value=0.
        for k in range(9):
            s=single(s,a)
            if k>=3: value=value+brinkman_force(s,a,grid)[0]/6
        return value
    expected,expected_grad=jax.value_and_grad(serial)(z)
    np.testing.assert_allclose(result["value"],expected,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(grad,expected_grad,atol=1e-11,rtol=1e-8)
    d=jnp.asarray(c["taylor_direction"]); d=d/jnp.linalg.norm(d)
    eps=np.array([.008,.004,.002,.001])
    errors=[abs(trajectory(z+e*d)["value"]-result["value"]-e*float(grad@d)) for e in eps]
    rates=np.log2(np.array(errors[:-1])/errors[1:])
    fd=(trajectory(z+1e-4*d)["value"]-trajectory(z-1e-4*d)["value"])/2e-4
    np.testing.assert_allclose(grad@d,fd,rtol=1e-5,atol=1e-10)
    assert min(rates[-2:])>1.9
    print(json.dumps(dict(check="multibody_full_transient_adjoint",backend=jax.default_backend(),
        value=result["value"],gradient=np.asarray(grad).tolist(),taylor_rates=rates.tolist(),finite_difference=fd,
        serial_gradient_max_error=float(np.max(abs(np.asarray(grad-expected_grad)))),paper_eligible=False)),flush=True)
