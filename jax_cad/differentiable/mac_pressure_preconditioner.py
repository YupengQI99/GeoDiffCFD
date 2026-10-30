"""Opt-in constant-Poisson preconditioner; same variable-mobility pressure PDE.

The MAC pressure operator has Neumann inlet/walls and a half-cell Dirichlet
outlet. Antisymmetric extension across the outlet turns it into a 2nx-by-ny
all-Neumann operator. Orthonormal type-II DCT diagonalises that operator;
the zero mean mode is removed and only the original half is returned.
This is an SPD PRECONDITIONER, not a replacement of variable mobility.
"""
import jax
import jax.numpy as jnp
from jax.scipy.fft import dctn, idctn


def constant_pressure_inverse(rhs, grid):
    if rhs.shape != (grid.nx,grid.ny) or rhs.dtype != jnp.float64:
        raise ValueError("constant pressure inverse requires native float64 cells")
    extended=jnp.concatenate((rhs,-rhs[::-1,:]),axis=0)
    kx=jnp.arange(2*grid.nx,dtype=rhs.dtype)
    ky=jnp.arange(grid.ny,dtype=rhs.dtype)
    eigenvalues=(4*jnp.sin(jnp.pi*kx/(4*grid.nx))**2/grid.dx**2)[:,None]+(4*jnp.sin(jnp.pi*ky/(2*grid.ny))**2/grid.dy**2)[None,:]
    safe=eigenvalues.at[0,0].set(1.)
    spectral=dctn(extended,type=2,norm="ortho")/safe
    spectral=spectral.at[0,0].set(0.)
    return idctn(spectral,type=2,norm="ortho")[:grid.nx,:]


def preconditioned_cg(operator,rhs,grid,*,tolerance,atol,maxiter):
    """Explicit data-dependent loop; differentiability is supplied by caller.

    Returns recursive iteration diagnostics AND a freshly recomputed residual.
    The old nested custom_linear_solve path is intentionally not used.
    """
    x=jnp.zeros_like(rhs)
    r=rhs-operator(x)
    z=constant_pressure_inverse(r,grid)
    rz=jnp.real(jnp.vdot(r,z))
    rr=jnp.real(jnp.vdot(r,r))
    norm_rhs=jnp.linalg.norm(rhs)
    threshold=jnp.maximum((tolerance*norm_rhs)**2,jnp.asarray(atol,dtype=rhs.dtype)**2)
    def condition(state):
        _,_,_,_,rr,k=state
        return (rr>threshold)&(k<maxiter)&jnp.isfinite(rr)
    def body(state):
        x,r,d,rz,_,k=state
        ad=operator(d)
        step=rz/jnp.real(jnp.vdot(d,ad))
        next_x=x+step*d
        next_r=r-step*ad
        next_z=constant_pressure_inverse(next_r,grid)
        next_rz=jnp.real(jnp.vdot(next_r,next_z))
        next_d=next_z+(next_rz/rz)*d
        return next_x,next_r,next_d,next_rz,jnp.real(jnp.vdot(next_r,next_r)),k+1
    x,_,_,_,rr,k=jax.lax.while_loop(condition,body,(x,r,z,rz,rr,jnp.array(0,dtype=jnp.int32)))
    fresh=jnp.linalg.norm(operator(x)-rhs)
    return x,k,jnp.sqrt(rr),fresh,fresh/jnp.maximum(norm_rhs,jnp.finfo(rhs.dtype).tiny)


def manufactured_preflight(mobility_u,mobility_v,grid,params):
    """Three changing exact solutions in a while-loop catch stale CUDA solves."""
    from .mac_channel import weighted_pressure_operator, solve_pressure
    ix=jnp.arange(grid.nx,dtype=jnp.float64)[:,None]
    iy=jnp.arange(grid.ny,dtype=jnp.float64)[None,:]
    def body(state):
        i,max_error,max_residual=state
        exact=jnp.sin((i+1)*.17*ix)*jnp.cos((i+2)*.09*iy)+.2
        rhs=weighted_pressure_operator(exact,mobility_u,mobility_v,grid)
        actual,_,relative=solve_pressure(rhs,mobility_u,mobility_v,grid,params)
        error=jnp.max(jnp.abs(actual-exact))/jnp.max(jnp.abs(exact))
        return i+1,jnp.maximum(max_error,error),jnp.maximum(max_residual,relative)
    _,error,residual=jax.lax.while_loop(lambda s:s[0]<3,body,(jnp.array(0),jnp.array(0.),jnp.array(0.)))
    return error,residual
