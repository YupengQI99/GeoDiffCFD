"""Three fixed-area native geometries feeding ONE common material field.

No flow kernel, force law, boundary operator or transient adjoint is changed.
The product of compact fluid fractions is smooth; a separate whole-box
certificate keeps their non-fluid supports disjoint. A minimum level set is
provided for plotting only and is NOT in the differentiated solver interface.
"""
import jax.numpy as jnp
from .mac_channel import cell_centres
from jax_cad.unified.geometry import fourier_boundary_radius
from jax_cad.unified.materials import MaterialFields, compact_heaviside, compact_delta


def body_coefficients(theta, geometry):
    z=jnp.asarray(theta)
    if z.shape!=(12,): raise ValueError("exactly 12 native multibody variables")
    delta=z.reshape(3,4)*jnp.asarray(geometry["design_scales"],dtype=z.dtype)
    base=jnp.asarray([b["base_coefficients"] for b in geometry["bodies"]],dtype=z.dtype)
    return base.at[:,jnp.array([0,3,1,4])].add(delta)


def geometry_fields(theta, grid, config):
    coefficients=body_coefficients(theta,config["geometry"])
    x,y=cell_centres(grid,dtype=coefficients.dtype)
    phis=[]
    for i,b in enumerate(config["geometry"]["bodies"]):
        dx,dy=x-b["center"][0],y-b["center"][1]
        angle=jnp.arctan2(dy,dx)
        radius=fourier_boundary_radius(angle,coefficients[i],base_radius=b["radius"],
                                       modes=config["geometry"]["modes"],n_quadrature=1024)
        phis.append(jnp.sqrt(dx*dx+dy*dy+jnp.finfo(coefficients.dtype).tiny)-radius)
    phis=jnp.stack(phis)
    effective=phis
    if config.get('wall_shift',0.) != 0 or config.get('normalize_level_set',False):
        from .wall_shift import corrected_level_set
        effective=[]
        for i,b in enumerate(config['geometry']['bodies']):
            angle=jnp.arctan2(y-b['center'][1],x-b['center'][0])
            effective.append(corrected_level_set(phis[i],angle,coefficients[i],config['geometry']['modes'],
                wall_shift=config.get('wall_shift',0.),normalize_level_set=config.get('normalize_level_set',False)))
        effective=jnp.stack(effective)
    fractions=compact_heaviside(effective,config["epsilon"])
    chi=jnp.prod(fractions,axis=0)
    q=MaterialFields(chi,config["alpha_max"]*(1-chi),jnp.min(effective,axis=0),None,None)
    return q,phis,fractions,coefficients


def body_level_set_cotangent(material_cotangent, phis, fractions, config):
    if config.get('normalize_level_set',False) or config.get('wall_shift',0.) != 0:
        raise ValueError('legacy radial-cotangent diagnostic is not registered for corrected geometry; forward-only pilot')
    other=jnp.stack([jnp.prod(fractions[jnp.array([j for j in range(3) if j!=i])],axis=0)
                     for i in range(3)])
    return -config["alpha_max"]*compact_delta(phis,config["epsilon"])*other*material_cotangent
