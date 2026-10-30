import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from jax_cad.brinkman_mask_calibration import calibrate_planar_mask_shift,compact_quintic_fluid_fraction
from jax_cad.differentiable.mac_channel import create_channel_grid,cell_centres,ChannelParams,parabolic_inlet
from jax_cad.differentiable.wake_design import shape_fields,initial_flow,make_wake_kernels
from jax_cad.differentiable.multibody_wake import geometry_fields,body_coefficients
from jax_cad.differentiable.wall_shift import corrected_level_set
from jax_cad.unified.geometry import fourier_boundary_radius
from jax_cad.unified.materials import compact_heaviside
from audit_wall_shift_geometry import candidate_config,geometry_audit,quantities,body_parameters,reference_checks,FD_REFERENCES
import quintic_wall_calibration as supplied


def test_reference_free_calibration_matches_independent_supplied_solver():
    cos=calibrate_planar_mask_shift(1.,epsilon_over_h=1.5)
    assert cos.shift_over_h==pytest.approx(.7809903339948505,abs=1e-11)
    assert supplied.calibrate(1.,1.5,profile=supplied.cosine)==pytest.approx(cos.shift_over_h,abs=1e-9)
    for sampling,other in [('tangential-cell','tangential-cell'),('normal-face-average-sensitivity','normal-face')]:
        c=calibrate_planar_mask_shift(.25,epsilon_over_h=3,profile='compact-quintic',sampling=sampling)
        s=supplied.calibrate(.25,3.,sampling=other)
        assert c.shift_over_h==pytest.approx(s,abs=1e-9)
        if sampling=='tangential-cell':assert -c.shift_over_h*.0025==pytest.approx(.00310199302228,abs=1e-12)
    bigger=calibrate_planar_mask_shift(.25,epsilon_over_h=3,profile='compact-quintic',solid_cells=128,fluid_cells=128,far_field_fit_cells=32)
    assert -bigger.shift_over_h*.0025==pytest.approx(.00310199302228,abs=1e-12)
    with pytest.raises(ValueError):calibrate_planar_mask_shift(.25,epsilon_over_h=3,profile='not-registered')


def test_quintic_numpy_matches_material_function():
    phi=np.linspace(-.02,.02,257)
    np.testing.assert_allclose(compact_quintic_fluid_fraction(phi,.0075),compact_heaviside(jnp.array(phi),.0075),atol=2e-16,rtol=0)


def test_fine_circle_half_fraction_radius_sign():
    g=create_channel_grid(880,164,2.2,.41);shift=.0031020
    q=shape_fields(jnp.zeros(2),g,center=[.2,.2],radius=.05,alpha_max=2560,epsilon=.0075,wall_shift=shift)
    chi=np.asarray(q.fluid_fraction);x=(np.arange(880)+.5)*g.dx;y=(np.arange(164)+.5)*g.dy
    radii=[]
    for j in range(164):
        row=chi[:,j]-.5
        for i in np.flatnonzero(row[:-1]*row[1:]<0):
            crossing=x[i]-row[i]*(x[i+1]-x[i])/(row[i+1]-row[i])
            radii.append(np.hypot(crossing-.2,y[j]-.2))
    assert len(radii)>50
    assert max(abs(np.array(radii)-(.05-shift)))<=g.dx/2
    assert max(radii)<.05
    normalized=shape_fields(jnp.zeros(2),g,center=[.2,.2],radius=.05,alpha_max=2560,epsilon=.0075,wall_shift=shift,normalize_level_set=True)
    np.testing.assert_array_equal(normalized.fluid_fraction,q.fluid_fraction)


def test_zero_shift_multibody_fields_and_forward_are_bitwise_legacy():
    c=candidate_config();c.update(wall_shift=0.,normalize_level_set=False)
    g=create_channel_grid(32,16,2.2,.41);theta=jnp.array(c['theta'])
    actual=geometry_fields(theta,g,c)
    coefficients=body_coefficients(theta,c['geometry']);x,y=cell_centres(g,dtype=coefficients.dtype)
    old=[]
    for i,b in enumerate(c['geometry']['bodies']):
        dx,dy=x-b['center'][0],y-b['center'][1];angle=jnp.arctan2(dy,dx)
        R=fourier_boundary_radius(angle,coefficients[i],base_radius=b['radius'],modes=c['geometry']['modes'],n_quadrature=1024)
        old.append(jnp.sqrt(dx*dx+dy*dy+jnp.finfo(coefficients.dtype).tiny)-R)
    old=jnp.stack(old);chi=jnp.prod(compact_heaviside(old,c['epsilon']),axis=0);alpha=c['alpha_max']*(1-chi)
    np.testing.assert_array_equal(actual[0].momentum_resistance,alpha)
    np.testing.assert_array_equal(actual[1],old)
    p=ChannelParams(.001,.000125,include_advection=True,advection_scheme='central',pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(g,1.,dtype=jnp.float64);state=initial_flow(g,inlet)
    forward=make_wake_kernels(g,p,chunk_steps=2,force_scale=20)[0]
    a=forward(state,alpha,inlet,jnp.ones(2)/2,jnp.array(2))
    b=forward(state,actual[0].momentum_resistance,inlet,jnp.ones(2)/2,jnp.array(2))
    for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(u,v)
    cfg=dict(center=[.2,.2],radius=.05,alpha_max=2560,epsilon=.0075)
    a=shape_fields(jnp.zeros(2),g,**cfg);b=shape_fields(jnp.zeros(2),g,**cfg,wall_shift=0.,normalize_level_set=False)
    for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(u,v)


def test_analytic_band_derivative_matches_autodiff_and_exposes_non_distance():
    c=candidate_config();coeff,m=body_parameters(c)
    for i,body in enumerate(c['geometry']['bodies']):
        z=jnp.array(coeff[i]);centre=jnp.array(body['center'])
        def field(xy):
            d=xy-centre;t=jnp.arctan2(d[1],d[0]);r=jnp.sqrt(jnp.sum(d*d))
            R=fourier_boundary_radius(t,z,base_radius=.05,modes=(2,3,4),n_quadrature=1024)
            return corrected_level_set(r-R,t,z,(2,3,4),wall_shift=.0031020,normalize_level_set=True)
        t=np.linspace(.1,6.2,31);r=np.full(31,.05)
        a=quantities(r,t,coeff[i],m,.05)
        xy=np.array(body['center'])+np.c_[r*np.cos(t),r*np.sin(t)]
        actual=jax.vmap(jax.grad(field))(jnp.array(xy))
        np.testing.assert_allclose(np.linalg.norm(actual,axis=1),a['gradient_norm'],rtol=1e-11,atol=1e-12)
    report=geometry_audit(c)
    assert report['envelope']['passed']
    # The diagnostic must reject the formula as uniformly unit-gradient.
    # This is a PASS of the diagnostic implementation, NOT of GPU prerequisite 3.
    assert not report['full_band_within_10_percent_diagnostic_only']
    assert report['bodies'][1]['bands']['corrected_band']['maximum']>1.39


def test_legacy_gpu_entries_reject_unregistered_corrections():
    from guard_wake_design import validate_config as wake_validate
    from multibody_wake_config import validate_config as multi_validate
    for validate in (wake_validate,multi_validate):
        for addition in ({'wall_shift':.003102},{'normalize_level_set':True}):
            with pytest.raises(ValueError,match='own registered entry'):validate(addition)


def test_fine_normalized_implementation_matches_numpy_reference_and_supplied_counts():
    import copy
    g=create_channel_grid(880,164,2.2,.41)
    for final in (False,True):
        c=candidate_config(final=final);audit=geometry_audit(c)
        q,nominal,body_chi,_=geometry_fields(jnp.array(c['theta']),g,c)
        coeff,m=body_parameters(c)
        x,y=np.meshgrid((np.arange(880)+.5)*.0025,(np.arange(164)+.5)*.0025,indexing='ij')
        effective=[]
        for i,b in enumerate(c['geometry']['bodies']):
            r=np.hypot(x-b['center'][0],y-b['center'][1]);t=np.arctan2(y-b['center'][1],x-b['center'][0])
            v=quantities(r,t,coeff[i],m,b['radius'])
            np.testing.assert_allclose(nominal[i],v['phi'],rtol=0,atol=1e-14)
            phi=v['normalized']+c['wall_shift'];effective.append(phi)
            np.testing.assert_allclose(body_chi[i],compact_quintic_fluid_fraction(phi,c['epsilon']),rtol=0,atol=1e-13)
        np.testing.assert_allclose(q.level_set,np.min(effective,axis=0),rtol=0,atol=1e-14)
        shape='final10' if final else 'initial';checks=reference_checks(shape,audit['bodies'])
        assert all(b['nominal_interface_gate'] for b in checks)
        assert audit['envelope']['passed']
        for b in checks:
            for region in b['regions']:
                if region['reference_available']:
                    assert region['endpoint_gate'] and region['exact_count_gate']
                    assert region['maximum_endpoint_difference']<.000051
                else:
                    assert not region['endpoint_gate'] and not region['exact_count_gate']
        # A single-cell mismatch must be detected independently of endpoints.
        altered=copy.deepcopy(audit['bodies']);altered[0]['finite_difference']['nominal_interface']['count']+=1
        bad=reference_checks(shape,altered)[0]['regions'][0]
        assert bad['endpoint_gate'] and not bad['exact_count_gate']


def test_missing_independent_reference_is_not_silently_accepted(monkeypatch):
    refs=list(FD_REFERENCES['final10']);refs[0]=(*refs[0][:2],None)
    monkeypatch.setitem(FD_REFERENCES,'final10',refs)
    fake=[]
    for i,refs in enumerate(FD_REFERENCES['final10']):
        fake.append(dict(name=str(i),finite_difference={
            key:dict(minimum=ref[0],maximum=ref[1],count=ref[2])
            if ref is not None else dict(minimum=.9405,maximum=1.2477,count=764)
            for key,ref in zip(('nominal_interface','corrected_interface','corrected_band'),refs)}))
    checks=reference_checks('final10',fake)
    assert not checks[0]['reference_complete']
    assert not checks[0]['reference_endpoint_gate'] and not checks[0]['reference_count_gate']
    for row,refs in zip(checks,FD_REFERENCES['final10']):
        assert row['reference_complete']==all(r is not None for r in refs)
