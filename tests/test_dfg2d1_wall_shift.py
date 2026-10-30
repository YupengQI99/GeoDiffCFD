import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import dfg2d1_wall_shift_config as cfg
import wall_shift_dfg_config as old
from dfg2d1_wall_shift_metrics import analyze_history,quantity_statistics,REFERENCES


def test_new_schema_is_exact_and_not_a_reactivation_of_s15_or_stage2():
    paths=list((ROOT/'configs/dfg2d1_wall_shift').glob('*.json'));assert len(paths)==4
    for p in paths:
        s=json.loads(p.read_text());c=cfg.expand(s)
        previous=old.expand(old.specification('S1_2D1',s['phase']))
        for key,value in previous.items():
            if key not in ('schema','wall_shift_spec','wall_shift'):assert c[key]==value,key
        assert c['mean_inlet']*.1/c['viscosity']==pytest.approx(20)
        assert 2/(c['mean_inlet']**2*c['reference_diameter'])==pytest.approx(500)
        assert c['wall_shift']==(.0031020 if s['case']=='S1_2D1' else 0.)
        assert c['n_steps']==(200 if s['phase']=='preflight' else 32000)
        assert c['window_start_step']==(100 if s['phase']=='preflight' else 28000)
        support=.05+.0075-c['wall_shift'];assert min(.2,2.2-.2,.2,.41-.2)-support>.02
        with pytest.raises(ValueError):old.expand(s)
    s=cfg.specification('S1_2D1','production')
    for diff in ({'case':'S15'},{'case':'T1i'},{'wall_shift':.00310199302228},{'wall_shift':True},
                 {'normalize_level_set':True},{'normalize_level_set':0},{'dt':.00025},
                 {'n_steps':40000},{'phase':'retry'},{'paper_eligible':True},{'schema':old.specification('S1','production')['schema']}):
        with pytest.raises(ValueError):cfg.expand(dict(s,**diff))
    with pytest.raises(ValueError):cfg.expand(old.specification('S1_2D1','production'))


def test_steady_means_errors_and_fixed_physical_window():
    h=np.zeros(32000,dtype=[('time',float),('Cd',float),('Cl',float)])
    h['time']=.0005*np.arange(1,32001);h['Cd'][:28000]=1e6;h['Cl'][:28000]=-1e6
    cd=REFERENCES['Cd'][0]*1.03;cl=REFERENCES['Cl'][0]*1.2
    h['Cd'][28000:30000]=cd*(1-2e-5);h['Cd'][30000:]=cd*(1+2e-5)
    h['Cl'][28000:30000]=cl*(1-.001);h['Cl'][30000:]=cl*(1+.001)
    r=analyze_history(h)
    assert r['quantities']['Cd']['mean_14_16']==pytest.approx(cd)
    assert r['quantities']['Cd']['signed_point_error_percent']==pytest.approx(3)
    assert r['quantities']['Cl']['signed_point_error_percent']==pytest.approx(20)
    assert r['quantities']['Cd']['relative_half_window_drift']==pytest.approx(4e-5)
    assert r['quantities']['Cl']['relative_half_window_drift']==pytest.approx(.002)
    assert r['Cd_citation_drift_passed'] and not r['permits_stage2'] and not r['paper_eligible']
    h['Cd'][30000:]*=1.001
    assert not analyze_history(h)['Cd_citation_drift_passed']
    with pytest.raises(ValueError):analyze_history(h[:-1])
    h['time'][29000]+=.00025
    with pytest.raises(AssertionError):analyze_history(h)


def test_undefined_zero_mean_drift_is_not_zero():
    r=quantity_statistics(np.r_[-np.ones(2000),np.ones(2000)],*REFERENCES['Cl'])
    assert r['relative_half_window_drift'] is None and r['drift_undefined_reason']=='zero full-window mean'


def test_wake_driver_reused_and_discrete_flux_initial_state():
    import run_dfg2d1_wall_shift as driver
    import run_wall_shift_dfg as previous
    assert driver.main is previous.main
    from jax_cad.differentiable.mac_channel import create_channel_grid,parabolic_inlet
    from jax_cad.differentiable.wake_design import initial_flow,shape_fields
    from wall_shift_observer import masks
    c=cfg.expand(cfg.specification('W0_2D1','preflight'));g=create_channel_grid(**c['grid'])
    inlet=parabolic_inlet(g,.2,dtype=jnp.float64);state=initial_flow(g,inlet)
    assert float(inlet.sum()*g.dy)==pytest.approx(.2*.41,abs=1e-14)
    assert float(inlet.max())==pytest.approx(.3,abs=3e-5)
    np.testing.assert_array_equal(state.u,jnp.broadcast_to(inlet,state.u.shape))
    np.testing.assert_array_equal(state.v,np.zeros(state.v.shape))
    args=dict(center=c['center'],radius=.05,alpha_max=2560,epsilon=.0075)
    a=shape_fields(jnp.zeros(2),g,**args);b=shape_fields(jnp.zeros(2),g,**args,wall_shift=0.,normalize_level_set=False)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    cells,planes,_=masks(np.asarray(a.level_set),np.asarray(b.level_set),c)
    np.testing.assert_array_equal(cells[0],cells[1]);np.testing.assert_array_equal(planes[0],planes[1])


def test_steady_force500_passive_observer_matches_original_wake():
    from jax_cad.differentiable.mac_channel import create_channel_grid,ChannelParams,parabolic_inlet
    from jax_cad.differentiable.wake_design import initial_flow,shape_fields,make_wake_kernels
    from jax_cad.differentiable.wake_leakage import make_observed_forward
    g=create_channel_grid(32,16,2.2,.41)
    p=ChannelParams(.001,.0005,include_advection=True,advection_scheme='central',pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(g,.2,dtype=jnp.float64);state=initial_flow(g,inlet)
    q=shape_fields(jnp.zeros(2),g,center=[.2,.2],radius=.05,alpha_max=2560.,epsilon=.0075,wall_shift=.0031020)
    cells=np.zeros((2,32,16),bool);cells[:,2:4,6:10]=True
    planes=np.zeros((2,33,16),bool);planes[:,3,6:10]=True
    args=(state,q.momentum_resistance,inlet,jnp.ones(2)/2,jnp.array(2))
    scale=2/(.2**2*.1)
    a=make_wake_kernels(g,p,chunk_steps=2,force_scale=scale)[0](*args)
    b=make_observed_forward(g,p,chunk_steps=2,force_scale=scale,mean_inlet=.2)(*args,jnp.array(cells),jnp.array(planes))
    for x,y in zip(jax.tree.leaves(a[0]),jax.tree.leaves(b[0])):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[1],b[1],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[2],b[2][:,:7],rtol=1e-12,atol=1e-12)
