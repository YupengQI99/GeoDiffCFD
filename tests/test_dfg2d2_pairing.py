import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from dfg2d2_config import expand
from dfg2d2_metrics import analyze_history,relative_error
from jax_cad.differentiable.wake_leakage import core_masks,make_observed_forward,statistics,columns_for_bodies
from jax_cad.differentiable.mac_channel import create_channel_grid,ChannelParams,parabolic_inlet
from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels


def test_registry_is_exact_and_conditional_capacity_separate():
    files=list((ROOT/'configs/dfg2d2_pairing').glob('*.json'))
    assert len(files)==10
    for path in files:
        s=json.loads(path.read_text());c=expand(s)
        assert c['theta']==[0,0] and c['center']==[.2,.2] and c['radius']==.05
        if s['phase']=='production':assert c['n_steps']*c['dt']==10 and c['window_start_step']*c['dt']==8
        else:assert c['n_steps']==200
        for change in ({'case':'D'},{'dt_factor':4},{'dt_factor':True},{'n_steps':96000},{'theta':[.1,0]},{'paper_eligible':True}):
            with pytest.raises(ValueError):expand(dict(s,**change))


def test_one_body_passive_observer_equivalence():
    g=create_channel_grid(8,6,.8,.6)
    c=dict(grid=dict(lx=.8,ly=.6),epsilon=.0075,geometry=dict(bodies=[dict(center=[.2,.3])]))
    phi=np.full((1,8,6),.1);phi[:,1:4,1:5]=-.03
    cells,planes,meta=core_masks(phi,c)
    assert cells.shape==(2,8,6) and len(meta['labels'])==2 and len(columns_for_bodies(1))==6
    inlet=parabolic_inlet(g,.2,dtype=jnp.float64);state=initial_flow(g,inlet)
    p=ChannelParams(.03,.002,include_advection=True,advection_scheme='central',pressure_tolerance=1e-13,
        pressure_atol=1e-15,pressure_maxiter=1000,pressure_preconditioner='constant-poisson-dct')
    alpha=jnp.ones((8,6))*10;weights=jnp.ones(3)/3;active=jnp.array(3)
    old=make_wake_kernels(g,p,chunk_steps=3,force_scale=20)[0](state,alpha,inlet,weights,active)
    new=make_observed_forward(g,p,chunk_steps=3,force_scale=20,mean_inlet=.2)(state,alpha,inlet,weights,active,jnp.array(cells),jnp.array(planes))
    for a,b in zip(old[0],new[0]):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(old[1],new[1],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(old[2],np.asarray(new[2])[:,:7],rtol=1e-11,atol=1e-12)
    assert len(statistics(new[2],0)['mean_speed_ratio'])==2


def synthetic_history():
    t=np.arange(8001,10001)*.001
    h=np.zeros(len(t),dtype=[('time',float),('Cd',float),('Cl',float)])
    h['time']=t;h['Cd']=3.23+.01*np.cos(2*np.pi*6*t);h['Cl']=.01+np.sin(2*np.pi*3*t)
    return h


def test_period_drift_and_failure_are_not_selected_after_run():
    h=synthetic_history();a=analyze_history(h,0)
    assert a['passed_benchmark_metrics'] and a['full_window']['St']==pytest.approx(.3,abs=1e-7)
    assert a['full_window']['period_cv']<1e-6 and a['maximum_acceptance_drift']<1e-5
    h['Cd']*=1.4;assert not analyze_history(h,0)['gates']['Cd_max']
    h=synthetic_history();h['Cd'][1000:]*=1.1
    assert not analyze_history(h,0)['gates']['half_window_drift']
    h['Cl']=0;a=analyze_history(h,0)
    assert a['full_window']['St'] is None and not a['passed_benchmark_metrics']


def test_reference_midpoint_not_nearest_favorable_endpoint():
    r=relative_error(3.4,(3.22,3.24))
    assert r['absolute_midpoint_percent']==pytest.approx(100*(3.4/3.23-1))
    assert r['outside_interval_percent']<r['absolute_midpoint_percent']


def test_half_step_requires_original_index_trigger(monkeypatch):
    from dfg2d2_config import require_conditional_half_step
    import audit_dfg2d2
    import wake_window_metrics
    monkeypatch.setattr(audit_dfg2d2,'audit',lambda directory:dict(passed=True))
    monkeypatch.setattr(wake_window_metrics,'window_metrics',lambda directory:dict(component_screening_index=1.0))
    spec=dict(case='C',dt_factor=2)
    with pytest.raises(ValueError,match='index >1'):require_conditional_half_step(spec)
    monkeypatch.setattr(wake_window_metrics,'window_metrics',lambda directory:dict(component_screening_index=1.01))
    require_conditional_half_step(spec)
