import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from wall_shift_dfg_config import expand,specification,output_path,SHIFTS
from wall_shift_observer import masks,statistics,measured_radius
from jax_cad.differentiable.mac_channel import create_channel_grid,ChannelParams,parabolic_inlet
from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels,shape_fields
from jax_cad.differentiable.wake_leakage import make_observed_forward


def test_exact_stage_one_and_changed_config_envelopes():
    paths=list((ROOT/'configs/wall_shift_dfg').glob('*.json'))
    assert len(paths)==8
    for path in paths:
        s=json.loads(path.read_text());c=expand(s)
        assert c['grid']==dict(nx=880,ny=164,lx=2.2,ly=.41)
        assert c['alpha_max']==2560 and c['epsilon']==.0075
        assert c['normalize_level_set'] is False and c['theta']==[0.,0.]
        radius=c['radius']+c['epsilon']-c['wall_shift']
        assert min(.2,2.2-.2,.2,.41-.2)-radius>=.02
        if s['phase']=='preflight':assert (c['n_steps'],c['window_start_step'])==(200,100)
        else:
            steady=s['case']=='S1_2D1'
            assert c['n_steps']*c['dt']==(16 if steady else 10)
            assert c['window_start_step']*c['dt']==(14 if steady else 8)
        assert output_path(s).name.endswith('_20260927')
    s=specification('S1','production')
    for change in ({'case':'T1i'},{'wall_shift':.0031},{'normalize_level_set':True},
                   {'normalize_level_set':0},{'dt':.000125},{'paper_eligible':True},
                   {'phase':'retry'},{'n_steps':80000}):
        with pytest.raises(ValueError):expand(dict(s,**change))


def test_both_cores_are_distinct_and_use_same_fixed_margin():
    c=expand(specification('S1','preflight'));g=create_channel_grid(**c['grid'])
    q=shape_fields(jnp.zeros(2),g,center=c['center'],radius=.05,alpha_max=2560.,epsilon=.0075,wall_shift=c['wall_shift'])
    x,y=np.meshgrid((np.arange(880)+.5)*g.dx,(np.arange(164)+.5)*g.dy,indexing='ij')
    nominal=np.hypot(x-.2,y-.2)-.05;eff=nominal+c['wall_shift']
    cells,planes,m=masks(nominal,eff,c)
    assert m['cell_counts'][0]>m['cell_counts'][1]>0
    assert not np.any(cells[1]&~cells[0]) and np.all(planes[:,80].sum(axis=1)>0)
    r=measured_radius(np.asarray(q.fluid_fraction),c)
    assert r['max_error_m']<g.dx/2 and r['maximum_m']<.05
    history=np.zeros((3,13));history[:,7:]=[1,2,4,9,.1,.2]
    stat=statistics(history,1)
    assert stat['labels']==['nominal_body1','corrected_body1']
    assert stat['mean_speed_ratio']==[1,2] and stat['rms_speed_ratio']==[2,3]


def test_new_core_observation_does_not_change_small_forward():
    g=create_channel_grid(32,16,2.2,.41)
    p=ChannelParams(.001,.00025,include_advection=True,advection_scheme='central',pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(g,1.,dtype=jnp.float64);state=initial_flow(g,inlet)
    q=shape_fields(jnp.zeros(2),g,center=[.2,.2],radius=.05,alpha_max=2560.,epsilon=.0075,wall_shift=.003102)
    cells=np.zeros((2,32,16),bool);cells[0,2:4,6:10]=True;cells[1,3:4,7:9]=True
    planes=np.zeros((2,33,16),bool);planes[0,3,6:10]=True;planes[1,3,7:9]=True
    old=make_wake_kernels(g,p,chunk_steps=2,force_scale=20)[0]
    new=make_observed_forward(g,p,chunk_steps=2,force_scale=20,mean_inlet=1.)
    args=(state,q.momentum_resistance,inlet,jnp.ones(2)/2,jnp.array(2))
    a=old(*args);b=new(*args,jnp.array(cells),jnp.array(planes))
    for x,y in zip(jax.tree.leaves(a[0]),jax.tree.leaves(b[0])):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[1],b[1],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[2],b[2][:,:7],rtol=1e-12,atol=1e-12)
