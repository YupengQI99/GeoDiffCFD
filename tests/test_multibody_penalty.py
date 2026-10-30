"""Bounded scope, passive-observer equivalence, and leakage normalization."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from multibody_penalty_config import expand
from multibody_wake_config import validate_config
from multibody_penalty_storage import ExternalStorage
from audit_multibody_penalty import discrete_inlet
from jax_cad.differentiable.wake_leakage import core_masks,observe,make_observed_forward,statistics
from jax_cad.differentiable.mac_channel import ChannelState,ChannelParams,create_channel_grid,parabolic_inlet
from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels
jax.config.update('jax_enable_x64',True)


def test_independent_inlet_uses_discrete_flux_normalization():
    for ny in (6,82,164):
        grid=create_channel_grid(12,ny,2.2,.41)
        c=dict(grid=dict(ny=ny,ly=.41),mean_inlet=1)
        actual=discrete_inlet(c)
        np.testing.assert_allclose(actual,parabolic_inlet(grid,1,dtype=jnp.float64),rtol=1e-14)
        np.testing.assert_allclose(actual.sum()*grid.dy,.41,rtol=1e-14)


def test_exact_scope_and_old_limits_preserved():
    files=list((ROOT/'configs/multibody_penalty').glob('*.json'))
    assert len(files)==8
    for path in files:
        s=json.loads(path.read_text());c=expand(s)
        assert c['max_optimization_steps']==0
        assert c['n_steps']==(200 if s['phase']=='preflight' else 24000)
        if s['shape']=='initial': assert c['theta']==[0.]*12
        else: assert c['theta'][3]==-0.3941003562933799
        with pytest.raises(ValueError):validate_config(c)
        for change in ({'case':'D'},{'theta':[0]*12},{'paper_eligible':True},{'final_sha256':'0'*64},{'shape':'candidate8'}):
            with pytest.raises(ValueError):expand(dict(s,**change))


def test_core_speed_and_absolute_flux():
    g=create_channel_grid(8,6, .8,.6)
    phi=np.full((3,8,6),.1);phi[:,2:5,1:5]=-.03
    c=dict(epsilon=.0075,grid=dict(lx=.8,ly=.6),geometry=dict(bodies=[dict(center=[.4,.3])]*3))
    cells,planes,meta=core_masks(phi,c)
    assert meta['cell_counts']==[12]*6
    state=ChannelState(jnp.ones((9,6))*2,jnp.zeros((8,7)),jnp.zeros((8,6)))
    inlet=jnp.ones(6)*2
    values=np.asarray(observe(state,jnp.array(cells),jnp.array(planes),inlet,dy=g.dy,mean_inlet=2))
    np.testing.assert_allclose(values[:12],1)
    np.testing.assert_allclose(values[12:],4/6)
    reversed_state=state._replace(u=state.u.at[4,1:3].set(-2))
    reverse=np.asarray(observe(reversed_state,jnp.array(cells),jnp.array(planes),inlet,dy=g.dy,mean_inlet=2))
    np.testing.assert_allclose(reverse[12:],values[12:])
    h=np.tile(np.r_[np.zeros(7),values],(5,1));s=statistics(h,2)
    assert s['sample_count']==3
    np.testing.assert_allclose(s['rms_speed_ratio'],1)
    with pytest.raises(ValueError):core_masks(np.ones((3,8,6)),c)


def test_passive_observer_does_not_change_forward():
    g=create_channel_grid(12,8,1.2,.8)
    p=ChannelParams(.03,.002,include_advection=True,advection_scheme='central',pressure_tolerance=1e-13,
        pressure_atol=1e-15,pressure_maxiter=1000,pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(g,.2,dtype=jnp.float64);state=initial_flow(g,inlet)
    alpha=jnp.ones((12,8))*10;weights=jnp.array([0.,.3,.7,0.]);active=jnp.array(3)
    cells=jnp.ones((6,12,8),dtype=bool);planes=jnp.zeros((6,13,8),dtype=bool).at[:,4,:].set(True)
    old=make_wake_kernels(g,p,chunk_steps=4,force_scale=2)[0](state,alpha,inlet,weights,active)
    new=make_observed_forward(g,p,chunk_steps=4,force_scale=2,mean_inlet=.2)(state,alpha,inlet,weights,active,cells,planes)
    for x,y in zip(jax.tree.leaves(old[0]),jax.tree.leaves(new[0])):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(old[1],new[1],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(old[2],np.asarray(new[2])[:,:7],rtol=1e-11,atol=1e-12)
    assert np.isfinite(new[2]).all()


def test_external_scope_and_root_stop(tmp_path):
    policy=ExternalStorage()
    with pytest.raises(ValueError):policy.prepare(ROOT/'unauthorised-output')
    policy.runtime=tmp_path
    values=dict(disk_free_gib=1000,disk_used_fraction=.87,available_ram_fraction=.6,
        gpu_memory_mib=17000,temperature_c=60,cpu_temperature_c=70,nvme_temperature_c=50,system_disk_loss_gib=0)
    assert policy.rejection(values,startup=False,low_disk_exception=True) is None
    for change in (dict(system_disk_loss_gib=.126),dict(disk_free_gib=29),dict(gpu_memory_mib=22001)):
        assert policy.rejection(dict(values,**change),startup=False,low_disk_exception=True)
