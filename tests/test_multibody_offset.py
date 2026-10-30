import json
import sys
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import multibody_offset_config as cfg
import multibody_offset_observer as obs


def test_exact_registration_does_not_reopen_old_gates():
    from multibody_penalty_config import expand as old
    from wall_shift_dfg_config import expand as stage1
    files=list((ROOT/'configs/multibody_offset').glob('*.json'));assert len(files)==8
    for f in files:
        s=json.loads(f.read_text());c=cfg.expand(s)
        assert c['grid']==dict(nx=880,ny=164,lx=2.2,ly=.41)
        assert c['alpha_max']==2560 and c['epsilon']==.0075 and c['dt']==.000125
        assert c['n_steps']==(200 if s['phase']=='preflight' else 48000)
        assert c['window_start_step']==(100 if s['phase']=='preflight' else 32000)
        assert c['max_optimization_steps']==0 and c['advection_scheme']=='central'
        assert c['normalize_level_set']==s['case'].startswith('T1')
        with pytest.raises(ValueError):old(s)
        with pytest.raises(ValueError):stage1(s)
    s=cfg.specification('T1i','production')
    for diff in ({'case':'S15'},{'wall_shift':.003101993},{'wall_shift':True},
                 {'normalize_level_set':1},{'normalize_level_set':False},{'phase':'retry'},
                 {'paper_eligible':True},{'dt':.00025},{'n_steps':96000},{'shape':'final10'}):
        with pytest.raises(ValueError):cfg.expand(dict(s,**diff))


@pytest.mark.parametrize('case',list(cfg.CASES))
def test_actual_fine_grid_mapping_cores_and_confirmed_reference(case):
    from jax_cad.differentiable.mac_channel import create_channel_grid
    from jax_cad.differentiable.multibody_wake import geometry_fields
    from audit_wall_shift_geometry import quantities,body_parameters
    c=cfg.expand(cfg.specification(case,'production'));g=create_channel_grid(**c['grid'])
    q,phi,chi,coeff=geometry_fields(jnp.array(c['theta']),g,c)
    geom=cfg.geometry_checks(c);assert geom['envelope']['passed']
    x,y=np.meshgrid((np.arange(880)+.5)*.0025,(np.arange(164)+.5)*.0025,indexing='ij')
    coeff,m=body_parameters(c);expected=[];nominal=[]
    for i,b in enumerate(c['geometry']['bodies']):
        a=quantities(np.hypot(x-b['center'][0],y-b['center'][1]),np.arctan2(y-b['center'][1],x-b['center'][0]),coeff[i],m,b['radius'])
        nominal.append(a['phi']);expected.append(a['normalized' if c['normalize_level_set'] else 'phi']+c['wall_shift'])
    expected=np.array(expected);s=np.clip(expected/.0075,-1,1);fraction=.5+s*(15/16+s*s*(-5/8+s*s*3/16))
    np.testing.assert_allclose(phi,nominal,atol=1e-14)
    np.testing.assert_allclose(q.level_set,expected.min(axis=0),atol=1e-14)
    np.testing.assert_allclose(chi,fraction,atol=1e-13)
    np.testing.assert_allclose(q.momentum_resistance,2560*(1-fraction.prod(axis=0)),atol=1e-10)
    cells,planes,meta=obs.core_masks(phi,c)
    assert meta['labels']==obs.LABELS and cells.shape==(6,880,164) and planes.shape==(6,881,164)
    for i in range(3):
        np.testing.assert_array_equal(cells[2*i],np.array(nominal[i])<=-.015)
        np.testing.assert_array_equal(cells[2*i+1],expected[i]<=-.015)
    # w=0/raw preserves every legacy geometry array bit for bit.
    legacy=dict(c);legacy.pop('wall_shift');legacy.pop('normalize_level_set')
    a=geometry_fields(jnp.array(c['theta']),g,legacy)
    b=geometry_fields(jnp.array(c['theta']),g,dict(legacy,wall_shift=0.,normalize_level_set=False))
    for v,w in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(v,w)


def test_passive_six_mask_observer_preserves_original_forward():
    from jax_cad.differentiable.mac_channel import create_channel_grid,ChannelParams,parabolic_inlet
    from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels
    from jax_cad.differentiable.multibody_wake import geometry_fields
    from jax_cad.differentiable.wake_leakage import make_observed_forward
    c=cfg.expand(cfg.specification('T1f','preflight'));c['grid']=dict(nx=32,ny=16,lx=2.2,ly=.41)
    g=create_channel_grid(**c['grid']);p=ChannelParams(.001,.000125,include_advection=True,
        advection_scheme='central',pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(g,1.,dtype=jnp.float64);state=initial_flow(g,inlet)
    q=geometry_fields(jnp.array(c['theta']),g,c)[0]
    cells=np.zeros((6,32,16),bool);cells[:,2:4,6:10]=True
    planes=np.zeros((6,33,16),bool);planes[:,3,6:10]=True
    args=(state,q.momentum_resistance,inlet,jnp.ones(2)/2,jnp.array(2))
    a=make_wake_kernels(g,p,chunk_steps=2,force_scale=20)[0](*args)
    b=make_observed_forward(g,p,chunk_steps=2,force_scale=20,mean_inlet=1.)(*args,jnp.array(cells),jnp.array(planes))
    for v,w in zip(jax.tree.leaves(a[0]),jax.tree.leaves(b[0])):np.testing.assert_allclose(v,w,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[1],b[1],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[2],b[2][:,:7],rtol=1e-12,atol=1e-12)
    assert b[2].shape==(2,25)
    assert obs.statistics(b[2],0)['sample_count']==2


def test_driver_reuses_existing_solver():
    import run_multibody_offset as new
    import run_multibody_penalty as old
    assert new.main is old.main
