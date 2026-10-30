#!/usr/bin/env python3
"""Strict fixed-shape A/B/C forwards, with passive per-step solid-core records."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multibody_penalty_config import expand, EXTERNAL
from multibody_wake_config import envelope_report
from run_multibody_wake import write_json


def require_preflights(spec):
    from guard_mac_normal_tests import snapshot
    before=snapshot()
    from audit_multibody_wake import audit
    from audit_multibody_penalty import leakage_readback
    for shape in ('initial','final10'):
        suffix='dt2_preflight' if spec['schema']=='multibody-penalty-timecheck-1' else 'preflight_v2'
        directory=EXTERNAL/f'C_{shape}_{suffix}'
        audit(directory)
        leakage_readback(directory)
        source=json.loads(directory.with_name(directory.name+'.guard').joinpath('source.json').read_text())
        if source!=before: raise ValueError('C preflight must use identical clean source')
        r=json.loads((directory/'result.json').read_text())
        expected=dict(spec,shape=shape,phase='preflight')
        if r['sensitivity_spec']!=expected: raise ValueError('preflight must match timecheck schema and model')
        if not r['observer_equivalence']['passed']: raise ValueError('C observer equivalence failed')


def main(*, registry=None, observer=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['forward'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();c=(expand if registry is None else registry.expand)(json.loads(a.config.read_text()));s=c['sensitivity_spec'];output=a.output_dir
    if os.environ.get('JAX_PLATFORMS')!='cuda' or os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')!='0.70':
        raise RuntimeError('captured CUDA supervisor required')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if registry is not None: registry.require_preflight(s)
    elif s['case']=='C' and s['phase']=='production': require_preflights(s)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_cad.run_artifacts import provenance_record
    from jax_cad.differentiable.mac_channel import ChannelParams,create_channel_grid,parabolic_inlet
    from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels,evaluate_trajectory
    from jax_cad.differentiable.multibody_wake import geometry_fields
    from jax_cad.differentiable.wake_leakage import core_masks,make_observed_forward,statistics,COLUMNS
    if observer is not None:
        core_masks,statistics,COLUMNS=observer.core_masks,observer.statistics,observer.COLUMNS
    from jax_cad.unified.geometry import fourier_shape_area
    if jax.default_backend()!='gpu': raise RuntimeError('CUDA required')
    provenance=provenance_record();write_json(output/'environment.json',provenance)
    write_json(output/'config.json',c)
    write_json(output/'geometry_envelope.json',envelope_report(c) if registry is None else registry.geometry_checks(c))
    grid=create_channel_grid(**c['grid'])
    params=ChannelParams(c['viscosity'],c['dt'],include_advection=True,advection_scheme=c['advection_scheme'],
        pressure_tolerance=1e-11,pressure_atol=1e-13,pressure_maxiter=1000,pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(grid,c['mean_inlet'],dtype=jnp.float64);state0=initial_flow(grid,inlet)
    z=jnp.asarray(c['theta'],dtype=jnp.float64);q,phi,chi,coefficients=geometry_fields(z,grid,c)
    alpha=q.momentum_resistance
    cells,planes,core=core_masks(phi,c);core['discrete_inlet_flux']=float(inlet.sum()*grid.dy)
    write_json(output/'core_definition.json',core)
    cells,planes=jnp.asarray(cells),jnp.asarray(planes)
    scale=2/(c['mean_inlet']**2*c['reference_diameter'])
    observed=make_observed_forward(grid,params,chunk_steps=c['chunk_steps'],force_scale=scale,mean_inlet=c['mean_inlet'])
    def forward(state,alpha,inlet,weights,active):
        return observed(state,alpha,inlet,weights,active,cells,planes)
    equivalence=None
    if s['phase']=='preflight':
        original=make_wake_kernels(grid,params,chunk_steps=c['chunk_steps'],force_scale=scale)[0]
        w=jnp.ones(c['chunk_steps'])/c['chunk_steps'];active=jnp.array(c['chunk_steps'])
        old=original(state0,alpha,inlet,w,active);new=forward(state0,alpha,inlet,w,active)
        errors=[]
        for x,y in zip(jax.tree.leaves(old[0]),jax.tree.leaves(new[0])):
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-10);errors.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
        np.testing.assert_allclose(old[1],new[1],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(old[2],np.asarray(new[2])[:,:7],rtol=1e-10,atol=1e-10)
        equivalence=dict(passed=True,state_max_absolute_errors=errors,objective_difference=float(new[1]-old[1]))
        write_json(output/'observer_equivalence.json',equivalence)
    started=time.monotonic();last=[-1]
    def progress(stage,step,value,row):
        if step==c['n_steps'] or step-last[0]>=max(100,c['n_steps']//24):
            event=dict(stage=stage,step=step,total_steps=c['n_steps'],objective=value,elapsed_seconds=time.monotonic()-started)
            write_json(output/'progress.json',event);print(json.dumps(event),flush=True);last[0]=step
    result=evaluate_trajectory(alpha,inlet,state0,(forward,None,None),n_steps=c['n_steps'],chunk_steps=c['chunk_steps'],
        window_start_step=c['window_start_step'],progress=progress)
    directory=output/'evaluation-000';directory.mkdir()
    references=[np.pi*b['radius']**2 for b in c['geometry']['bodies']]
    areas=[float(fourier_shape_area(coefficients[i],base_radius=b['radius'],modes=c['geometry']['modes'],n_quadrature=4096)) for i,b in enumerate(c['geometry']['bodies'])]
    np.testing.assert_allclose(areas,references,rtol=1e-12,atol=1e-14)
    leakage=statistics(result['history'],c['window_start_step'])
    record=dict(theta=c['theta'],body_coefficients=np.asarray(coefficients).tolist(),objective=result['value'],gradient=None,
        geometric_areas=areas,reference_areas=references,diffuse_areas=np.asarray(jnp.sum(1-chi,axis=(1,2))*grid.dx*grid.dy).tolist(),
        diffuse_area=float(jnp.sum(1-q.fluid_fraction)*grid.dx*grid.dy),seconds=time.monotonic()-started,
        max_diagnostics=result['history'].max(axis=0).tolist(),paper_eligible=False,directory='evaluation-000',leakage=leakage)
    write_json(directory/'metrics.json',record)
    with (directory/'history.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time','Cd','Cl','pressure_residual','divergence_max','flux_mismatch','cfl','max_velocity',*COLUMNS])
        for i,row in enumerate(result['history']):writer.writerow([(i+1)*c['dt'],*row])
    samples=result['snapshots']
    np.savez_compressed(directory/'fields.npz',steps=np.array([s[0] for s in samples]),times=np.array([s[0]*c['dt'] for s in samples]),
        u=np.stack([s[1].u for s in samples]),v=np.stack([s[1].v for s in samples]),p=np.stack([s[1].p for s in samples]),
        phi=np.asarray(q.level_set),body_phi=np.asarray(phi),body_chi=np.asarray(chi),alpha=np.asarray(alpha))
    write_json(output/'evaluations.json',[record])
    summary=dict(schema='multibody-penalty-result-1' if registry is None else registry.RESULT_SCHEMA,mode='forward',provenance=provenance,paper_eligible=False,
        sensitivity_spec=s,value=result['value'],evaluations=[record],observer_equivalence=equivalence,
        scope='fixed-design finite-Brinkman sensitivity; no sharp-wall or gradient-convergence claim' if registry is None else registry.SCOPE)
    write_json(output/'result.json',summary);print(json.dumps(record),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
