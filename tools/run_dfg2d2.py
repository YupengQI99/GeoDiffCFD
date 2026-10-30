#!/usr/bin/env python3
"""Single-circle forward check, same material-only MAC physics and native reaction."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfg2d2_config import expand,require_conditional_half_step,EXTERNAL
from run_multibody_wake import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['forward'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();c=expand(json.loads(a.config.read_text()));s=c['dfg_spec'];out=a.output_dir
    if os.environ.get('JAX_PLATFORMS')!='cuda' or os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')!='0.70':
        raise RuntimeError('captured CUDA supervisor required')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    require_conditional_half_step(s)
    if s['case']=='C' and s['phase']=='production':
        from audit_dfg2d2 import audit
        from guard_mac_normal_tests import snapshot
        pre=EXTERNAL/f"DFG2D2_C_dt{s['dt_factor']}_preflight"
        audit(pre)
        if json.loads(pre.with_name(pre.name+'.guard').joinpath('source.json').read_text())!=snapshot():
            raise ValueError('fine preflight and production need identical clean source')
        r=json.loads((pre/'result.json').read_text())
        if r['dfg_spec']!=dict(s,phase='preflight') or not r['observer_equivalence']['passed']:
            raise ValueError('wrong fine preflight')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_cad.run_artifacts import provenance_record
    from jax_cad.differentiable.mac_channel import ChannelParams,create_channel_grid,parabolic_inlet
    from jax_cad.differentiable.wake_design import shape_fields,initial_flow,make_wake_kernels,evaluate_trajectory
    from jax_cad.differentiable.wake_leakage import core_masks,make_observed_forward,statistics,columns_for_bodies
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA required')
    provenance=provenance_record();write_json(out/'environment.json',provenance);write_json(out/'config.json',c)
    grid=create_channel_grid(**c['grid'])
    params=ChannelParams(c['viscosity'],c['dt'],include_advection=True,advection_scheme='central',
        pressure_tolerance=1e-11,pressure_atol=1e-13,pressure_maxiter=1000,pressure_preconditioner='constant-poisson-dct')
    inlet=parabolic_inlet(grid,c['mean_inlet'],dtype=jnp.float64);state0=initial_flow(grid,inlet)
    q=shape_fields(jnp.array(c['theta']),grid,center=c['center'],radius=c['radius'],alpha_max=c['alpha_max'],epsilon=c['epsilon'])
    alpha=q.momentum_resistance
    cells,planes,core=core_masks(np.asarray(q.level_set)[None],c)
    core['discrete_inlet_flux']=float(inlet.sum()*grid.dy);write_json(out/'core_definition.json',core)
    cells,planes=jnp.asarray(cells),jnp.asarray(planes)
    force_scale=2/(c['mean_inlet']**2*c['reference_diameter'])
    observed=make_observed_forward(grid,params,chunk_steps=100,force_scale=force_scale,mean_inlet=c['mean_inlet'])
    def forward(state,alpha,inlet,weights,active):return observed(state,alpha,inlet,weights,active,cells,planes)
    equivalence=None
    if s['phase']=='preflight':
        original=make_wake_kernels(grid,params,chunk_steps=100,force_scale=force_scale)[0]
        weights=jnp.ones(100)/100;active=jnp.array(100)
        old=original(state0,alpha,inlet,weights,active);new=forward(state0,alpha,inlet,weights,active)
        errors=[]
        for x,y in zip(jax.tree.leaves(old[0]),jax.tree.leaves(new[0])):
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-10);errors.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
        np.testing.assert_allclose(old[1],new[1],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(old[2],np.asarray(new[2])[:,:7],rtol=1e-10,atol=1e-10)
        equivalence=dict(passed=True,state_max_absolute_errors=errors,objective_difference=float(new[1]-old[1]))
        write_json(out/'observer_equivalence.json',equivalence)
    started=time.monotonic();last=[-1]
    def progress(stage,step,value,row):
        if step==c['n_steps'] or step-last[0]>=max(100,c['n_steps']//20):
            event=dict(step=step,total_steps=c['n_steps'],objective=value,elapsed_seconds=time.monotonic()-started)
            print(json.dumps(event),flush=True);write_json(out/'progress.json',event);last[0]=step
    r=evaluate_trajectory(alpha,inlet,state0,(forward,None,None),n_steps=c['n_steps'],chunk_steps=100,
        window_start_step=c['window_start_step'],progress=progress)
    d=out/'evaluation-000';d.mkdir()
    record=dict(theta=c['theta'],objective=r['value'],gradient=None,directory='evaluation-000',paper_eligible=False,
        geometric_area=float(np.pi*c['radius']**2),diffuse_area=float(jnp.sum(1-q.fluid_fraction)*grid.dx*grid.dy),
        seconds=time.monotonic()-started,max_diagnostics=r['history'].max(axis=0).tolist(),leakage=statistics(r['history'],c['window_start_step']))
    write_json(d/'metrics.json',record)
    with (d/'history.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time','Cd','Cl','pressure_residual','divergence_max','flux_mismatch','cfl','max_velocity',*columns_for_bodies(1)])
        for i,row in enumerate(r['history']):writer.writerow([(i+1)*c['dt'],*row])
    samples=r['snapshots']
    np.savez_compressed(d/'fields.npz',steps=np.array([v[0] for v in samples]),times=np.array([v[0]*c['dt'] for v in samples]),
        u=np.stack([v[1].u for v in samples]),v=np.stack([v[1].v for v in samples]),p=np.stack([v[1].p for v in samples]),
        phi=np.asarray(q.level_set),alpha=np.asarray(alpha),body_phi=np.asarray(q.level_set)[None],body_chi=np.asarray(q.fluid_fraction)[None])
    result=dict(schema='dfg2d2-pairing-result-1',mode='forward',dfg_spec=s,provenance=provenance,paper_eligible=False,
        value=r['value'],evaluations=[record],observer_equivalence=equivalence,
        scope='finite-Brinkman DFG-geometry diagnostic; benchmark acceptance requires independent postprocessing')
    write_json(out/'evaluations.json',[record]);write_json(out/'result.json',result)
    print(json.dumps(record),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
