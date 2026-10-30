#!/usr/bin/env python3
"""Independent circle, native-force, passive-core and periodic-window readback."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_wake_design import audit as audit_wake
from audit_multibody_penalty import leakage_readback
from wake_window_metrics import window_metrics
from dfg2d2_metrics import analyze_history
from dfg2d2_config import expand


def audit(directory):
    directory=Path(directory);base=audit_wake(directory)
    c=json.loads((directory/'config.json').read_text());r=json.loads((directory/'result.json').read_text())
    assert c==expand(c['dfg_spec']) and r['dfg_spec']==c['dfg_spec']
    assert r['provenance']['backend']=='gpu'
    g=c['grid'];dx=g['lx']/g['nx'];dy=g['ly']/g['ny']
    x,y=np.meshgrid((np.arange(g['nx'])+.5)*dx,(np.arange(g['ny'])+.5)*dy,indexing='ij')
    phi=np.hypot(x-.2,y-.2)-.05;s=np.clip(phi/c['epsilon'],-1,1)
    chi=.5+s*(15/16+s*s*(-5/8+s*s*3/16));alpha=c['alpha_max']*(1-chi)
    d=directory/'evaluation-000';h=np.genfromtxt(d/'history.csv',delimiter=',',names=True)
    assert h['cfl'].max()<=.8 and h['max_velocity'].max()<=8
    error=0.
    with np.load(d/'fields.npz') as f:
        assert all(np.isfinite(f[k]).all() for k in f.files)
        np.testing.assert_allclose(f['phi'],phi,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(f['body_phi'],phi[None],rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(f['body_chi'],chi[None],atol=1e-12)
        np.testing.assert_allclose(f['alpha'],alpha,rtol=1e-12,atol=1e-10)
        au=np.concatenate([alpha[:1],.5*(alpha[:-1]+alpha[1:]),alpha[-1:]])
        av=np.concatenate([alpha[:,:1],.5*(alpha[:,:-1]+alpha[:,1:]),alpha[:,-1:]],axis=1)
        au[[0,-1]]*=.5;av[:,[0,-1]]*=.5
        for a,key,state in ((au,'Cd','u'),(av,'Cl','v')):
            force=np.sum(f[state]*a,axis=(1,2))*dx*dy*20
            actual=h[key][f['steps']-1]
            np.testing.assert_allclose(force,actual,rtol=1e-10,atol=1e-10)
            error=max(error,float(np.max(abs(force-actual))))
    core=leakage_readback(directory)
    metrics=window_metrics(directory)
    periodic=None if c['dfg_spec']['phase']=='preflight' else analyze_history(h,c['window_start_step'])
    eligible=periodic is not None and periodic['passed_benchmark_metrics'] and metrics['component_screening_index']<=1
    return dict(directory=str(directory),passed=True,raw_audit=base,raw_field_force_max_error=error,
        core=core,window=metrics,periodic=periodic,passed_registered_benchmark=eligible,
        half_step_required=c['dfg_spec']['phase']=='production' and c['dfg_spec']['dt_factor']==1 and metrics['component_screening_index']>1,
        paper_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit(a.directory);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('core','raw_audit')},indent=2))
