#!/usr/bin/env python3
"""Independent NumPy readback: nominal/corrected cores, force, radius, windows."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_wake_design import audit as audit_wake
from audit_multibody_penalty import discrete_inlet
from wake_window_metrics import window_metrics
from dfg2d2_metrics import analyze_history,relative_error
from wall_shift_dfg_config import expand


def audit(directory, *, config_expander=expand, steady_override=False):
    directory=Path(directory);base=audit_wake(directory)
    c=json.loads((directory/'config.json').read_text());r=json.loads((directory/'result.json').read_text())
    assert c==config_expander(c['wall_shift_spec']) and r['wall_shift_spec']==c['wall_shift_spec']
    assert r['provenance']['backend']=='gpu'
    g=c['grid'];dx=g['lx']/g['nx'];dy=g['ly']/g['ny'];inlet=discrete_inlet(c)
    x,y=np.meshgrid((np.arange(g['nx'])+.5)*dx,(np.arange(g['ny'])+.5)*dy,indexing='ij')
    nominal=np.hypot(x-.2,y-.2)-.05;effective=nominal+c['wall_shift'];s=np.clip(effective/c['epsilon'],-1,1)
    chi=.5+s*(15/16+s*s*(-5/8+s*s*3/16));alpha=c['alpha_max']*(1-chi)
    d=directory/'evaluation-000';h=np.genfromtxt(d/'history.csv',delimiter=',',names=True)
    assert h['cfl'].max()<=.8 and h['max_velocity'].max()<=8
    error=0.;scale=2/(c['mean_inlet']**2*.1)
    labels=['nominal_body1','corrected_body1']
    columns=[f'{stat}_{label}' for stat in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for label in labels]
    with np.load(d/'fields.npz') as f:
        assert all(np.isfinite(f[k]).all() for k in f.files)
        for key,expected in [('phi',effective),('phi_nominal',nominal),('body_phi',nominal[None]),('body_chi',chi[None]),('alpha',alpha)]:
            np.testing.assert_allclose(f[key],expected,rtol=1e-12,atol=1e-10 if key=='alpha' else 1e-12)
        au=np.concatenate([alpha[:1],.5*(alpha[:-1]+alpha[1:]),alpha[-1:]])
        av=np.concatenate([alpha[:,:1],.5*(alpha[:,:-1]+alpha[:,1:]),alpha[:,-1:]],axis=1)
        au[[0,-1]]*=.5;av[:,[0,-1]]*=.5
        for a,key,state in ((au,'Cd','u'),(av,'Cl','v')):
            force=np.sum(f[state]*a,axis=(1,2))*dx*dy*scale;actual=h[key][f['steps']-1]
            np.testing.assert_allclose(force,actual,rtol=1e-10,atol=1e-10)
            error=max(error,float(np.max(abs(force-actual))))
        u=f['u'];v=f['v'];speed2=((.5*(u[:,:-1]+u[:,1:]))**2+(.5*(v[:,:,:-1]+v[:,:,1:]))**2)/c['mean_inlet']**2
        values=[];counts=[];widths=[];k=round(.2/dx)
        for mask in (nominal<=-.015,effective<=-.015):
            section=mask[k-1]&mask[k];assert mask.any() and section.any()
            values.append((np.mean(np.sqrt(speed2[:,mask]),axis=1),np.mean(speed2[:,mask],axis=1),np.sum(abs(u[:,k,section]),axis=1)/inlet.sum()))
            counts.append(int(mask.sum()));widths.append(float(section.sum()*dy))
        samples=np.stack(values,axis=-1).transpose(1,0,2).reshape(len(u),6)
        saved=np.stack([h[key][f['steps']-1] for key in columns],axis=1)
        np.testing.assert_allclose(samples,saved,rtol=1e-10,atol=1e-12)
        core=json.loads((directory/'core_definition.json').read_text())
        assert core['labels']==labels and core['margin_m']==.015
        np.testing.assert_array_equal(counts,core['cell_counts']);np.testing.assert_allclose(widths,core['section_widths'],atol=1e-12)
        np.testing.assert_allclose(inlet.sum()*dy,core['discrete_inlet_flux'],rtol=1e-12)
    full=np.stack([h[key][c['window_start_step']:] for key in columns],axis=1);means=full.mean(axis=0)
    leak=r['evaluations'][0]['leakage'];assert leak['labels']==labels
    for key,actual in [('mean_speed_ratio',means[:2]),('rms_speed_ratio',np.sqrt(means[2:4])),
                       ('mean_absolute_flux_ratio',means[4:]),('max_absolute_flux_ratio',full[:,4:].max(axis=0))]:
        np.testing.assert_allclose(actual,leak[key],rtol=1e-12)
    from wall_shift_observer import measured_radius
    radius=measured_radius(chi,c)
    assert radius['max_error_m']<=dx/2
    observed=r['evaluations'][0]['material_half_fraction_radius']
    for key in ('minimum_m','maximum_m','mean_m','max_error_m'):
        np.testing.assert_allclose(radius[key],observed[key],atol=1e-12)
    metrics=window_metrics(directory);periodic=None;steady=None
    if c['wall_shift_spec']['phase']=='production':
        if steady_override or c['wall_shift_spec']['case']=='S1_2D1':
            w=h[c['window_start_step']:];mid=len(w)//2;cd=float(w['Cd'].mean())
            steady=dict(Cd_mean=cd,Cd_max=float(w['Cd'].max()),Cl_mean=float(w['Cl'].mean()),Cl_max=float(w['Cl'].max()),
                Cd_point_reference=5.57953523384,Cd_point_error_percent=100*(cd/5.57953523384-1),
                Cd_interval_error=relative_error(cd,(5.57,5.59)),
                half_window_Cd_mean_drift=float(abs(w['Cd'][mid:].mean()-w['Cd'][:mid].mean())/abs(cd)),
                report_only=True)
        else:periodic=analyze_history(h,c['window_start_step'])
    eligible=periodic is not None and periodic['passed_benchmark_metrics'] and metrics['component_screening_index']<=1
    return dict(directory=str(directory),passed_raw_audit=True,raw_audit=base,raw_field_force_max_error=error,
        core=leak,core_counts=counts,window=metrics,periodic=periodic,steady=steady,
        material_half_fraction_radius=radius,passed_registered_metrics=eligible,
        permits_stage2_metrics=c['wall_shift_spec']['case']=='S1' and eligible,
        paper_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit(a.directory);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='raw_audit'},indent=2))
