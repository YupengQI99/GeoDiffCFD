"""Independent geometry/force readback and both core conventions for Stage 2-prime."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_multibody_wake import audit as base_audit
from audit_multibody_penalty import discrete_inlet
from wake_window_metrics import window_metrics
from multibody_offset_config import expand,geometry_checks,RESULT_SCHEMA,LABEL


def audit(directory):
    directory=Path(directory)
    c=json.loads((directory/'config.json').read_text());r=json.loads((directory/'result.json').read_text())
    assert c==expand(c['sensitivity_spec']) and r['sensitivity_spec']==c['sensitivity_spec']
    assert r['schema']==RESULT_SCHEMA and r['mode']=='forward' and len(r['evaluations'])==1
    assert r['evaluations'][0]['gradient'] is None
    base=base_audit(directory,calibrated_offset=True);geom=geometry_checks(c)
    g=c['grid'];dx=g['lx']/g['nx'];dy=g['ly']/g['ny'];inlet=discrete_inlet(c)
    d=directory/'evaluation-000';h=np.genfromtxt(d/'history.csv',delimiter=',',names=True)
    assert h['cfl'].max()<=.8 and h['max_velocity'].max()<=8
    np.testing.assert_allclose(r['value'],h['Cd'][c['window_start_step']:].mean(),rtol=1e-12)
    labels=[f'{kind}_body{i+1}' for i in range(3) for kind in ('nominal','corrected')]
    columns=[f'{stat}_{label}' for stat in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for label in labels]
    x,y=np.meshgrid((np.arange(g['nx'])+.5)*dx,(np.arange(g['ny'])+.5)*dy,indexing='ij')
    counts=[];widths=[];values=[]
    with np.load(d/'fields.npz') as f:
        u=f['u'];v=f['v'];coeff=np.asarray(r['evaluations'][0]['body_coefficients'])
        speed2=((.5*(u[:,:-1]+u[:,1:]))**2+(.5*(v[:,:,:-1]+v[:,:,1:]))**2)/c['mean_inlet']**2
        for i,b in enumerate(c['geometry']['bodies']):
            angle=np.arctan2(y-b['center'][1],x-b['center'][0]);nominal=f['body_phi'][i]
            fp=sum(m*(-coeff[i,k]*np.sin(m*angle)+coeff[i,k+3]*np.cos(m*angle)) for k,m in enumerate(c['geometry']['modes']))
            corrected=nominal/np.sqrt(1+fp**2)+c['wall_shift'] if c['normalize_level_set'] else nominal+c['wall_shift']
            k=round(b['center'][0]/dx)
            for mask in (nominal<=-.015,corrected<=-.015):
                section=mask[k-1]&mask[k];assert mask.any() and section.any()
                values.append((np.mean(np.sqrt(speed2[:,mask]),axis=1),np.mean(speed2[:,mask],axis=1),np.sum(abs(u[:,k,section]),axis=1)/inlet.sum()))
                counts.append(int(mask.sum()));widths.append(float(section.sum()*dy))
        samples=np.stack(values,axis=-1).transpose(1,0,2).reshape(len(u),18)
        saved=np.stack([h[key][f['steps']-1] for key in columns],axis=1)
        np.testing.assert_allclose(samples,saved,rtol=1e-10,atol=1e-12)
    core=json.loads((directory/'core_definition.json').read_text())
    assert core['labels']==labels and core['margin_m']==.015
    np.testing.assert_array_equal(counts,core['cell_counts'])
    np.testing.assert_allclose(widths,core['section_widths'],atol=1e-12)
    np.testing.assert_allclose(inlet.sum()*dy,core['discrete_inlet_flux'],rtol=1e-12)
    full=np.stack([h[key][c['window_start_step']:] for key in columns],axis=1);means=full.mean(axis=0)
    leak=r['evaluations'][0]['leakage'];assert leak['labels']==labels and leak['sample_count']==len(full)
    for key,actual in [('mean_speed_ratio',means[:6]),('rms_speed_ratio',np.sqrt(means[6:12])),
                       ('mean_absolute_flux_ratio',means[12:]),('max_absolute_flux_ratio',full[:,12:].max(axis=0))]:
        np.testing.assert_allclose(actual,leak[key],rtol=1e-12)
    metrics=window_metrics(directory)
    return dict(directory=str(directory),passed_raw_audit=True,raw_audit=base,geometry=geom,
        core=leak,core_counts=counts,window=metrics,objective=r['value'],
        screening_passed=metrics['component_screening_index']<=1,
        label=LABEL,validation=False,gain_gate=None,paper_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit(a.directory);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('directory','passed_raw_audit','objective','window','screening_passed')},indent=2))
