#!/usr/bin/env python3
"""Independent NumPy readback, including all saved core observations and window means."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_multibody_wake import audit


def discrete_inlet(c):
    dy=c['grid']['ly']/c['grid']['ny']
    y=(np.arange(c['grid']['ny'])+.5)*dy
    raw=y*(c['grid']['ly']-y)
    return raw*(c['mean_inlet']*c['grid']['ly'])/(raw.sum()*dy)


def leakage_readback(directory, *, recorded=True):
    c=json.loads((directory/'config.json').read_text());r=json.loads((directory/'result.json').read_text())
    e=r['evaluations'][0];path=directory/e['directory']
    h=np.genfromtxt(path/'history.csv',delimiter=',',names=True)
    dx=c['grid']['lx']/c['grid']['nx'];dy=c['grid']['ly']/c['grid']['ny']
    inlet=discrete_inlet(c)
    with np.load(path/'fields.npz') as f:
        phi=f['body_phi'];cells=np.concatenate([phi<=-c['epsilon'],phi<=-.015])
        bodies=len(phi);count=2*bodies
        assert bodies in (1,3) and len(c['geometry']['bodies'])==bodies
        u=f['u'];v=f['v'];speed2=(.5*(u[:,:-1]+u[:,1:]))**2+(.5*(v[:,:,:-1]+v[:,:,1:]))**2
        values=[];counts=[];widths=[]
        for i,mask in enumerate(cells):
            assert mask.any()
            k=round(c['geometry']['bodies'][i%bodies]['center'][0]/dx)
            section=mask[k-1]&mask[k];assert section.any()
            values.append((np.mean(np.sqrt(speed2[:,mask]),axis=1)/c['mean_inlet'],
                np.mean(speed2[:,mask],axis=1)/c['mean_inlet']**2,
                np.sum(abs(u[:,k,section]),axis=1)/inlet.sum()))
            counts.append(int(mask.sum()));widths.append(float(section.sum()*dy))
        samples=np.stack(values,axis=-1).transpose(1,0,2).reshape(len(u),3*count)
        labels=[f'{core}_body{i+1}' for core in ('case','common') for i in range(bodies)]
        columns=[f'{stat}_{label}' for stat in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for label in labels]
        if recorded:
            saved=np.stack([h[key][f['steps']-1] for key in columns],axis=1)
            np.testing.assert_allclose(samples,saved,rtol=1e-10,atol=1e-12)
            full=np.stack([h[key][c['window_start_step']:] for key in columns],axis=1)
            means=full.mean(axis=0);leak=e['leakage']
            np.testing.assert_allclose(means[:count],leak['mean_speed_ratio'],rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(means[count:2*count]),leak['rms_speed_ratio'],rtol=1e-12)
            np.testing.assert_allclose(means[2*count:],leak['mean_absolute_flux_ratio'],rtol=1e-12)
            np.testing.assert_allclose(full[:,2*count:].max(axis=0),leak['max_absolute_flux_ratio'],rtol=1e-12)
            core=json.loads((directory/'core_definition.json').read_text())
            np.testing.assert_array_equal(counts,core['cell_counts'])
            np.testing.assert_allclose(widths,core['section_widths'],atol=1e-12)
            np.testing.assert_allclose(inlet.sum()*dy,core['discrete_inlet_flux'],rtol=1e-12)
        selected=f['steps']>c['window_start_step']
        late=samples[selected]
        return dict(directory=str(directory),sampled_steps=f['steps'][selected].tolist(),
            snapshot_only_mean_speed=late[:,:count].mean(axis=0).tolist(),
            snapshot_only_mean_absolute_flux=late[:,2*count:].mean(axis=0).tolist(),
            full_window=e.get('leakage'),objective=e['objective'],passed=True,paper_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',type=Path,nargs='+');p.add_argument('--baseline',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=[]
    for path in a.directories:
        base=audit(path);record=leakage_readback(path,recorded=not a.baseline);record['raw_audit']=base
        records.append(record);print(json.dumps(record),flush=True)
    a.output.write_text(json.dumps(dict(passed=True,records=records,paper_eligible=False),indent=2)+'\n')
