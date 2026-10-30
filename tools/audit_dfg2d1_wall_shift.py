#!/usr/bin/env python3
"""Independent raw readback and separately frozen 2D-1 reporting protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_wall_shift_dfg import audit as raw_audit
from dfg2d1_wall_shift_config import expand
from dfg2d1_wall_shift_metrics import analyze_history


def audit(directory):
    directory=Path(directory);c=json.loads((directory/'config.json').read_text())
    r=raw_audit(directory,config_expander=expand,steady_override=True)
    result=json.loads((directory/'result.json').read_text())
    assert result['schema']=='dfg2d1-wall-shift-corroboration-result-1'
    if c['wall_shift_spec']['phase']=='production':
        h=np.genfromtxt(directory/'evaluation-000/history.csv',delimiter=',',names=True)
        r['steady']=analyze_history(h)
        np.testing.assert_allclose(r['steady']['quantities']['Cd']['mean_14_16'],result['value'],rtol=1e-12)
    r.update(schema='dfg2d1-wall-shift-audit-1',case=c['wall_shift_spec']['case'],phase=c['wall_shift_spec']['phase'],
        force_scale=2/(c['mean_inlet']**2*c['reference_diameter']),nominal_Re=c['mean_inlet']*.1/c['viscosity'],
        corroboration_only=True,passed_registered_metrics=False,permits_stage2_metrics=False,paper_eligible=False)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit(a.directory);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='raw_audit'},indent=2))
