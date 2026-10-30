#!/usr/bin/env python3
"""Paired benchmark summary: original time step, or its conditionally required half step."""
import argparse
import json
from pathlib import Path
from audit_dfg2d2 import audit


def summarize(root):
    cases=[]
    for case in ('P','A','B','C'):
        original=audit(root/f'DFG2D2_{case}_dt1')
        selected=original
        if original['half_step_required']:
            selected=audit(root/f'DFG2D2_{case}_dt2')
        c=json.loads((Path(selected['directory'])/'config.json').read_text())
        periodic=selected['periodic']
        cases.append(dict(case=case,grid=c['grid'],alpha_max=c['alpha_max'],epsilon=c['epsilon'],dt=c['dt'],
            selected_directory=selected['directory'],conditional_half_step_used=original['half_step_required'],
            Cd_max=periodic['full_window']['Cd_max'],Cl_max=periodic['full_window']['Cl_max'],
            St=periodic['full_window']['St'],period_cv=periodic['full_window']['period_cv'],
            maximum_half_window_drift=periodic['maximum_acceptance_drift'],errors=periodic['errors'],
            window=selected['window'],core=selected['core']['full_window'],
            passed_registered_benchmark=selected['passed_registered_benchmark'],
            original_audit=original,selected_audit=selected))
    return dict(schema='dfg2d2-pairing-summary-1',paper_eligible=False,cases=cases,
        C_passed=cases[-1]['passed_registered_benchmark'],
        scope='Forward single-cylinder accuracy at the registered settings only; not proof of three-body sharp-interface, gradient, or grid convergence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=summarize(a.root);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for c in r['cases']:print(json.dumps({k:v for k,v in c.items() if k not in ('original_audit','selected_audit','core','window','errors')}))
