#!/usr/bin/env python3
"""Independent raw readback and preregistered A/C half-step comparisons."""
import argparse
import json
from pathlib import Path
from audit_multibody_wake import audit
from audit_multibody_penalty import leakage_readback
from wake_window_metrics import window_metrics,compare_values


def summarize(root, *, c_final_run='C_final10_dt2'):
    if Path(c_final_run).name!=c_final_run or not c_final_run.startswith('C_final10_dt2'):
        raise ValueError('explicit C final half-step run basename required')
    cases=[]
    for case in ('A','C'):
        old=[];new=[];windows=[];leakage=[];checks=[];selected=[]
        for shape in ('initial','final10'):
            prior=root/f'{case}_{shape}_production';current=root/f'{case}_{shape}_dt2'
            if case=='C' and shape=='final10':current=root/c_final_run
            selected.append(dict(shape=shape,original=str(prior),dt2=str(current)))
            checks.append(audit(current));leakage.append(leakage_readback(current))
            p=json.loads((prior/'config.json').read_text());c=json.loads((current/'config.json').read_text())
            for key in ('geometry','theta','grid','viscosity','mean_inlet','reference_diameter','epsilon','alpha_max','advection_scheme','chunk_steps'):
                assert p[key]==c[key],key
            assert c['dt']*2==p['dt'] and c['n_steps']==48000 and c['window_start_step']==32000
            old.append(json.loads((prior/'result.json').read_text())['value'])
            new.append(json.loads((current/'result.json').read_text())['value'])
            windows.append(dict(shape=shape,original=window_metrics(prior),dt2=window_metrics(current)))
        result=compare_values(*old,*new)
        result.update(case=case,windows=windows,leakage=leakage,raw_checks=checks,selected_runs=selected)
        result['component_index_gate']=all(w['dt2']['component_screening_index']<=1 for w in windows)
        result['passed_registered_criteria']=result['component_index_gate'] and result['gain_gate'] and result['individual_objective_gate']
        cases.append(result)
    excluded=[]
    failed=root/'C_final10_dt2'
    if c_final_run!=failed.name and failed.exists():
        guard=json.loads(failed.with_name(failed.name+'.guard').joinpath('guard.json').read_text())
        assert guard['stop_reason'] is not None or guard['exit_code']!=0
        excluded.append(dict(directory=str(failed),reason='hardware-aborted incomplete attempt; retry explicitly selected',guard=guard))
    return dict(schema='multibody-timecheck-summary-1',paper_eligible=False,cases=cases,excluded_attempts=excluded,
        all_passed=all(c['passed_registered_criteria'] for c in cases),
        limits='No claim of general numerical stability or time/grid convergence from one halving.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--c-final-run',default='C_final10_dt2',help='explicit basename, never select by objective')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.root,c_final_run=a.c_final_run);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for c in result['cases']:
        print(json.dumps({k:v for k,v in c.items() if k not in ('windows','leakage','raw_checks')}))
