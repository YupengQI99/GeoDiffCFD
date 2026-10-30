#!/usr/bin/env python3
"""Assemble audited fixed-design evidence without changing manuscripts or raw runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize(root):
    baseline=read(root/'baseline_snapshot_audit.json')
    assert baseline['passed'] and len(baseline['records'])==2
    baseline=baseline['records']
    j0,jf=[r['objective'] for r in baseline]
    rows=[dict(case='baseline',nx=440,ny=82,alpha_max=160,epsilon=.015,J_initial=j0,J_final10=jf,gain_percent=100*(1-jf/j0))]
    leakage=[];sources=set();hardware=[];diagnostics=[];elapsed=0.;selected=[]
    for case in 'ABC':
        values=[]
        for index,shape in enumerate(('initial','final10')):
            path=root/f'{case}_{shape}_production';guard=path.with_name(path.name+'.guard')
            result=read(path/'result.json');config=read(path/'config.json');audit=read(root/(path.name+'_audit.json'))
            assert audit['passed'] and len(audit['records'])==1
            record=audit['records'][0]
            assert record['passed'] and record['raw_audit']['passed']
            assert record['sampled_steps']==baseline[index]['sampled_steps']
            assert result['sensitivity_spec']['case']==case and result['sensitivity_spec']['shape']==shape
            assert result['sensitivity_spec']['phase']=='production'
            assert (config['n_steps'],config['window_start_step'],config['dt'])==(24000,16000,.00025)
            assert config['theta']==([0.]*12 if index==0 else read(root/'A_final10_production/config.json')['theta'])
            assert result['paper_eligible'] is False
            values.append(result['value']);assert values[-1]==record['objective']
            e=result['evaluations'][0];leak=e['leakage'];assert leak['sample_count']==8000
            leakage.append(dict(case=case,shape=shape,full_window=leak,
                max_common_mean_speed_percent=100*max(leak['mean_speed_ratio'][3:]),
                max_common_mean_absolute_flux_percent=100*max(leak['mean_absolute_flux_ratio'][3:]),
                matching_snapshot_only_mean_speed=record['snapshot_only_mean_speed'],
                matching_snapshot_only_mean_absolute_flux=record['snapshot_only_mean_absolute_flux']))
            safety=read(guard/'guard.json');assert safety['source_unchanged'] and safety['exit_code']==0 and safety['stop_reason'] is None
            sources.add(read(guard/'source.json')['commit']);elapsed+=safety['elapsed_seconds']
            with (guard/'hardware.csv').open() as stream:hardware.extend(list(csv.DictReader(stream)))
            diagnostics.append(e['max_diagnostics'][:7]);selected.extend([path,guard,root/(path.name+'_audit.json')])
            for name,sha in read(guard/'artifacts.json').items():assert digest(root/name)==sha,name
        rows.append(dict(case=case,nx=config['grid']['nx'],ny=config['grid']['ny'],alpha_max=config['alpha_max'],
            epsilon=config['epsilon'],J_initial=values[0],J_final10=values[1],gain_percent=100*(1-values[1]/values[0])))
    assert len(sources)==1
    for shape in ('initial','final10'):
        path=root/f'C_{shape}_preflight_v2';guard=path.with_name(path.name+'.guard')
        assert read(root/(path.name+'_audit.json'))['passed']
        assert read(path/'result.json')['observer_equivalence']['passed']
        assert read(guard/'source.json')['commit'] in sources
        selected.extend([path,guard,root/(path.name+'_audit.json')])
    for name in ('cpu_checks_frozen_v2','cpu_wake_regression_v2'):
        path=root/name;g=read(path/'guard.json')
        assert g['exit_code']==0 and g['stop_reason'] is None and g['source_unchanged']
        assert read(path/'source.json')['commit'] in sources
        selected.append(path)
    selected.append(root/'baseline_snapshot_audit.json')
    peaks={key:max(float(h[key]) for h in hardware if h[key]) for key in
        ('gpu_memory_mib','temperature_c','cpu_temperature_c','nvme_temperature_c','power_w','system_disk_loss_gib')}
    minima={key:min(float(h[key]) for h in hardware if h[key]) for key in
        ('disk_free_gib','available_ram_fraction','system_disk_free_gib')}
    summary=dict(schema='multibody-penalty-summary-1',paper_eligible=False,all_six_forwards_and_audits_passed=True,
        experiment_source_commit=next(iter(sources)),objective_rows=rows,leakage=leakage,baseline_snapshot_only=baseline,
        production_elapsed_seconds=elapsed,production_hardware_peaks=peaks,production_minimum_reserves=minima,
        maximum_original_diagnostics=[max(d[i] for d in diagnostics) for i in range(7)],
        interpretation='improvement retained in tested finite-Brinkman settings; not alpha independence, sharp-wall convergence, or reoptimization',
        leakage_definition='common radial core phi_i<=-0.015; per-body mean speed/Umean and absolute centre-section flux/discrete inlet flux',
        baseline_leakage_caution='snapshot means only; 24 identical, nonuniform saved times, not the full-window time average')
    files=[]
    for path in selected:files.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    summary['selected_artifact_count']=len(manifest);summary['selected_artifact_bytes']=sum(r['bytes'] for r in manifest.values())
    return summary,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    summary,manifest=summarize(a.root)
    a.output_dir.mkdir()
    for name,data in (('summary.json',summary),('artifacts_sha256.json',manifest)):
        (a.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (a.output_dir/'objectives.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary['objective_rows'][0]));writer.writeheader();writer.writerows(summary['objective_rows'])
    print(json.dumps(summary,indent=2))
