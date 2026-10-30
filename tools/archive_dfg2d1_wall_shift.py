#!/usr/bin/env python3
"""Compact, raw-verified steady corroboration; never launches or selects CFD."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from archive_multibody_review import digest,write_json,check
from dfg2d1_wall_shift_config import ROOT,EXTERNAL,CPU_EVIDENCE,SHIFTS,specification,output_path


def archive(out):
    from audit_dfg2d1_wall_shift import audit
    if out.exists():raise FileExistsError(out)
    raw={};guards={};hardware=[];cases=[];sources=set()
    def add(path):raw[str(path.relative_to(EXTERNAL))]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for case,w in SHIFTS.items():
        audits={}
        for phase in ('preflight','production'):
            run=output_path(specification(case,phase));gd=run.with_name(run.name+'.guard')
            guard=json.loads((gd/'guard.json').read_text());source=json.loads((gd/'source.json').read_text())
            assert guard['exit_code']==0 and guard['stop_reason'] is None and guard['source_unchanged'] and not source['status']
            assert not guard['foreign_gpu_processes']
            sources.add(source['commit']);guards[run.name]=dict(source_commit=source['commit'],guard=guard)
            for relative,sha in json.loads((gd/'artifacts.json').read_text()).items():
                path=EXTERNAL/relative;assert digest(path)==sha,relative;add(path)
            add(gd/'artifacts.json')
            with (gd/'hardware.csv').open() as stream:hardware.extend(csv.DictReader(stream))
            result=audit(run)
            recorded=EXTERNAL/f'dfg2d1_wall_shift_{case}_{phase}_audit_20260928.json'
            assert result==json.loads(recorded.read_text()),'independent readback changed'
            add(recorded);audits[phase]=result
        cases.append(dict(case=case,wall_shift_m=w,preflight=audits['preflight'],production=audits['production']))
    cpu=json.loads((CPU_EVIDENCE/'guard.json').read_text());child=json.loads((CPU_EVIDENCE/'child.json').read_text())
    source=json.loads((CPU_EVIDENCE/'source.json').read_text())
    assert cpu['exit_code']==0 and cpu['stop_reason'] is None and cpu['source_unchanged'] and child['exit_code']==0
    assert not source['status'] and sources=={source['commit']}
    for p in CPU_EVIDENCE.iterdir():
        if p.is_file():add(p)
    suites=list(ET.parse(CPU_EVIDENCE/'junit.xml').getroot().iter('testsuite'))
    tests={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','errors','failures','skipped')}
    assert tests['errors']==tests['failures']==0
    maxima={};minima={}
    for key in ('gpu_memory_mib','temperature_c','cpu_temperature_c','nvme_temperature_c','system_disk_loss_gib'):
        values=[float(r[key]) for r in hardware if r.get(key) not in (None,'','None')]
        maxima[key]=max(values) if values else None
    for key in ('available_ram_fraction','disk_free_gib','system_disk_free_gib'):
        values=[float(r[key]) for r in hardware if r.get(key) not in (None,'','None')]
        minima[key]=min(values) if values else None
    shifted=cases[0]['production'];control=cases[1]['production']
    assert cases[0]['case']=='S1_2D1' and cases[1]['case']=='W0_2D1'
    for key in ('mean_speed_ratio','rms_speed_ratio','mean_absolute_flux_ratio','max_absolute_flux_ratio'):
        assert control['core'][key][0]==control['core'][key][1],'zero-shift cores differ'
    comparison={}
    for key in ('Cd','Cl'):
        s=shifted['steady']['quantities'][key];z=control['steady']['quantities'][key]
        comparison[key]=dict(shifted=s['mean_14_16'],unshifted=z['mean_14_16'],difference=s['mean_14_16']-z['mean_14_16'],
            shifted_point_error_percent=s['signed_point_error_percent'],unshifted_point_error_percent=z['signed_point_error_percent'],
            error_change_percentage_points=s['signed_point_error_percent']-z['signed_point_error_percent'])
    physics_paths=['jax_cad/differentiable/mac_channel.py','jax_cad/differentiable/wake_design.py',
        'jax_cad/differentiable/wake_leakage.py','jax_cad/differentiable/wall_shift.py','jax_cad/verification.py']
    same_physics={}
    run_commit=next(iter(sources))
    for name in physics_paths:
        changed=subprocess.check_output(['git','diff','--name-only','04185bad95b37ec513ab0099e8aeb81f30a943ec',run_commit,'--',name],cwd=ROOT,text=True)
        assert not changed,name
        same_physics[name]=digest(ROOT/name)
    summary=dict(schema='dfg2d1-wall-shift-corroboration-evidence-1',paper_eligible=False,corroboration_only=True,
        complete=True,source_commits=sorted(sources),cases=cases,same_grid_comparison=comparison,
        benchmark_gate_applicable=False,stage2_started=False,S15_retried=False,
        prior_S1_Re100_drag_failure_unchanged=True,
        prediction=dict(Cd_signed_error_percent_range=[2.,4.],source='user-supplied 440x82 CPU pre-run; not rerun here',
            observed_Cd_signed_error_percent=comparison['Cd']['shifted_point_error_percent'],
            within_predicted_range=2<=comparison['Cd']['shifted_point_error_percent']<=4),
        physics_unchanged_from_S1_04185ba=same_physics,
        interpretation='Only Cd drift<1e-4 permits citation as the requested steady value. No accuracy gate or claim of Re100, geometry-gradient, grid or sharp-interface convergence.')
    out.mkdir(parents=True)
    write_json(out/'summary.json',summary)
    write_json(out/'raw_manifest.json',dict(external_root=str(EXTERNAL),files=raw,total_bytes=sum(v['bytes'] for v in raw.values())))
    write_json(out/'hardware_and_tests.json',dict(maxima=maxima,minima=minima,records=guards,
        cpu=dict(source_commit=source['commit'],guard=cpu,child=child,junit=tests),
        interruptions=0,retries=0,external_drive_temperature='not measured; SMART unavailable'))
    with (out/'results.csv').open('x',newline='') as stream:
        keys=['case','wall_shift_m','Cd_mean','Cd_error_percent','Cd_drift','Cl_mean','Cl_error_percent','Cl_drift',
            'Cd_citation_drift_passed','window_component_index','snapshot_component_upper_index',
            'nominal_core_mean_speed_ratio','corrected_core_mean_speed_ratio','nominal_core_flux_ratio','corrected_core_flux_ratio']
        writer=csv.DictWriter(stream,keys,lineterminator='\n');writer.writeheader()
        for c in cases:
            a=c['production'];q=a['steady']['quantities'];core=a['core'];window=a['window']
            row=dict(case=c['case'],wall_shift_m=c['wall_shift_m'],Cd_citation_drift_passed=a['steady']['Cd_citation_drift_passed'],
                window_component_index=window['component_screening_index'],snapshot_component_upper_index=window['snapshot_component_upper_index'],
                nominal_core_mean_speed_ratio=core['mean_speed_ratio'][0],corrected_core_mean_speed_ratio=core['mean_speed_ratio'][1],
                nominal_core_flux_ratio=core['mean_absolute_flux_ratio'][0],corrected_core_flux_ratio=core['mean_absolute_flux_ratio'][1])
            for key in ('Cd','Cl'):
                row.update({key+'_mean':q[key]['mean_14_16'],key+'_error_percent':q[key]['signed_point_error_percent'],key+'_drift':q[key]['relative_half_window_drift']})
            writer.writerow(row)
    (out/'README.md').write_text('# DFG 2D-1 wall-shift corroboration, 2026-09-28\n\n'
        'Both preregistered steady forwards and their200-step preflights completed. This is independent corroboration, not stage2 or a retry of S1.5. '
        'Means use(14,16]s; both half means, strict Cd drift citation flags, point/interval errors, two core definitions and stability screens are preserved. '
        'All results remain paper_eligible:false. The Re100 S1 drag-accuracy failure is unchanged.\n\n'
        'Large raw fields/histories stay on the external disk specified in raw_manifest.json. '
        'The legacy audit field passed_registered_metrics:false is not an accuracy verdict here; benchmark_gate_applicable:false is explicit.\n\n'
        'Check without GPU: `.venv/bin/python -B tools/archive_dfg2d1_wall_shift.py --check --output runs/dfg2d1_wall_shift_20260928`. '
        'Add --verify-raw on the external archive host.\n')
    (out/'.gitattributes').write_text('* text eol=lf\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()))
    return dict(complete=True,same_grid_comparison=comparison,tests=tests,hardware_maxima=maxima,minimum_reserves=minima,
        raw_files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--verify-raw',action='store_true');a=p.parse_args()
    print(json.dumps(check(a.output,a.verify_raw) if a.check else archive(a.output.resolve()),indent=2,allow_nan=False))
