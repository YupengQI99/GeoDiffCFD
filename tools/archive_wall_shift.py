#!/usr/bin/env python3
"""Read-only raw audit plus a new compact bundle; no GPU or result selection."""
import argparse
import csv
import json
from pathlib import Path
from archive_multibody_review import digest,write_json,check
from wall_shift_dfg_config import SHIFTS,EXTERNAL,ROOT,specification,output_path,CPU_EVIDENCE


def archive(out):
    from audit_wall_shift_dfg import audit
    if out.exists():raise FileExistsError(out)
    raw={};guards={};hardware=[];cases=[];sources=set()
    def add(path):
        raw[str(path.relative_to(EXTERNAL))]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for case in SHIFTS:
        stages={}
        for phase in ('preflight','production'):
            run=output_path(specification(case,phase));gd=run.with_name(run.name+'.guard')
            if not gd.exists():
                assert not run.exists(),'unexplained output without guard'
                stages[phase]=dict(status='not_started',directory=str(run));continue
            guard=json.loads((gd/'guard.json').read_text());source=json.loads((gd/'source.json').read_text())
            assert not source['status'] and guard['source_unchanged']
            sources.add(source['commit']);guards[run.name]=dict(guard=guard,source_commit=source['commit'])
            for relative,expected in json.loads((gd/'artifacts.json').read_text()).items():
                path=EXTERNAL/relative
                assert digest(path)==expected,relative
                add(path)
            add(gd/'artifacts.json')
            with (gd/'hardware.csv').open() as stream:hardware.extend(csv.DictReader(stream))
            if guard['exit_code']==0 and guard['stop_reason'] is None:
                result=audit(run)
                recorded=EXTERNAL/f'wall_shift_{case}_{phase}_audit_20260927.json'
                if recorded.exists():
                    assert result==json.loads(recorded.read_text()),'readback changed'
                    add(recorded)
                stages[phase]=dict(status='completed_raw_audit_passed',audit=result)
            else:
                stages[phase]=dict(status='interrupted_excluded',directory=str(run),
                    reason=guard['stop_reason'],exit_code=guard['exit_code'],
                    usable_production_result=False,retried=False)
        cases.append(dict(case=case,wall_shift_m=SHIFTS[case],stages=stages))
    cpu=json.loads((CPU_EVIDENCE/'guard.json').read_text());child=json.loads((CPU_EVIDENCE/'child.json').read_text())
    assert cpu['exit_code']==0 and cpu['stop_reason'] is None and cpu['source_unchanged'] and child['exit_code']==0
    cpu_source=json.loads((CPU_EVIDENCE/'source.json').read_text())
    assert not cpu_source['status'] and sources=={cpu_source['commit']}
    for p in CPU_EVIDENCE.iterdir():
        if p.is_file():add(p)
    geometry_path=EXTERNAL/'wall_shift_geometry_frozen_20260927.json'
    geometry=json.loads(geometry_path.read_text());assert geometry['geometric_gates_passed'] and geometry['source_commit'] in sources
    add(geometry_path)
    maximum={};minimum={}
    for key in ('gpu_memory_mib','temperature_c','cpu_temperature_c','nvme_temperature_c','system_disk_loss_gib'):
        values=[float(r[key]) for r in hardware if r.get(key) not in (None,'','None')]
        maximum[key]=max(values) if values else None
    for key in ('available_ram_fraction','disk_free_gib','system_disk_free_gib'):
        values=[float(r[key]) for r in hardware if r.get(key) not in (None,'','None')]
        minimum[key]=min(values) if values else None
    completed={c['case']:c['stages']['production']['audit'] for c in cases if c['stages']['production']['status']=='completed_raw_audit_passed'}
    primary=completed.get('S1');passed=primary is not None and primary['permits_stage2_metrics']
    slopes={}
    for name,left,right in [('lower_one_sided','S05','S1'),('upper_one_sided','S1','S15'),('central_secant','S05','S15')]:
        if left not in completed or right not in completed:
            slopes[name]=dict(available=False,reason='required production result unavailable');continue
        a=completed[left]['periodic']['full_window'];b=completed[right]['periodic']['full_window']
        dw=SHIFTS[right]-SHIFTS[left]
        slopes[name]=dict(available=True,left=left,right=right,delta_w_m=dw,
            dCdmax_dw_per_m=(b['Cd_max']-a['Cd_max'])/dw,dSt_dw_per_m=(b['St']-a['St'])/dw)
    baseline_path=ROOT/'runs/multibody_review_checks_20260926/summary.json'
    baseline=next(c for c in json.loads(baseline_path.read_text())['dfg2d2']['cases'] if c['case']=='C')
    summary=dict(schema='wall-shift-stage1-evidence-1',paper_eligible=False,source_commits=sorted(sources),
        all_registered_stage1_runs_completed=len(completed)==len(SHIFTS),cases=cases,
        S1_passed=passed,stage2_started=False,
        stage2_reason='S1 drag error exceeds the preregistered 5% limit' if primary and not passed else 'requires full user stage-2 protocol and passed S1',
        sensitivity_slopes=slopes,baseline_C=dict(Cd_max=baseline['Cd_max'],St=baseline['St'],errors=baseline['errors'],
            repository_path=str(baseline_path.relative_to(ROOT)),sha256=digest(baseline_path)),
        scope='Partial stage-1 forward evidence. S15 production interrupted by single-GPU protection; no automatic retry. No manuscript edits or stage-2 results.')
    out.mkdir(parents=True)
    write_json(out/'summary.json',summary)
    write_json(out/'geometry_and_calibration.json',geometry)
    write_json(out/'raw_manifest.json',dict(external_root=str(EXTERNAL),files=raw,total_bytes=sum(r['bytes'] for r in raw.values())))
    write_json(out/'hardware_and_tests.json',dict(maxima=maximum,minima=minimum,records=guards,
        cpu=dict(guard=cpu,child=child,source_commit=cpu_source['commit']),
        external_drive_temperature='not measured; SMART unavailable',
        interrupted_process_identity='not captured by the legacy guard; no attribution to a person or job'))
    with (out/'results.csv').open('x',newline='') as stream:
        columns=['case','status','wall_shift_m','Cd_max','Cl_max','St','Cd_error_percent','St_error_percent',
            'maximum_half_window_drift','component_screening_index','primary_gate_eligible']
        writer=csv.DictWriter(stream,columns,lineterminator='\n');writer.writeheader()
        for c in cases:
            row=dict(case=c['case'],status=c['stages']['production']['status'],wall_shift_m=c['wall_shift_m'])
            if c['case'] in completed and completed[c['case']]['periodic'] is not None:
                a=completed[c['case']];p=a['periodic']
                row.update({k:p['full_window'][k] for k in ('Cd_max','Cl_max','St')})
                row.update(Cd_error_percent=p['errors']['Cd_max']['signed_midpoint_percent'],
                    St_error_percent=p['errors']['St']['signed_midpoint_percent'],
                    maximum_half_window_drift=p['maximum_acceptance_drift'],component_screening_index=a['window']['component_screening_index'],
                    primary_gate_eligible=a['permits_stage2_metrics'])
            writer.writerow(row)
    (out/'README.md').write_text('# Partial wall-shift stage-1 evidence\n\n'
        'S1 and S05 production and three preflights completed. S15 production was safely interrupted when another GPU process group appeared. '
        'S1_2D1 was not launched; no automatic retries or stage-2 runs. S1 failed the preregistered drag accuracy threshold. '
        'Missing production values and the full central sensitivity slope are not imputed. All results remain paper_eligible: false.\n\n'
        'Raw histories, fields and guard records remain on the mounted external disk; raw_manifest.json includes interrupted evidence. '
        'The compact bundle can be checked without a GPU or the external disk:\n\n'
        '`.venv/bin/python -B tools/archive_wall_shift.py --check --output runs/wall_shift_stage1_20260927`\n\n'
        'Add `--verify-raw` on the archive host.\n')
    (out/'.gitattributes').write_text('* text eol=lf\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()))
    return dict(completed_production=list(completed),S1_passed=passed,hardware_maxima=maximum,minimum_reserves=minimum,
        sensitivity_slopes=slopes,raw_files=len(raw),raw_bytes=sum(r['bytes'] for r in raw.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check',action='store_true');p.add_argument('--verify-raw',action='store_true');a=p.parse_args()
    print(json.dumps(check(a.output,a.verify_raw) if a.check else archive(a.output.resolve()),indent=2,allow_nan=False))
