#!/usr/bin/env python3
"""Compact evidence for the separately authorized Stage 2-prime sensitivity."""
import argparse
import ast
import hashlib
import csv
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from archive_multibody_review import digest,write_json,check
from multibody_offset_config import ROOT,EXTERNAL,CPU_EVIDENCE,CASES,specification,output_path,LABEL


def pair(initial,final):
    if not np.isfinite([initial,final]).all() or initial<=0:
        raise ValueError('finite objectives and positive initial objective required')
    return dict(J_initial=initial,J_final=final,gain_percent=100*(1-final/initial))


def contrast(a,b):
    return {k:a[k]-b[k] for k in ('J_initial','J_final','gain_percent')}


def archive(out):
    from audit_multibody_offset import audit
    from audit_multibody_wake import audit as baseline_audit
    if out.exists():raise FileExistsError(out)
    raw={};guards={};hardware=[];cases=[];sources=set()
    def add(path):
        raw[str(path.relative_to(EXTERNAL))]=dict(sha256=digest(path),bytes=path.stat().st_size)
    def capture(run,collect_hardware=True):
        gd=run.with_name(run.name+'.guard')
        guard=json.loads((gd/'guard.json').read_text());source=json.loads((gd/'source.json').read_text())
        assert guard['exit_code']==0 and guard['stop_reason'] is None and guard['source_unchanged'] and not source['status']
        if collect_hardware:
            assert not guard['foreign_gpu_processes']
            sources.add(source['commit']);guards[run.name]=dict(source_commit=source['commit'],guard=guard)
            with (gd/'hardware.csv').open() as stream:hardware.extend(csv.DictReader(stream))
        for relative,sha in json.loads((gd/'artifacts.json').read_text()).items():
            path=EXTERNAL/relative;assert digest(path)==sha,relative;add(path)
        add(gd/'artifacts.json')
        return source
    for case in CASES:
        audits={}
        for phase in ('preflight','production'):
            run=output_path(specification(case,phase));capture(run)
            result=audit(run)
            recorded=EXTERNAL/f'multibody_offset_{case}_{phase}_audit_20260928.json'
            assert json.loads(json.dumps(result))==json.loads(recorded.read_text()),'independent readback changed'
            add(recorded);audits[phase]=result
        cases.append(dict(case=case,normalized=CASES[case][1],shape=CASES[case][0],**audits))
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
    baseline_rows=[]
    for name in ('C_initial_dt2','C_final10_dt2_retry1'):
        run=EXTERNAL/name;old_source=capture(run,False);a=baseline_audit(run)
        result=json.loads((run/'result.json').read_text())
        baseline_rows.append(dict(directory=str(run),objective=result['value'],source_commit=old_source['commit'],audit=a))
    baseline=pair(*(r['objective'] for r in baseline_rows))
    old_path=ROOT/'runs/multibody_review_checks_20260926/summary.json'
    old=json.loads(old_path.read_text())
    old_case=next(c for c in old['timecheck']['cases'] if c['case']=='C')
    np.testing.assert_allclose([baseline['J_initial'],baseline['J_final'],baseline['gain_percent']],
        [old_case['J_initial_dt2'],old_case['J_final_dt2'],old_case['gain_dt2_percent']],rtol=1e-13)
    by_case={c['case']:c['production'] for c in cases}
    t1=pair(by_case['T1i']['objective'],by_case['T1f']['objective'])
    t2=pair(by_case['T2i']['objective'],by_case['T2f']['objective'])
    offset=contrast(t2,baseline);normalization=contrast(t1,t2);total=contrast(t1,baseline)
    np.testing.assert_allclose([offset[k]+normalization[k] for k in total],list(total.values()),atol=1e-13)
    unchanged={}
    for path in ('jax_cad/differentiable/mac_channel.py','jax_cad/verification.py'):
        for old_source in {r['source_commit'] for r in baseline_rows}:
            changed=subprocess.check_output(['git','diff','--name-only',old_source,source['commit'],'--',path],cwd=ROOT,text=True)
            assert not changed,path
        unchanged[path]=digest(ROOT/path)
    trajectory_functions={}
    for name in ('initial_flow','make_wake_kernels','evaluate_trajectory'):
        dumps=[]
        for ref in [baseline_rows[0]['source_commit'],source['commit']]:
            code=subprocess.check_output(['git','show',ref+':jax_cad/differentiable/wake_design.py'],cwd=ROOT,text=True)
            node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name)
            dumps.append(ast.dump(node,include_attributes=False))
        assert dumps[0]==dumps[1],name
        trajectory_functions[name]=hashlib.sha256(dumps[0].encode()).hexdigest()
    for case in cases:
        new_c=json.loads((output_path(specification(case['case'],'production'))/'config.json').read_text())
        old_name='C_initial_dt2' if case['shape']=='initial' else 'C_final10_dt2_retry1'
        old_c=json.loads((EXTERNAL/old_name/'config.json').read_text())
        for key in ('grid','theta','geometry','alpha_max','epsilon','mean_inlet','viscosity','reference_diameter','dt','n_steps','window_start_step','chunk_steps','advection_scheme'):
            assert new_c[key]==old_c[key],(case['case'],key)
    summary=dict(schema='multibody-calibrated-offset-sensitivity-evidence-1',paper_eligible=False,
        complete=True,validation=False,gain_gate=None,label=LABEL,source_commits=sorted(sources),cases=cases,
        baseline=dict(**baseline,runs=baseline_rows,prior_summary_sha256=digest(old_path)),
        pairs=dict(T1=t1,T2=t2),
        contrasts=dict(offset_with_radial_definition_T2_minus_C=offset,
            normalization_at_fixed_offset_T1_minus_T2=normalization,
            total_T1_minus_C=total,gain_difference_units='percentage points'),
        level_set_definition_sensitive=abs(t1['gain_percent']-t2['gain_percent'])>3,
        level_set_sensitivity_flag_threshold_percentage_points=3,
        all_production_screening_passed=all(c['production']['screening_passed'] for c in cases),
        all_snapshot_upper_indices_le_one=all(c['production']['window']['snapshot_component_upper_index']<=1 for c in cases),
        all_snapshot_cell_center_indices_le_one=all(c['production']['window']['snapshot_cell_center_index']<=1 for c in cases),
        stability_limit='The history maximum is a velocity-component maximum. Snapshot component maxima may come from different points/times; an upper bound above one is inconclusive, not an instability verdict. No new-model dt/4 sensitivity was authorized or run.',
        unchanged_solver_and_force=unchanged,unchanged_trajectory_function_ast_sha256=trajectory_functions,baseline_physical_settings_and_fixed_designs_match=True,
        attribution_limit='Sequential contrasts only. Normalization changes transition width/profile and local offset; not a pure offset effect. No normalized-zero-shift controls or interaction attribution. Full v2 rules were not supplied.',
        prior_S1_Re100_drag_failure_unchanged=True,original_validated_stage2_started=False,S15_retried=False,
        interpretation='Fixed-design model sensitivity, not validated sharp-wall performance, geometry-gradient verification, or re-optimization.')
    out.mkdir(parents=True)
    write_json(out/'summary.json',summary)
    write_json(out/'raw_manifest.json',dict(external_root=str(EXTERNAL),files=raw,total_bytes=sum(v['bytes'] for v in raw.values())))
    write_json(out/'hardware_and_tests.json',dict(maxima=maxima,minima=minima,records=guards,
        cpu=dict(source_commit=source['commit'],guard=cpu,child=child,junit=tests),
        interruptions=0,retries=0,external_drive_temperature='not measured; SMART unavailable'))
    with (out/'pairs.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,['model','J_initial','J_final','gain_percent','change_from_C_percentage_points'],lineterminator='\n')
        writer.writeheader()
        for name,p in [('C_unshifted_radial_dt2',baseline),('T1_normalized_plus_offset',t1),('T2_radial_plus_offset',t2)]:
            writer.writerow(dict(model=name,**p,change_from_C_percentage_points=p['gain_percent']-baseline['gain_percent']))
    with (out/'diagnostics.csv').open('x',newline='') as stream:
        keys=['case','objective','component_index','snapshot_upper_index','window_max_velocity_component','window_max_cfl','body',
            'nominal_mean_speed_ratio','corrected_mean_speed_ratio','nominal_flux_ratio','corrected_flux_ratio']
        writer=csv.DictWriter(stream,keys,lineterminator='\n');writer.writeheader()
        for c in cases:
            a=c['production'];core=a['core'];w=a['window']
            for i in range(3):
                writer.writerow(dict(case=c['case'],objective=a['objective'],component_index=w['component_screening_index'],
                    snapshot_upper_index=w['snapshot_component_upper_index'],window_max_velocity_component=w['max_velocity_component_window'],
                    window_max_cfl=w['window_max_cfl'],body=i+1,nominal_mean_speed_ratio=core['mean_speed_ratio'][2*i],
                    corrected_mean_speed_ratio=core['mean_speed_ratio'][2*i+1],nominal_flux_ratio=core['mean_absolute_flux_ratio'][2*i],
                    corrected_flux_ratio=core['mean_absolute_flux_ratio'][2*i+1]))
    (out/'README.md').write_text('# Calibrated-offset multibody sensitivity, Stage 2-prime\n\n'
        +LABEL+'. This is sensitivity, not validation. No gain gate and no paper changes. All results remain paper_eligible:false.\n\n'
        'Four 48000-step forwards, each with its own 200-step preflight. Initial and converged tenth-update geometry are fixed. '
        'Objective window is (4,6] s with dt=0.000125 s. Both nominal/corrected core definitions and geometric gradient bands are preserved.\n\n'
        'Raw histories/fields remain on the external volume; hashes and compact readbacks are included. '
        'The baseline is independently read back, not taken only from a rounded 40.50% label. '
        '\n'
        'Check: .venv/bin/python -B tools/archive_multibody_offset.py --check --output runs/multibody_offset_20260928 '
        '(add --verify-raw on the external archive host).\n')
    (out/'.gitattributes').write_text('* text eol=lf\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()))
    return dict(complete=True,pairs=summary['pairs'],baseline=baseline,contrasts=summary['contrasts'],
        level_set_definition_sensitive=summary['level_set_definition_sensitive'],
        all_production_screening_passed=summary['all_production_screening_passed'],tests=tests,
        hardware_maxima=maxima,minimum_reserves=minima,raw_files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--verify-raw',action='store_true');a=p.parse_args()
    print(json.dumps(check(a.output,a.verify_raw) if a.check else archive(a.output.resolve()),indent=2,allow_nan=False))
