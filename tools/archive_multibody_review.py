#!/usr/bin/env python3
"""Compact expert handoff for completed time-step and DFG checks (no GPU)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def archive(root,out):
    if out.exists():raise FileExistsError(out)
    timecheck=json.loads((root/'timecheck_summary.json').read_text())
    dfg=json.loads((root/'dfg2d2_summary.json').read_text())
    paths=set()
    for case in timecheck['cases']:
        for run in case['selected_runs']:
            paths.update((Path(run['original']),Path(run['dt2'])))
    paths.update(Path(e['directory']) for e in timecheck['excluded_attempts'])
    paths.update(root/name for name in ('C_initial_dt2_preflight','C_final10_dt2_preflight','DFG2D2_C_dt1_preflight'))
    for case in dfg['cases']:
        paths.update(Path(case[k]['directory']) for k in ('original_audit','selected_audit'))
        if case['case']=='C' and case['conditional_half_step_used']:paths.add(root/'DFG2D2_C_dt2_preflight')
    raw={};guards={};hardware=[];sources=set()
    for run in sorted(paths):
        guard_dir=run.with_name(run.name+'.guard')
        guard=json.loads((guard_dir/'guard.json').read_text())
        source=json.loads((guard_dir/'source.json').read_text())
        assert guard['source_unchanged'] and not source['status']
        sources.add(source['commit']);guards[run.name]=dict(guard=guard,source_commit=source['commit'])
        for relative,expected in json.loads((guard_dir/'artifacts.json').read_text()).items():
            p=root/relative;actual=digest(p)
            assert actual==expected,relative
            raw[relative]=dict(sha256=actual,bytes=p.stat().st_size)
        relative=str((guard_dir/'artifacts.json').relative_to(root))
        raw[relative]=dict(sha256=digest(root/relative),bytes=(root/relative).stat().st_size)
        with (guard_dir/'hardware.csv').open() as stream:
            hardware.extend(csv.DictReader(stream))
    for name in ('cpu_timecheck_development','cpu_timecheck_frozen','cpu_dfg2d2_development','cpu_dfg2d2_frozen'):
        d=root/name;g=json.loads((d/'guard.json').read_text())
        assert g['source_unchanged'] and g['exit_code']==0 and g['stop_reason'] is None
        guards[name]=dict(guard=g,child=json.loads((d/'child.json').read_text()))
        for p in sorted(d.iterdir()):
            if p.is_file():raw[str(p.relative_to(root))]=dict(sha256=digest(p),bytes=p.stat().st_size)
    for name in ('timecheck_summary.json','dfg2d2_summary.json'):
        p=root/name;raw[name]=dict(sha256=digest(p),bytes=p.stat().st_size)
    maxima={}
    for key in ('gpu_memory_mib','temperature_c','cpu_temperature_c','nvme_temperature_c','system_disk_loss_gib'):
        values=[float(h[key]) for h in hardware if h.get(key) not in (None,'','None')]
        maxima[key]=max(values) if values else None
    minima={}
    for key in ('disk_free_gib','system_disk_free_gib','available_ram_fraction'):
        values=[float(h[key]) for h in hardware if h.get(key) not in (None,'','None')]
        minima[key]=min(values) if values else None
    out.mkdir(parents=True)
    write_json(out/'summary.json',dict(schema='multibody-review-checks-1',paper_eligible=False,
        source_commits=sorted(sources),timecheck=timecheck,dfg2d2=dfg,
        scope='Fixed geometry forward checks only; no new optimization or manuscript edits.'))
    write_json(out/'raw_manifest.json',dict(external_root=str(root),files=raw,total_bytes=sum(v['bytes'] for v in raw.values())))
    write_json(out/'hardware_and_tests.json',dict(maxima=maxima,minima=minima,records=guards,
        external_temperature='not measured: SMART unavailable',
        incident='C_final10_dt2 interrupted when another GPU process appeared; retained, excluded, and rerun from the same initial state as C_final10_dt2_retry1.'))
    with (out/'timecheck.csv').open('w',newline='') as stream:
        columns=['case','J_initial_original','J_final_original','J_initial_dt2','J_final_dt2','gain_original_percent','gain_dt2_percent','gain_change_percentage_points','passed_registered_criteria']
        writer=csv.DictWriter(stream,columns,lineterminator='\n');writer.writeheader()
        writer.writerows({k:c[k] for k in columns} for c in timecheck['cases'])
    with (out/'dfg2d2.csv').open('w',newline='') as stream:
        columns=['case','dt','Cd_max','Cl_max','St','period_cv','maximum_half_window_drift','passed_registered_benchmark']
        writer=csv.DictWriter(stream,columns,lineterminator='\n');writer.writeheader()
        writer.writerows({k:c[k] for k in columns} for c in dfg['cases'])
    (out/'README.md').write_text('# Fixed-design time-step and DFG pairing evidence\n\n'
        'This compact bundle contains raw-derived values, preregistered gate outcomes, hardware records, and raw SHA256 references. '
        'Large histories/fields remain on the external disk specified by `raw_manifest.json`; they are not embedded here. '
        'The interrupted C final attempt is retained and explicitly excluded.\n\n'
        'Offline compact check: `.venv/bin/python -B tools/archive_multibody_review.py --check --output '+str(out.relative_to(Path(__file__).resolve().parents[1]))+'`. '
        'Add `--verify-raw` only on the machine with the external archive. No GPU is used.\n')
    (out/'.gitattributes').write_text('* text eol=lf\n')
    entries=[f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()]
    (out/'SHA256SUMS').write_text(''.join(entries))
    return dict(files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values()),hardware_maxima=maxima,minimum_reserves=minima)


def check(out,verify_raw=False):
    for line in (out/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert Path(name).name==name
        assert digest(out/name)==expected,name
    if verify_raw:
        m=json.loads((out/'raw_manifest.json').read_text());root=Path(m['external_root'])
        for name,entry in m['files'].items():assert digest(root/name)==entry['sha256'],name
    return dict(passed=True,raw_verified=verify_raw)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/media/user/external/jax_cad_multibody_sensitivity_20260926'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--verify-raw',action='store_true')
    a=p.parse_args();print(json.dumps(check(a.output,a.verify_raw) if a.check else archive(a.root,a.output.resolve()),indent=2))
