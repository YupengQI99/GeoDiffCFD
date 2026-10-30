"""Separate Stage 2-prime authority: four fixed-design sensitivities, not validation."""
import json
from multibody_penalty_config import ROOT, EXTERNAL, ANCHOR_SHA, FINAL_SHA

SCHEMA='multibody-calibrated-offset-sensitivity-20260928-1'
RESULT_SCHEMA='multibody-calibrated-offset-result-1'
CASES={'T1i':('initial',True),'T1f':('final10',True),
       'T2i':('initial',False),'T2f':('final10',False)}
LABEL='calibrated-offset model (DFG 2D-1 C_D +0.8%, DFG 2D-2 C_D,max +7.5%)'
SCOPE=LABEL+'; fixed-design sensitivity only, not benchmark validation; no gain gate; paper_eligible=false'
CPU_EVIDENCE=EXTERNAL/'cpu_multibody_offset_frozen_20260928'


def specification(case,phase):
    shape,normalized=CASES[case]
    return dict(schema=SCHEMA,paper_eligible=False,case=case,shape=shape,phase=phase,
        wall_shift=.0031020,normalize_level_set=normalized,
        anchor_sha256=ANCHOR_SHA,final_sha256=FINAL_SHA)


def expand(spec):
    if (spec.get('case') not in CASES or spec.get('phase') not in ('preflight','production')
        or spec!=specification(spec['case'],spec['phase']) or spec.get('paper_eligible') is not False
        or type(spec.get('normalize_level_set')) is not bool or type(spec.get('wall_shift')) not in (float,int)):
        raise ValueError('exact separately registered Stage 2-prime sensitivity required')
    from multibody_penalty_config import expand as baseline
    parent={k:spec[k] for k in ('paper_eligible','shape','phase','anchor_sha256','final_sha256')}
    parent.update(schema='multibody-penalty-timecheck-1',case='C')
    c=baseline(parent)
    c.update(schema=SCHEMA,paper_eligible=False,sensitivity_spec=spec,
        wall_shift=spec['wall_shift'],normalize_level_set=spec['normalize_level_set'])
    return c


def geometry_checks(c):
    from audit_wall_shift_geometry import geometry_audit,reference_checks
    a=geometry_audit(c);checks=reference_checks(c['sensitivity_spec']['shape'],a['bodies'])
    if not a['envelope']['passed'] or not all(x['nominal_interface_gate'] and x['reference_endpoint_gate'] and x['reference_count_gate'] for x in checks):
        raise ValueError('corrected geometry/reference prerequisite failed')
    a['reference_checks']=checks
    return a


def validate_config(spec):expand(spec)


def output_path(spec):
    expand(spec)
    return EXTERNAL/f"multibody_offset_{spec['case']}_{spec['phase']}_20260928"


def require_cpu_prerequisites():
    from guard_mac_normal_tests import snapshot
    guard=json.loads((CPU_EVIDENCE/'guard.json').read_text());child=json.loads((CPU_EVIDENCE/'child.json').read_text())
    if not (guard['exit_code']==0 and guard['stop_reason'] is None and guard['source_unchanged']
            and child['exit_code']==0 and child['software']['backend']=='cpu'
            and 'multibody-offset' in guard['command']):
        raise ValueError('successful bounded offset CPU suite required')
    if json.loads((CPU_EVIDENCE/'source.json').read_text())!=snapshot():
        raise ValueError('same frozen source required for CPU tests and GPU')


def require_preflight(spec):
    if spec['phase']!='production':return
    from guard_mac_normal_tests import snapshot
    from audit_multibody_offset import audit
    pre=output_path(dict(spec,phase='preflight'));audit(pre)
    if json.loads(pre.with_name(pre.name+'.guard').joinpath('source.json').read_text())!=snapshot():
        raise ValueError('same-source 200-step preflight required for each design/mapping')
    r=json.loads((pre/'result.json').read_text())
    if r['sensitivity_spec']!=dict(spec,phase='preflight') or not r['observer_equivalence']['passed']:
        raise ValueError('wrong preflight or passive observer mismatch')
