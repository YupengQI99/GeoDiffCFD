"""Frozen stage-1 wall shift only; stage 2 awaits the complete user protocol."""
import json
from multibody_penalty_config import ROOT, EXTERNAL

SHIFTS={'S1':.0031020,'S05':.0015510,'S15':.0046530,'S1_2D1':.0031020}
CPU_EVIDENCE=EXTERNAL/'cpu_wall_shift_frozen_20260927'


def specification(case,phase):
    return dict(schema='wall-shift-dfg-stage1-1',paper_eligible=False,case=case,phase=phase,
        wall_shift=SHIFTS[case],normalize_level_set=False)


def expand(spec):
    if (spec.get('case') not in SHIFTS or spec.get('phase') not in ('preflight','production')
        or spec!=specification(spec['case'],spec['phase'])
        or type(spec.get('normalize_level_set')) is not bool or spec.get('paper_eligible') is not False):
        raise ValueError('exact registered stage-1 wall-shift specification required')
    steady=spec['case']=='S1_2D1';pre=spec['phase']=='preflight'
    return dict(schema=spec['schema'],paper_eligible=False,wall_shift_spec=spec,
        grid=dict(nx=880,ny=164,lx=2.2,ly=.41),center=[.2,.2],radius=.05,theta=[0.,0.],
        geometry=dict(kind='single-circle',bodies=[dict(center=[.2,.2],radius=.05)]),
        viscosity=.001,mean_inlet=.2 if steady else 1.,reference_diameter=.1,
        alpha_max=2560.,epsilon=.0075,wall_shift=spec['wall_shift'],normalize_level_set=False,
        dt=.0005 if steady else .00025,n_steps=200 if pre else (32000 if steady else 40000),
        window_start_step=100 if pre else (28000 if steady else 32000),
        chunk_steps=100,advection_scheme='central',max_optimization_steps=0,
        initial_condition='same full-channel discrete-flux-normalized parabolic profile as wake_design')


def output_path(spec):
    expand(spec)
    return EXTERNAL/f"wall_shift_{spec['case']}_{spec['phase']}_20260927"


def validate_config(spec):
    expand(spec)


def require_cpu_prerequisites():
    from guard_mac_normal_tests import snapshot
    from audit_wall_shift_geometry import report
    guard=json.loads((CPU_EVIDENCE/'guard.json').read_text())
    child=json.loads((CPU_EVIDENCE/'child.json').read_text())
    if not (guard['exit_code']==0 and guard['stop_reason'] is None and guard['source_unchanged']
            and child['exit_code']==0 and child['software']['backend']=='cpu'
            and 'wall-shift-unit' in guard['command']):
        raise ValueError('successful bounded CPU prerequisite suite required')
    if json.loads((CPU_EVIDENCE/'source.json').read_text())!=snapshot():
        raise ValueError('CPU prerequisites and GPU must use identical frozen source')
    if not report()['geometric_gates_passed']:
        raise ValueError('registered geometry endpoint/count/envelope gates failed')


def require_preflight(spec):
    if spec['phase']!='production':return
    from guard_mac_normal_tests import snapshot
    from audit_wall_shift_dfg import audit
    pre=output_path(dict(spec,phase='preflight'))
    audit(pre)
    if json.loads(pre.with_name(pre.name+'.guard').joinpath('source.json').read_text())!=snapshot():
        raise ValueError('same-source successful 200-step preflight required')
    r=json.loads((pre/'result.json').read_text())
    if r['wall_shift_spec']!=dict(spec,phase='preflight') or not r['observer_equivalence']['passed']:
        raise ValueError('wrong preflight or failed passive-observer equivalence')
