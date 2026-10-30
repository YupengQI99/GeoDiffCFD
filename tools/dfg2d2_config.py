"""Preregistered single-cylinder P/A/B/C checks; separate from multibody limits."""
from multibody_penalty_config import CASES, EXTERNAL

MODELS=dict(P=(440,82,160.,.015),**CASES)


def expand(spec):
    if (set(spec)!={'schema','paper_eligible','case','phase','dt_factor'}
        or spec['schema']!='dfg2d2-penalty-pairing-1' or spec['paper_eligible'] is not False
        or spec['case'] not in MODELS or spec['phase'] not in ('preflight','production')
        or type(spec['dt_factor']) is not int or spec['dt_factor'] not in (1,2)):
        raise ValueError('exact registered DFG 2D-2 case required')
    if spec['phase']=='preflight' and spec['case']!='C':raise ValueError('preflight is the registered fine C case')
    nx,ny,alpha,epsilon=MODELS[spec['case']];factor=spec['dt_factor']
    return dict(schema=spec['schema'],paper_eligible=False,dfg_spec=spec,
        grid=dict(nx=nx,ny=ny,lx=2.2,ly=.41),center=[.2,.2],radius=.05,theta=[0.,0.],
        geometry=dict(kind='single-circle',bodies=[dict(center=[.2,.2],radius=.05)]),
        viscosity=.001,mean_inlet=1.,reference_diameter=.1,alpha_max=alpha,epsilon=epsilon,
        dt=.00025/factor,n_steps=200 if spec['phase']=='preflight' else 40000*factor,
        window_start_step=100 if spec['phase']=='preflight' else 32000*factor,
        chunk_steps=100,advection_scheme='central',max_optimization_steps=0,
        initial_condition='same full-channel discrete-flux-normalized parabolic profile as wake_design')


def validate_config(spec):expand(spec)


def require_conditional_half_step(spec):
    if spec['dt_factor']==1:return
    from audit_dfg2d2 import audit
    from wake_window_metrics import window_metrics
    parent=EXTERNAL/f"DFG2D2_{spec['case']}_dt1"
    audit(parent)
    if window_metrics(parent)['component_screening_index']<=1:
        raise ValueError('DFG half-step authority requires original window component index >1')
