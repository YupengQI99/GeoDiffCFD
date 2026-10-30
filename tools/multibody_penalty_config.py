"""Exact, separately authorised A/B/C forward scope; no widening of old guards."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = 'configs/multibody_wake/converged_medium.json'
ANCHOR_SHA = '911134bb4a9ea3546865103c6d63fc17bef64a1a3744b7d402a90bb2986f3a14'
FINAL = 'runs/multibody_wake_continuation_20260921/continuation_01_retry/result.json'
FINAL_SHA = 'f44a8576610f36435c00e6c7505d0db3099abf5bdc9994a2067f50786348c89c'
CASES = {'A': (440, 82, 640., .015), 'B': (440, 82, 160., .0075),
         'C': (880, 164, 2560., .0075)}
EXTERNAL = Path('/media/user/external/jax_cad_multibody_sensitivity_20260926')


def pinned(path, sha):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('changed fixed geometry/model anchor')
    return json.loads(raw)


def expand(spec):
    timecheck = spec.get('schema') == 'multibody-penalty-timecheck-1'
    if (set(spec) != {'schema', 'paper_eligible', 'case', 'shape', 'phase',
                     'anchor_sha256', 'final_sha256'}
        or spec['schema'] not in ('multibody-penalty-sensitivity-1', 'multibody-penalty-timecheck-1')
        or spec['paper_eligible'] is not False or spec['case'] not in CASES
        or spec['shape'] not in ('initial', 'final10')
        or spec['phase'] not in ('preflight', 'production')
        or spec['anchor_sha256'] != ANCHOR_SHA or spec['final_sha256'] != FINAL_SHA):
        raise ValueError('exact authorised A/B/C specification required')
    if timecheck and spec['case'] not in ('A','C'):
        raise ValueError('half-timestep follow-up authorised only for A and C')
    if spec['phase'] == 'preflight' and spec['case'] != 'C':
        raise ValueError('short preflight is limited to C')
    c = pinned(ANCHOR, ANCHOR_SHA)
    final = pinned(FINAL, FINAL_SHA)['optimization']
    if not final['converged'] or final['total_accepted_steps'] != 10 or c['theta'] != final['coefficients']:
        raise ValueError('must use the verified tenth update, not the eighth')
    from multibody_wake_config import validate_config as original_validate
    original_validate(c)
    c = copy.deepcopy(c)
    nx, ny, alpha, epsilon = CASES[spec['case']]
    c.update(schema=spec['schema'], grid=dict(nx=nx, ny=ny, lx=2.2, ly=.41),
             alpha_max=alpha, epsilon=epsilon, theta=[0.] * 12 if spec['shape']=='initial' else final['coefficients'],
             sensitivity_spec=spec, common_core_margin=.015, max_optimization_steps=0)
    if c['dt'] != .00025 or c['chunk_steps'] != 100:
        raise ValueError('frozen timestep/checkpoint capacity changed')
    if (c['n_steps'], c['window_start_step']) != (24000,16000):
        raise ValueError('production must use the original 0–6 / 4–6 s horizon/window')
    if timecheck:
        c.update(dt=.000125,n_steps=48000,window_start_step=32000)
    if spec['phase'] == 'preflight':
        c.update(n_steps=200, window_start_step=100)
    return c


def validate_config(spec):
    expand(spec)
