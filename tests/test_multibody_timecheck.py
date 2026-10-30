import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from multibody_penalty_config import expand
from wake_window_metrics import compare_values


def test_exact_half_step_registry():
    files=list((ROOT/'configs/multibody_penalty_timecheck').glob('*.json'))
    assert len(files)==6
    for path in files:
        s=json.loads(path.read_text());c=expand(s)
        old=expand(dict(s,schema='multibody-penalty-sensitivity-1'))
        assert c['dt']==.000125 and c['chunk_steps']==100
        for k in ('theta','geometry','grid','alpha_max','epsilon','viscosity','mean_inlet'):
            assert c[k]==old[k]
        assert (c['n_steps'],c['window_start_step'])==((200,100) if s['phase']=='preflight' else (48000,32000))
        for change in ({'case':'B'},{'dt':.0000625},{'schema':'multibody-penalty-timecheck-2'},{'phase':'dt4'},{'n_steps':96000}):
            with pytest.raises(ValueError):expand(dict(s,**change))


def test_registered_gain_and_individual_objective_gates():
    c=compare_values(10,6,10.1,6.1)
    assert c['gain_gate'] and c['individual_objective_gate']
    assert c['individual_objective_change_limit']==pytest.approx(.4)
    c=compare_values(10,6,11,6.6)
    assert c['gain_gate'] and not c['individual_objective_gate']
    c=compare_values(10,6,10,5.8)
    assert not c['gain_gate'] and c['individual_objective_gate']
