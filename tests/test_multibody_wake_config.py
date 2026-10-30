import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("multibody_config",ROOT/"tools/multibody_wake_config.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_bounded_development_scope():
    c=json.loads((ROOT/"configs/multibody_wake/coarse_initial.json").read_text()); module.validate_config(c)
    for changes in (dict(paper_eligible=True),dict(theta=[0.]*11),dict(theta=[1.1]*12),dict(n_steps=50000),
                    dict(dt=.001),dict(max_optimization_steps=9),dict(epsilon=.001),
                    dict(grid=dict(nx=880,ny=164,lx=2.2,ly=.41))):
        with pytest.raises(ValueError): module.validate_config(dict(c,**changes))


def test_whole_box_separation_certificate():
    c=json.loads((ROOT/"configs/multibody_wake/coarse_initial.json").read_text())
    report=module.envelope_report(c)
    assert report["minimum_support_wall_gap"]>.02 and report["minimum_support_pair_gap"]>.04
    for point in ([.35,.08],[.69,.275]):
        bad=copy.deepcopy(c); bad["geometry"]["bodies"][0]["center"]=point
        with pytest.raises(ValueError): module.validate_config(bad)
