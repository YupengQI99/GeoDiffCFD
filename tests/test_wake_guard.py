import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("wakeguard",ROOT/"tools/guard_wake_design.py")
guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)


def test_scope_is_bounded_and_development_only():
    c=json.loads((ROOT/"configs/wake_design/smoke.json").read_text())
    guard.validate_config(c)
    for changes in (dict(paper_eligible=True),dict(grid=dict(nx=880,ny=164)),
                    dict(n_steps=50000),dict(chunk_steps=1000),dict(theta=[.3,0]),
                    dict(window_start_step=c["n_steps"]),dict(max_optimization_steps=7)):
        with pytest.raises(ValueError): guard.validate_config(dict(c,**changes))


def test_hardware_policy_is_not_weakened():
    values=dict(disk_free_gib=44,disk_used_fraction=.98,available_ram_fraction=.6,
                gpu_memory_mib=17000,temperature_c=60,cpu_temperature_c=70,nvme_temperature_c=50)
    assert guard.rejection(values,startup=True,low_disk_exception=True) is None
    for change in (dict(disk_free_gib=29),dict(gpu_memory_mib=22001),dict(temperature_c=82),
                   dict(cpu_temperature_c=90),dict(nvme_temperature_c=75),dict(available_ram_fraction=.09)):
        assert guard.rejection(dict(values,**change),startup=False,low_disk_exception=True)
