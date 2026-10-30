#!/usr/bin/env python3
"""Single-process captured GPU wake pilot; reuse existing hardware stop policy."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from guard_cylinder_run import sample,rejection,gpu_pids,interrupt_group
from guard_mac_normal_tests import snapshot


def validate_config(c):
    if 'wall_shift' in c or 'normalize_level_set' in c:
        raise ValueError('wall corrections require their own registered entry and passed CPU prerequisites')
    if c.get("schema")!="wake-design-development-1" or c.get("paper_eligible") is not False:
        raise ValueError("explicit development-only wake configuration required")
    g=c["grid"]
    if not (8<=g["nx"]<=440 and 5<=g["ny"]<=82):
        raise ValueError("wake pilot limited to 440x82; no reused fine authority")
    if not (1<=c["n_steps"]<=48000 and 1<=c["chunk_steps"]<=200):
        raise ValueError("bounded physical horizon and checkpoint capacity required")
    if not (0<=c["window_start_step"]<c["n_steps"] and 0<c["dt"]<=.002):
        raise ValueError("invalid time window/time step")
    if c["advection_scheme"]!="central" or c["alpha_max"]<=0 or c["epsilon"]<=0:
        raise ValueError("invalid frozen model")
    if len(c["theta"])!=2 or any(abs(t)>.25 for t in c["theta"]):
        raise ValueError("shape coefficients exceed pilot box")
    if not 0<=c.get("max_optimization_steps",0)<=6:
        raise ValueError("bounded optimisation required")


def main(*, driver="run_wake_design.py", validator=validate_config, storage_policy=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("audit","forward","gradient","optimize"),required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--timeout-seconds",type=float,default=900)
    p.add_argument("--user-approved-low-disk",action="store_true")
    a=p.parse_args()
    if not 0<a.timeout_seconds<=1800: p.error("bounded segments <=1800 seconds")
    config=a.config.resolve(); c=json.loads(config.read_text()); validator(c)
    before=snapshot()
    if before["status"]: raise RuntimeError("freeze all source/configuration before GPU launch")
    subprocess.run(["git","ls-files","--error-unmatch",str(config)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    output=a.output_dir.resolve()
    if output.exists(): raise FileExistsError("new output directory required")
    if output.is_relative_to(ROOT): raise ValueError("external development output directory required")
    guard=output.with_name(output.name+".guard")
    if guard.exists(): raise FileExistsError(guard)
    sample_now = sample if storage_policy is None else storage_policy.sample
    reject_now = rejection if storage_policy is None else storage_policy.rejection
    storage_env = {} if storage_policy is None else storage_policy.prepare(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(storage_env)
    env.update(CUDA_VISIBLE_DEVICES="0",JAX_ENABLE_X64="true",JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="true",XLA_PYTHON_CLIENT_MEM_FRACTION="0.70",
               XLA_FLAGS="--xla_backend_optimization_level=0",OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               MPLBACKEND="Agg")
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR",None)
    command=[sys.executable,str(ROOT/"tools"/driver),"--mode",a.mode,
             "--config",str(config),"--output-dir",str(output)]
    proc=None; reason=None; foreign_gpu_processes=[]; started=time.monotonic()
    def interrupted(signum,frame): raise InterruptedError(f"supervisor signal {signum}")
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,interrupted)
    with (ROOT/"runs/.cylinder-gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=sample_now(output.parent)
        problem=reject_now(initial,startup=True,low_disk_exception=a.user_approved_low_disk)
        if problem or gpu_pids(): raise RuntimeError(problem or "GPU is occupied")
        guard.mkdir(); output.mkdir()
        (guard/"source.json").write_text(json.dumps(before,indent=2)+"\n")
        (guard/"config.json").write_bytes(config.read_bytes())
        peaks=dict(initial)
        try:
            with (guard/"hardware.csv").open("x",newline="") as hw, (guard/"stdout.log").open("xb") as out, (guard/"stderr.log").open("xb") as err:
                writer=csv.DictWriter(hw,list(initial)); writer.writeheader(); writer.writerow(initial); hw.flush()
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
                while proc.poll() is None:
                    s=sample_now(output.parent); writer.writerow(s); hw.flush()
                    for key in ("gpu_memory_mib","temperature_c","cpu_temperature_c","nvme_temperature_c","power_w","disk_used_fraction"):
                        if s[key] is not None: peaks[key]=max(peaks[key] or 0,s[key])
                    for key in ("disk_free_gib","available_ram_fraction"): peaks[key]=min(peaks[key],s[key])
                    reason=reject_now(s,startup=False,low_disk_exception=a.user_approved_low_disk)
                    for pid in gpu_pids():
                        try:
                            pgid=os.getpgid(pid)
                            if pgid!=proc.pid:
                                reason="another GPU process appeared"
                                foreign_gpu_processes.append(dict(pid=pid,process_group=pgid,unix_time=s['unix_time']))
                        except ProcessLookupError: pass
                    if time.monotonic()-started>a.timeout_seconds: reason="bounded wall timeout"
                    if sum(x.stat().st_size for x in output.rglob("*") if x.is_file())>512*1024**2:
                        reason="512 MiB per-run output budget"
                    if reason: interrupt_group(proc); break
                    time.sleep(2)
        except BaseException as e:
            reason=f"{type(e).__name__}: {e}"
            if proc is not None: interrupt_group(proc)
            raise
        finally:
            after=snapshot()
            if after!=before: reason=reason or "source changed during run"
            record=dict(schema="wake-hardware-guard-1",command=command,initial=initial,
                        peak_temperatures_memory_and_minimum_reserves=peaks,stop_reason=reason,
                        elapsed_seconds=time.monotonic()-started,source_unchanged=after==before,
                        exit_code=None if proc is None else proc.poll(),paper_eligible=False,
                        foreign_gpu_processes=foreign_gpu_processes)
            if storage_policy is not None: record['storage_policy']=storage_policy.metadata()
            (guard/"guard.json").write_text(json.dumps(record,indent=2)+"\n")
            artifacts={}
            for directory in (guard,output):
                for path in directory.rglob("*"):
                    if path.is_file(): artifacts[str(path.relative_to(output.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
            (guard/"artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
        print(json.dumps(record,indent=2),flush=True)
        return 2 if reason else proc.returncode


if __name__=="__main__": raise SystemExit(main())
