"""One registered attempt per design/phase; unchanged hardware safety policy."""
import argparse
import json
from pathlib import Path
from guard_wake_design import main
from multibody_offset_config import validate_config,require_cpu_prerequisites,output_path
from multibody_penalty_storage import ExternalStorage

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--mode',choices=['forward'],required=True)
    p.add_argument('--user-approved-low-disk',action='store_true',required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a,_=p.parse_known_args();s=json.loads(a.config.read_text());validate_config(s)
    if a.output_dir.resolve()!=output_path(s):raise ValueError('one registered attempt; no automatic retry')
    require_cpu_prerequisites()
    raise SystemExit(main(driver='run_multibody_offset.py',validator=validate_config,storage_policy=ExternalStorage()))
