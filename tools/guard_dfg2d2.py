"""One registered single-cylinder diagnostic, original hardware/storage protections."""
import argparse
from guard_wake_design import main
from dfg2d2_config import validate_config
from multibody_penalty_storage import ExternalStorage

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--mode',choices=['forward'],required=True)
    p.add_argument('--user-approved-low-disk',action='store_true',required=True)
    p.parse_known_args()
    raise SystemExit(main(driver='run_dfg2d2.py',validator=validate_config,storage_policy=ExternalStorage()))
