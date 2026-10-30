"""Registered original forwards and A/C half-timestep follow-ups; no optimisation."""
import argparse
from guard_wake_design import main
from multibody_penalty_config import validate_config
from multibody_penalty_storage import ExternalStorage

if __name__ == '__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', required=True, choices=['forward'])
    parser.add_argument('--user-approved-low-disk', action='store_true', required=True)
    parser.parse_known_args()
    raise SystemExit(main(driver='run_multibody_penalty.py', validator=validate_config,
                          storage_policy=ExternalStorage()))
