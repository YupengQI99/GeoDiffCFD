#!/usr/bin/env python3
"""Same captured hardware supervisor, explicit new multibody development scope."""
from guard_wake_design import main
from multibody_wake_config import validate_config

if __name__=="__main__":
    raise SystemExit(main(driver="run_multibody_wake.py",validator=validate_config))
