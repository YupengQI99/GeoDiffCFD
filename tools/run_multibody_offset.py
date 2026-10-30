"""Exact same transient solver; separate calibrated-offset sensitivity scope."""
from run_multibody_penalty import main
import multibody_offset_config as registry
import multibody_offset_observer as observer

if __name__=='__main__':raise SystemExit(main(registry=registry,observer=observer))
