#!/usr/bin/env python3
"""Reuse S1's exact wake trajectory driver, never forward_verification."""
import dfg2d1_wall_shift_config as registry
from run_wall_shift_dfg import main

if __name__=='__main__':
    raise SystemExit(main(registry=registry,result_schema='dfg2d1-wall-shift-corroboration-result-1',
        scope='DFG 2D-1 steady corroboration only; no benchmark gate, stage 2 or manuscript changes'))
