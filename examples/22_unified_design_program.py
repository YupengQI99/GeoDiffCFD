"""End-to-end geometry -> Brinkman -> flow -> objective -> geometry gradient.

This experiment uses the verification-oriented periodic MAC kernel.  It checks
three properties before any optimization claim is made:

1. the mobility-weighted pressure projection removes discrete divergence;
2. reverse-mode AD reaches all geometric parameters in one call; and
3. a directional Taylor test exhibits second-order remainder decay.

The transient horizon (80 explicit steps with centred advection) exercises the
finite-horizon backend; the record states this horizon explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_cad.differentiable.mac_periodic import (
    MACParams,
    cell_centres,
    create_initial_state,
    create_periodic_grid,
    divergence_norms,
    mean_flow_loss,
    simulate,
)
from jax_cad.run_artifacts import provenance_record, write_json
from jax_cad.unified import (
    DifferentiableProgram,
    directional_taylor_test,
    make_periodic_circle_brinkman_map,
)


def main() -> None:
    grid = create_periodic_grid(nx=32, ny=20, lx=2.0, ly=1.0)
    x, y = cell_centres(grid, dtype=jnp.float64)
    params = MACParams(
        viscosity=0.05,
        dt=0.0015,
        force_x=1.0,
        include_advection=True,
        mobility_weighted=True,
        pressure_tol=1e-12,
        pressure_maxiter=500,
    )
    n_steps = 80
    state0 = create_initial_state(grid, dtype=jnp.float64)

    design_map = make_periodic_circle_brinkman_map(
        x,
        y,
        lx=grid.lx,
        ly=grid.ly,
        alpha_max=120.0,
        epsilon=0.75 * min(grid.dx, grid.dy),
    )

    def solve(alpha):
        return simulate(state0, alpha, grid, params, n_steps=n_steps)

    program = DifferentiableProgram(
        design_map=design_map,
        solver=solve,
        objective=lambda state, alpha, theta: mean_flow_loss(state, alpha),
    )

    theta = jnp.array([1.0, 0.5, 0.17], dtype=jnp.float64)
    (value, aux), grad = program.value_grad_and_aux(theta)
    div_rms, div_max = divergence_norms(aux.state, grid)

    direction = jnp.array([0.3, -0.2, 0.5], dtype=jnp.float64)
    taylor = directional_taylor_test(
        program.value,
        theta,
        direction,
        epsilons=(4e-3, 2e-3, 1e-3, 5e-4),
    )

    report = {
        "configuration": {
            "grid": [grid.nx, grid.ny],
            "domain": [grid.lx, grid.ly],
            "spacing": [grid.dx, grid.dy],
            "viscosity": params.viscosity,
            "dt": params.dt,
            "n_steps": n_steps,
            "final_time": n_steps * params.dt,
            "alpha_max": 120.0,
            "epsilon": 0.75 * min(grid.dx, grid.dy),
            "epsilon_over_h": 0.75,
            "include_advection": params.include_advection,
            "mobility_weighted": params.mobility_weighted,
            "pressure_tol": params.pressure_tol,
            "sensitivity_backend": "unrolled reverse mode through jax.lax.scan",
        },
        "design": {
            "theta": np.asarray(theta).tolist(),
            "meaning": ["cx", "cy", "radius"],
        },
        "objective": float(value),
        "gradient": np.asarray(grad).tolist(),
        "divergence": {
            "rms": float(div_rms),
            "max_abs": float(div_max),
        },
        "taylor_test": {
            "epsilons": np.asarray(taylor.epsilons).tolist(),
            "zero_order_error": np.asarray(taylor.zero_order_error).tolist(),
            "first_order_error": np.asarray(taylor.first_order_error).tolist(),
            "zero_order_rate": np.asarray(taylor.zero_order_rate).tolist(),
            "first_order_rate": np.asarray(taylor.first_order_rate).tolist(),
            "directional_derivative": float(taylor.directional_derivative),
        },
        "provenance": {**provenance_record(), "script": "examples/22_unified_design_program.py"},
    }

    out = Path(__file__).resolve().parents[1] / "output" / "data"
    path = out / "22_unified_design_program.json"
    write_json(path, report)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
