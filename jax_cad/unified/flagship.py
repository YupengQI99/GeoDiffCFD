"""Committed configuration of the periodic Fourier-obstacle benchmark.

Every manuscript experiment that uses the area-preserving Fourier obstacle
(the closed-loop result-to-geometry demonstrations and the joint grid /
interface / penalization study) builds its problem from this dataclass so that
grid, physics, material, regularization and optimizer settings are defined in
exactly one committed place and are serialized into every archived record.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_cad.differentiable.mac_periodic import (
    MACParams,
    MACState,
    PeriodicGrid2D,
    cell_centres,
    create_initial_state,
    create_periodic_grid,
    make_fixed_point_preconditioner,
    mean_flow_loss,
    simulate,
    steady_solve_diagnostics,
    steady_stokes_brinkman,
    step,
)
from jax_cad.unified.geometry import (
    fourier_mode_sobolev_energy,
    make_periodic_fourier_material_map,
)
from jax_cad.unified.implicit import (
    make_implicit_affine_solver,
    make_implicit_fixed_point_solver,
)
from jax_cad.unified.materials import MaterialFields

Array = jax.Array


@dataclass(frozen=True)
class FourierObstacleBenchmark:
    """Periodic Stokes--Brinkman flow past an area-preserving Fourier obstacle."""

    nx: int = 32
    ny: int = 20
    lx: float = 2.0
    ly: float = 1.0
    viscosity: float = 0.06
    diffusion_number: float = 0.05904
    force_x: float = 1.0
    include_advection: bool = False
    center: tuple[float, float] = (0.92, 0.47)
    base_radius: float = 0.26
    modes: tuple[int, ...] = (2, 3, 4, 5)
    alpha_max: float = 2400.0
    epsilon_over_h: float = 1.5
    profile: str = "compact"
    regularization_weight: float = 1.0e-4
    regularization_order: int = 2
    coefficient_bound: float = 0.30
    initial_step: float = 0.04
    max_iterations: int = 60
    relative_gradient_tolerance: float = 1.0e-3
    mobility_weighted: bool = True
    pressure_tol: float = 1.0e-12
    pressure_maxiter: int = 500
    primal_method: str = "gmres"
    fixed_point_max_iterations: int = 400_000
    fixed_point_atol: float = 1.0e-12
    fixed_point_rtol: float = 1.0e-12
    primal_tol: float = 1.0e-12
    primal_atol: float = 1.0e-14
    primal_maxiter: int = 40_000
    primal_restart: int = 60
    use_preconditioner: bool = False
    preconditioner_alpha_fraction: float = 0.05
    adjoint_tol: float = 1.0e-10
    adjoint_atol: float = 1.0e-12
    adjoint_maxiter: int = 40_000
    adjoint_restart: int = 60

    # ------------------------------------------------------------------ basics
    def replace(self, **changes: Any) -> "FourierObstacleBenchmark":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record.update(
            {
                "dx": self.grid.dx,
                "dy": self.grid.dy,
                "h": self.h,
                "dt": self.dt,
                "epsilon": self.epsilon,
                "n_geometry_parameters": self.n_design,
                "brinkman_layer_thickness": float((self.viscosity / self.alpha_max) ** 0.5),
            }
        )
        return record

    @property
    def grid(self) -> PeriodicGrid2D:
        return create_periodic_grid(self.nx, self.ny, self.lx, self.ly)

    @property
    def h(self) -> float:
        grid = self.grid
        return min(grid.dx, grid.dy)

    @property
    def dt(self) -> float:
        """Time step fixed by the diffusion number ``nu dt (1/dx^2 + 1/dy^2)``."""

        grid = self.grid
        return self.diffusion_number / (
            self.viscosity * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)
        )

    @property
    def epsilon(self) -> float:
        return self.epsilon_over_h * self.h

    @property
    def n_design(self) -> int:
        return 2 * len(self.modes)

    def params(self, *, dt: float | None = None, mobility_weighted: bool | None = None) -> MACParams:
        return MACParams(
            viscosity=self.viscosity,
            dt=self.dt if dt is None else dt,
            force_x=self.force_x,
            force_y=0.0,
            include_advection=self.include_advection,
            mobility_weighted=self.mobility_weighted if mobility_weighted is None else mobility_weighted,
            pressure_tol=self.pressure_tol,
            pressure_maxiter=self.pressure_maxiter,
        )

    def initial_state(self) -> MACState:
        return create_initial_state(self.grid, dtype=jnp.float64)

    def design_map(self) -> Callable[[Array], MaterialFields]:
        grid = self.grid
        x, y = cell_centres(grid, dtype=jnp.float64)
        return make_periodic_fourier_material_map(
            x,
            y,
            center=self.center,
            base_radius=self.base_radius,
            modes=self.modes,
            lx=grid.lx,
            ly=grid.ly,
            alpha_max=self.alpha_max,
            epsilon=self.epsilon,
            profile=self.profile,
        )

    def zero_coefficients(self) -> Array:
        return jnp.zeros(self.n_design, dtype=jnp.float64)

    # -------------------------------------------------------------- objectives
    def physical_loss(self, state: MACState) -> Array:
        return mean_flow_loss(state)

    def objective(self, state: MACState, fields: MaterialFields, coefficients: Array) -> Array:
        del fields
        return self.physical_loss(state) + self.regularization_weight * (
            fourier_mode_sobolev_energy(coefficients, self.modes, order=self.regularization_order)
        )

    # ----------------------------------------------------------------- solvers
    def transient_solver(
        self, n_steps: int, *, params: MACParams | None = None
    ) -> Callable[[MaterialFields], MACState]:
        grid = self.grid
        params = self.params() if params is None else params
        state0 = self.initial_state()

        def solver(fields: MaterialFields) -> MACState:
            return simulate(state0, fields.momentum_resistance, grid, params, n_steps=n_steps)

        return solver

    def implicit_solver(
        self, *, params: MACParams | None = None
    ) -> tuple[Callable[[MaterialFields], MACState], Callable[[Array], tuple[MACState, Any]]]:
        """Return ``fields -> converged state`` with an implicit VJP and a diagnostic primal."""

        grid = self.grid
        params = self.params() if params is None else params
        state0 = self.initial_state()
        update = lambda state, alpha: step(state, alpha, grid, params)
        if self.primal_method == "saddle":
            if params.include_advection:
                raise ValueError("the direct steady solve requires an affine (Stokes) map")

            def solve(alpha: Array, _state0: MACState | None = None) -> MACState:
                return steady_stokes_brinkman(
                    alpha, grid, params,
                    tol=self.primal_tol, atol=self.primal_atol,
                    maxiter=self.primal_maxiter, restart=self.primal_restart,
                )

            def primal(alpha: Array, _state0: MACState | None = None):
                state = solve(alpha)
                diagnostics = steady_solve_diagnostics(state, alpha, grid, params)
                from jax_cad.unified.implicit import FixedPointDiagnostics

                return state, FixedPointDiagnostics(
                    iterations=jnp.asarray(-1, dtype=jnp.int32),
                    residual_norm=diagnostics.momentum_residual_rms,
                    converged=diagnostics.converged,
                )
        elif self.primal_method == "gmres":
            if params.include_advection:
                raise ValueError("the GMRES primal requires an affine (Stokes) update map")
            preconditioner = (
                make_fixed_point_preconditioner(
                    grid, params, self.preconditioner_alpha_fraction * self.alpha_max
                )
                if self.use_preconditioner
                else None
            )
            solve, primal = make_implicit_affine_solver(
                update,
                tol=self.primal_tol,
                atol=self.primal_atol,
                maxiter=self.primal_maxiter,
                restart=self.primal_restart,
                linear_tol=self.adjoint_tol,
                linear_atol=self.adjoint_atol,
                linear_maxiter=self.adjoint_maxiter,
                linear_restart=self.adjoint_restart,
                preconditioner=preconditioner,
            )
        elif self.primal_method == "picard":
            solve, primal = make_implicit_fixed_point_solver(
                update,
                max_iterations=self.fixed_point_max_iterations,
                atol=self.fixed_point_atol,
                rtol=self.fixed_point_rtol,
                linear_tol=self.adjoint_tol,
                linear_atol=self.adjoint_atol,
                linear_maxiter=self.adjoint_maxiter,
                linear_restart=self.adjoint_restart,
            )
        else:
            raise ValueError("primal_method must be 'gmres' or 'picard'")

        def solver(fields: MaterialFields) -> MACState:
            return solve(fields.momentum_resistance, state0)

        def primal_with_diagnostics(alpha: Array):
            return primal(alpha, state0)

        return solver, primal_with_diagnostics
