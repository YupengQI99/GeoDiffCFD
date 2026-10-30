"""Composable design-to-physics-to-objective programs.

The central object in this module is :class:`DifferentiableProgram`.  It makes
one contract explicit::

    design variables -> physics fields -> solver state -> scalar objective

Every stage may be any JAX-compatible callable and may exchange arbitrary
PyTrees.  Consequently the same orchestration layer can be used for analytic
SDF parameters, density/topology variables, sampled-SDF transformations, and
future neural geometry representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, NamedTuple, TypeVar

import jax


DesignT = TypeVar("DesignT")
PhysicsT = TypeVar("PhysicsT")
StateT = TypeVar("StateT")
ScalarT = TypeVar("ScalarT")


class ProgramAux(NamedTuple, Generic[PhysicsT, StateT]):
    """Auxiliary outputs from a differentiable program evaluation."""

    physics: PhysicsT
    state: StateT


@dataclass(frozen=True)
class DifferentiableProgram(Generic[DesignT, PhysicsT, StateT, ScalarT]):
    """Compose geometry/design, physics, and objective into one JAX program.

    Parameters
    ----------
    design_map:
        Maps active design variables to the PyTree consumed by the solver.
        Examples include ``theta -> alpha`` for analytic SDFs and
        ``gamma -> material_fields`` for density-based topology optimization.
    solver:
        Maps physics fields to the final solver state.  It may represent a
        fixed-horizon transient solve, a converged nonlinear solve with a custom
        derivative rule, or a surrogate model.
    objective:
        Returns a scalar from ``(state, physics, design)``.  The explicit design
        argument allows regularizers and constraints to remain in the same
        differentiated program.

    Notes
    -----
    The class intentionally does not prescribe a particular PDE, geometry
    representation, or differentiation backend.  ``jax.grad`` differentiates
    the *executed discrete program*.  A future implicit solver can therefore
    replace an unrolled solver without changing the design map or objective API.
    """

    design_map: Callable[[DesignT], PhysicsT]
    solver: Callable[[PhysicsT], StateT]
    objective: Callable[[StateT, PhysicsT, DesignT], ScalarT]

    def value(self, design: DesignT) -> ScalarT:
        """Evaluate the scalar objective."""

        physics = self.design_map(design)
        state = self.solver(physics)
        return self.objective(state, physics, design)

    def value_with_aux(self, design: DesignT) -> tuple[ScalarT, ProgramAux[PhysicsT, StateT]]:
        """Evaluate the objective and return the intermediate PyTrees."""

        physics = self.design_map(design)
        state = self.solver(physics)
        value = self.objective(state, physics, design)
        return value, ProgramAux(physics=physics, state=state)

    def value_and_grad(self, design: DesignT) -> tuple[ScalarT, DesignT]:
        """Return objective value and reverse-mode gradient."""

        return jax.value_and_grad(self.value)(design)

    def value_grad_and_aux(
        self, design: DesignT
    ) -> tuple[tuple[ScalarT, ProgramAux[PhysicsT, StateT]], DesignT]:
        """Return objective, intermediate PyTrees, and reverse-mode gradient."""

        return jax.value_and_grad(self.value_with_aux, has_aux=True)(design)

    def jitted_value_and_grad(self) -> Callable[[DesignT], tuple[ScalarT, DesignT]]:
        """Build a JIT-compiled value-and-gradient callable.

        The returned function closes over the program's Python callables.  Build
        one compiled callable per fixed numerical configuration.
        """

        return jax.jit(jax.value_and_grad(self.value))


def compose_program(
    design_map: Callable[[Any], Any],
    solver: Callable[[Any], Any],
    objective: Callable[[Any, Any, Any], Any],
) -> DifferentiableProgram[Any, Any, Any, Any]:
    """Functional constructor useful when type parameters are not needed."""

    return DifferentiableProgram(
        design_map=design_map,
        solver=solver,
        objective=objective,
    )
