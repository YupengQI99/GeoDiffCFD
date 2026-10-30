"""Safeguarded projected gradient descent for bound-constrained geometry updates.

The routine is deliberately simple and deterministic so that every reported
closed-loop result can be audited step by step: normalized steepest-descent
directions, box projection, Armijo backtracking, and a projected-gradient
stationarity measure.  It operates on host arrays and calls user-supplied JAX
functions for the objective and its gradient.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np


class ProjectedDescentResult(NamedTuple):
    """Outcome of :func:`projected_gradient_descent`."""

    coefficients: np.ndarray
    trace: list[dict[str, Any]]
    converged: bool
    iterations: int
    initial_projected_gradient_norm: float
    final_projected_gradient_norm: float
    termination: str


def projected_gradient(x: np.ndarray, gradient: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Zero the gradient components that point out of the active bounds."""

    projected = np.array(gradient, dtype=np.float64, copy=True)
    at_upper = (x >= upper - 1e-12) & (gradient < 0.0)
    at_lower = (x <= lower + 1e-12) & (gradient > 0.0)
    projected[at_upper | at_lower] = 0.0
    return projected


def projected_gradient_descent(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    value: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    lower: float,
    upper: float,
    initial_step: float,
    max_iterations: int,
    relative_gradient_tolerance: float = 1e-3,
    sufficient_decrease: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 12,
    step_growth: float = 1.0,
    reference_gradient_norm: float | None = None,
) -> ProjectedDescentResult:
    """Minimize ``value`` over the box ``[lower, upper]^n``.

    Each iteration moves along the normalized projected steepest-descent
    direction with a trial step ``s`` (so that ``||x_{k+1}-x_k||_2 <= s`` before
    projection), projects onto the box, and accepts the candidate if the Armijo
    condition ``J(x_c) <= J(x) + c * g^T (x_c - x)`` holds.  The loop stops when
    the projected-gradient norm falls below ``relative_gradient_tolerance``
    times its initial value, when the line search fails, or after
    ``max_iterations`` accepted or rejected iterations.

    A continuation may supply the ORIGINAL run's ``reference_gradient_norm``
    to preserve its stopping threshold across independently guarded segments.
    This does not change the reported initial norm of the current segment.
    """

    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    if initial_step <= 0:
        raise ValueError("initial_step must be positive")
    if reference_gradient_norm is not None and (
        not np.isfinite(reference_gradient_norm) or reference_gradient_norm <= 0.0
    ):
        raise ValueError("reference gradient norm must be finite and positive")
    x = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)
    step = float(initial_step)
    trace: list[dict[str, Any]] = []
    converged = False
    termination = "max_iterations"
    initial_pg_norm: float | None = None
    pg_norm = float("nan")

    for iteration in range(max_iterations):
        fx, gx = value_and_grad(x)
        fx = float(fx)
        gx = np.asarray(gx, dtype=np.float64)
        pg = projected_gradient(x, gx, lower, upper)
        pg_norm = float(np.linalg.norm(pg))
        if initial_pg_norm is None:
            initial_pg_norm = pg_norm
        record: dict[str, Any] = {
            "step": iteration,
            "objective": fx,
            "gradient_norm": float(np.linalg.norm(gx)),
            "projected_gradient_norm": pg_norm,
            "coefficients": x.tolist(),
        }
        stopping_reference = initial_pg_norm if reference_gradient_norm is None else reference_gradient_norm
        if pg_norm <= relative_gradient_tolerance * stopping_reference:
            record.update(
                {"accepted": False, "step_length": 0.0, "coefficient_change_norm": 0.0,
                 "linear_prediction": 0.0, "realised_change": 0.0, "backtracks": 0}
            )
            trace.append(record)
            converged = True
            termination = "projected_gradient_tolerance"
            break

        direction = -pg / pg_norm
        accepted = False
        trial = step
        backtracks = 0
        for backtracks in range(max_backtracks + 1):
            candidate = np.clip(x + trial * direction, lower, upper)
            delta = candidate - x
            prediction = float(gx @ delta)
            fc = float(value(candidate))
            if fc <= fx + sufficient_decrease * prediction and prediction < 0.0:
                accepted = True
                break
            trial *= backtracking_factor
        record.update(
            {
                "accepted": accepted,
                "step_length": float(trial) if accepted else 0.0,
                "coefficient_change_norm": float(np.linalg.norm(delta)) if accepted else 0.0,
                "linear_prediction": prediction if accepted else 0.0,
                "realised_change": (fc - fx) if accepted else 0.0,
                "backtracks": backtracks,
            }
        )
        trace.append(record)
        if not accepted:
            termination = "line_search_failure"
            break
        x = candidate
        step = min(float(initial_step), trial * step_growth)

    return ProjectedDescentResult(
        coefficients=x,
        trace=trace,
        converged=converged,
        iterations=len(trace),
        initial_projected_gradient_norm=float(initial_pg_norm if initial_pg_norm is not None else np.nan),
        final_projected_gradient_norm=pg_norm,
        termination=termination,
    )
