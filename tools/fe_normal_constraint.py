"""Independent FE convolution-normal constraints and full augmented Newton solve.

Prepared outside the frozen GPU source tree. Intended to be imported only
by the bounded, pinned serial DOLFINx diagnostic. No JAX operator is imported.
The continuous constraints are integral(delta_h(x-X_m) n_m.u(x)) dx = 0.
The JAX native B-spline rule is a midpoint approximation to these integrals;
FE quadrature/refinement error and MAC discretisation error remain distinct.
"""
from __future__ import annotations

import json
import math
import time


def validate_marker_scope(model, normal):
    """Keep ordinary scope; allow only the authorized fine model's 96 rows.

    CPU mesh, container RAM/no-swap and wall-time limits are unchanged.
    This does not authorize any GPU launch or new physical-model sweep.
    """
    count = normal["markers"]
    authorization = model.get("fine_reference_authorization")
    if type(count) is int and 8 <= count <= 64 and authorization is None:
        return count
    from run_normal_cylinder import FINE_AUTHORIZATION
    if type(count) is not int or count != 96 or authorization != FINE_AUTHORIZATION:
        raise ValueError("normal marker count outside bounded scope")
    expected = dict(reference_h=.0025, alpha=640., epsilon=.00375,
                    shift=.11248513031767304*.0025, radius=.05,
                    length=2.2, height=.41, viscosity=.001,
                    density=1., umax=.3, umean=.2)
    if model.get("centre") != [.2, .2] or any(not math.isclose(model[k], v, rel_tol=1e-12, abs_tol=1e-15) for k, v in expected.items()):
        raise ValueError("fine reference must preserve the authorized physical model")
    return count


def multiplier_correction(schur, constraint, projected_residual):
    """Small nonsymmetric Newton Schur solve, with an explicit rank gate."""
    import numpy as np
    scale = 1/np.sqrt(np.maximum(np.abs(np.diag(schur)), 1.e-300))
    equilibrated = scale[:, None]*schur*scale[None, :]
    singular_values = np.linalg.svd(equilibrated, compute_uv=False)
    relative_minimum = float(singular_values[-1]/singular_values[0])
    if not math.isfinite(relative_minimum) or relative_minimum <= 1.e-10:
        raise RuntimeError("rank-deficient FE normal constraint Schur complement")
    delta = scale*np.linalg.solve(equilibrated, scale*(constraint-projected_residual))
    return delta, relative_minimum


def assemble_constraints(problem, mesh, velocity_test, dx, model, normal, bcs):
    import numpy as np
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc
    from petsc4py import PETSc

    if mesh.comm.size != 1:
        raise ValueError("bounded normal FE diagnostic requires one MPI rank")
    count = validate_marker_scope(model, normal)
    h = model["reference_h"]
    theta = np.arange(count)*2*math.pi/count
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    points = np.asarray(model["centre"])+model["radius"]*normals
    arclength = np.full(count, 2*math.pi*model["radius"]/count)
    centre = fem.Constant(mesh, np.zeros(2, dtype=PETSc.ScalarType))
    direction = fem.Constant(mesh, np.zeros(2, dtype=PETSc.ScalarType))
    coordinate = ufl.SpatialCoordinate(mesh)
    def cubic(t):
        a = abs(t)
        return ufl.conditional(ufl.lt(a, 1), 2/3-a*a+.5*a*a*a,
                               ufl.conditional(ufl.lt(a, 2), (2-a)**3/6, 0.))
    kernel = cubic((coordinate[0]-centre[0])/h)*cubic((coordinate[1]-centre[1])/h)/(h*h)
    load = fem.form(kernel*ufl.dot(direction, velocity_test)*dx)
    mass_form = fem.form(kernel*dx)
    first_x_form = fem.form(kernel*coordinate[0]*dx)
    first_y_form = fem.form(kernel*coordinate[1]*dx)
    matrix = np.empty((count, problem.x.getSize()))
    masses, first_moments = [], []
    for i in range(count):
        centre.value[:] = points[i]; direction.value[:] = normals[i]
        vector = petsc.assemble_vector(load)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        matrix[i] = vector.array
        vector.destroy()
        masses.append(float(fem.assemble_scalar(mass_form)))
        first_moments.append([float(fem.assemble_scalar(first_x_form)), float(fem.assemble_scalar(first_y_form))])
    fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs]))
    if np.max(np.abs(matrix[:, fixed])) > 1.e-14:
        raise ValueError("normal kernel overlaps prescribed outer velocity DOFs")
    matrix[:, fixed] = 0.
    masses = np.asarray(masses); first_moments = np.asarray(first_moments)
    mass_error = float(np.max(np.abs(masses-1)))
    moment_error = float(np.max(np.abs(first_moments-points)))
    if mass_error > 1.e-5 or moment_error/h > 1.e-4:
        raise RuntimeError(f"FE kernel quadrature insufficient: mass={mass_error}, moment/h={moment_error/h}")
    return matrix, dict(points=points, normals=normals, arclength=arclength, kernel_masses=masses,
                        kernel_first_moments=first_moments, kernel_mass_error=mass_error,
                        kernel_first_moment_error=moment_error, reference_h=h)


def constrained_newton(problem, solution, mesh, velocity_test, dx, model, normal, bcs, output):
    """Solve [F(z)+B.T xi; B z]=0, with no normal-penalty parameter.

    Use one sparse MUMPS factorisation per Newton iteration and batched solves
    for the small multiplier Schur complement. The normal reaction is retained
    in every residual, line-search evaluation, saved field and force report.
    """
    import numpy as np
    from petsc4py import PETSc

    started = time.perf_counter()
    B, geometry = assemble_constraints(problem, mesh, velocity_test, dx, model, normal, bcs)
    n, count = B.shape[1], B.shape[0]
    problem.x.array[:] = solution.x.array
    z = problem.x
    multiplier = np.zeros(count)
    snes, A = problem.solver, problem.A
    ksp = snes.getKSP()
    rhs = problem.b.duplicate(); solved = problem.b.duplicate(); matvec = problem.b.duplicate()
    dense_rhs = PETSc.Mat().createDense((n, count), array=np.array(B.T, order="F"), comm=mesh.comm)
    dense_rhs.assemble()
    dense_solution = PETSc.Mat().createDense((n, count), comm=mesh.comm)
    dense_solution.setUp()
    history = []
    initial_norm = None

    def evaluate(candidate, reaction):
        z.array[:] = candidate
        snes.computeFunction(z, problem.b)
        momentum = problem.b.array.copy()+B.T @ reaction
        constraint = B @ candidate
        return momentum, constraint, math.sqrt(float(momentum @ momentum+constraint @ constraint))

    try:
        for iteration in range(41):
            current = z.array.copy()
            momentum, constraint, norm = evaluate(current, multiplier)
            if initial_norm is None:
                initial_norm = norm
            target = max(1.e-11, 1.e-10*initial_norm)
            row = dict(iteration=iteration, residual_l2=norm, normal_constraint_max=float(np.max(np.abs(constraint))))
            history.append(row)
            print(json.dumps(dict(normal_newton=row)), flush=True)
            if not math.isfinite(norm):
                raise RuntimeError("nonfinite augmented FE residual")
            if norm <= target:
                break
            if iteration == 40:
                raise RuntimeError("augmented FE Newton iteration cap")
            snes.computeJacobian(z, A, A)
            ksp.setOperators(A); ksp.setUp()
            rhs.array[:] = momentum
            ksp.solve(rhs, solved)
            inverse_residual = solved.array.copy()
            ksp.matSolve(dense_rhs, dense_solution)
            Y = dense_solution.getDenseArray()
            H = B @ Y
            delta_multiplier, relative_minimum = multiplier_correction(H, constraint, B @ inverse_residual)
            delta = -inverse_residual-Y @ delta_multiplier
            solved.array[:] = delta
            A.mult(solved, matvec)
            linear_defect = matvec.array+B.T @ delta_multiplier+momentum
            constraint_defect = B @ delta+constraint
            linear_relative = float(np.linalg.norm(np.r_[linear_defect, constraint_defect])/max(norm, 1.e-30))
            if not math.isfinite(linear_relative) or linear_relative > 1.e-7:
                raise RuntimeError(f"augmented FE linear solve failed fresh residual: {linear_relative}")
            accepted = False
            for exponent in range(9):
                damping = .5**exponent
                trial = current+damping*delta
                trial_multiplier = multiplier+damping*delta_multiplier
                _, _, trial_norm = evaluate(trial, trial_multiplier)
                if math.isfinite(trial_norm) and (trial_norm < (1-1.e-4*damping)*norm or trial_norm <= target):
                    multiplier = trial_multiplier
                    row.update(damping=damping, linear_relative_residual=linear_relative,
                               relative_minimum_singular_value=relative_minimum)
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError("augmented FE Newton line search failed")
        # Synchronise the actual FE function: it does not share z's memory.
        solution.x.array[:] = z.array
        solution.x.scatter_forward()
        final_momentum, final_normal, final_norm = evaluate(z.array.copy(), multiplier)
        force = np.sum((geometry["kernel_masses"]*multiplier)[:, None]*geometry["normals"], axis=0)
        normal_work = float(multiplier @ final_normal)
        np.savez_compressed(output/"normal_constraints.npz", **{k:v for k,v in geometry.items() if hasattr(v, "shape")},
                            integrated_reaction=multiplier, normal_traction=multiplier/geometry["arclength"],
                            normal_velocity=final_normal)
        return dict(converged=True, iterations=iteration, final_residual_l2=final_norm, initial_residual_l2=initial_norm,
                    threshold=target, history=history, constraint_force=force.tolist(),
                    normal_constraint_max=float(np.max(np.abs(final_normal))), normal_work=normal_work,
                    kernel_mass_error=geometry["kernel_mass_error"], kernel_first_moment_error=geometry["kernel_first_moment_error"],
                    markers=count, reference_h=geometry["reference_h"], wall_seconds=time.perf_counter()-started,
                    constraint_definition="integral tensor-cubic delta_h(x-X_m) n_m dot u dx = 0",
                    force_protocol="FE finite-mask reaction plus integrated normal multipliers; no sharp acceptance")
    finally:
        for obj in (rhs, solved, matvec, dense_rhs, dense_solution):
            obj.destroy()
