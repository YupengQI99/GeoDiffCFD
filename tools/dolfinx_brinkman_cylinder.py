#!/usr/bin/env python3
"""Independent full-domain P2/P1 finite-Brinkman diagnostic, NOT sharp DFG validation.

The physical mask is frozen from a JAX configuration, independently of FE mesh
spacing. No hole, cylinder Dirichlet condition, or benchmark force is used.
Only standard-library imports at module load; run the solver in the pinned image.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

import dolfinx_dfg2d1 as evidence

ROOT = Path(__file__).resolve().parents[1]


def load_model(config_path):
    config = json.loads(Path(config_path).read_text())
    if config.get("schema_version") != "dolfinx-finite-brinkman-config-1.0":
        raise ValueError("wrong finite-model config schema")
    if config.get("study") != {"development_only": True, "paper_eligible": False}:
        raise ValueError("finite-model diagnostics cannot claim paper eligibility")
    source_path = (ROOT / config["source_config"]).resolve()
    if not source_path.is_relative_to(ROOT):
        raise ValueError("source config must be in repository")
    if evidence._sha256(source_path) != config["source_config_sha256"]:
        raise ValueError("source configuration hash mismatch")
    source = json.loads(source_path.read_text())
    grid, physics, ib = source["grid"], source["physics"], source["immersed_boundary"]
    h = max(grid["lx"] / grid["nx"], grid["ly"] / grid["ny"])
    model = dict(length=grid["lx"], height=grid["ly"], density=physics["density"],
                 viscosity=physics["viscosity"], centre=physics["cylinder_centre"],
                 radius=physics["cylinder_radius"], umax=physics["maximum_inlet_velocity"],
                 umean=physics["mean_inlet_velocity"], alpha=ib["alpha_max"],
                 epsilon=ib["epsilon_over_h"] * h,
                 shift=ib["mask_midpoint_shift_over_h"] * h, reference_h=h)
    if ib["material_mapping"] != "compact-cosine-heaviside":
        raise ValueError("unsupported mask")
    if source["boundary_conditions"]["outlet"] != "zero-total-traction-finite-volume":
        raise ValueError("unsupported outer boundary model")
    if source["surface_measurement"]["stress_tensor"] != "nu-grad-u-minus-pI":
        raise ValueError("unsupported stress")
    for name in ("length", "height", "density", "viscosity", "radius", "umax", "umean", "alpha", "epsilon"):
        if not math.isfinite(model[name]) or model[name] <= 0:
            raise ValueError(f"invalid {name}")
    for name in ("nx", "ny"):
        value = config["mesh"][name]
        if type(value) is not int or value < 4 or value > (440 if name == "nx" else 82):
            raise ValueError("mesh outside bounded CPU diagnostic scope")
    if config["mesh"].get("diagonal") not in ("right", "left"):
        raise ValueError("explicit triangle diagonal required")
    levels = config["mesh"].get("local_refinements", 0)
    if type(levels) is not int or not 0 <= levels <= 3:
        raise ValueError("at most three bounded local refinements")
    if levels and (config["mesh"]["nx"] > 220 or config["mesh"]["ny"] > 41):
        raise ValueError("local refinement requires <=220x41 base grid")
    if config.get("advection_form", "advective") not in ("advective", "conservative"):
        raise ValueError("unsupported advection form")
    if config["quadrature_degree"] not in (6, 10, 14):
        raise ValueError("unsupported quadrature degree")
    if config["container"]["image"] != "dolfinx/dolfinx@" + evidence.IMAGE_DIGEST:
        raise ValueError("unapproved image digest")
    if "normal_constraint" in config:
        normal_source = (ROOT/config["normal_constraint"]["source_config"]).resolve()
        if not normal_source.is_relative_to(ROOT) or evidence._sha256(normal_source) != config["normal_constraint"]["source_config_sha256"]:
            raise ValueError("normal constraint configuration hash mismatch")
        normal_config = json.loads(normal_source.read_text())
        if normal_config.get("schema_version") != "normal-cylinder-diagnostic-config-1.0" or normal_config.get("source_config_sha256") != config["source_config_sha256"]:
            raise ValueError("normal constraint must pin the same physical mask")
        normal = normal_config["normal_constraint"]
        if normal.get("kernel") != "tensor-cubic-bspline-native-mac" or normal.get("nominal_surface_shift") != 0 or normal.get("angle_origin") != 0:
            raise ValueError("unsupported normal constraint kernel/geometry")
        if normal.get("force_protocol") != "mac-brinkman-plus-normal-reaction-prototype-v1":
            raise ValueError("normal comparison must include complete body reaction")
        if "fine_authorization" in normal_config:
            if not normal_config.get("execution_scope", "").startswith("CPU FE"):
                raise ValueError("fine FE source must be explicitly CPU-only")
            model["fine_reference_authorization"] = normal_config["fine_authorization"]
        from fe_normal_constraint import validate_marker_scope
        validate_marker_scope(model, normal)
        model["normal_constraint"] = normal
        model["normal_constraint_source"] = str(normal_source)
    if "reaction_pressure_diagnostic" in config:
        if "normal_constraint" not in model or config["reaction_pressure_diagnostic"] != dict(protocol="continuous-normal-reaction-pressure-reconstruction-v1",paper_eligible=False):
            raise ValueError("reaction pressure observer requires the frozen normal model/protocol")
    return config, source_path, source, model


def mask_value(radius, model):
    """Scalar independent definition for configuration/profile tests."""
    s = (model["radius"] + model["shift"] - radius) / model["epsilon"]
    return model["alpha"] * (0.0 if s < -1 else 1.0 if s > 1 else
                             0.5 * (1 + s + math.sin(math.pi * s) / math.pi))


def solve(config, source, model, snapshot, output):
    import basix.ufl
    import dolfinx
    import gmsh
    import numpy as np
    import petsc4py
    import ufl
    from dolfinx import fem, mesh as dmesh
    from dolfinx.fem import petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    started = time.perf_counter()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise ValueError("bounded diagnostic requires one MPI rank")
    lx, ly = model["length"], model["height"]
    mesh = dmesh.create_rectangle(comm, [np.array([0., 0.]), np.array([lx, ly])],
                                  [config["mesh"]["nx"], config["mesh"]["ny"]],
                                  cell_type=dmesh.CellType.triangle,
                                  diagonal=getattr(dmesh.DiagonalType, config["mesh"]["diagonal"]))
    for _ in range(config["mesh"].get("local_refinements", 0)):
        mesh.topology.create_entities(1)
        edges = dmesh.locate_entities(mesh, 1, lambda x:
            (x[0]-model["centre"][0])**2 + (x[1]-model["centre"][1])**2 < .12**2)
        # The installed identity partitioner requires parent-cell bookkeeping.
        mesh, _, _ = dmesh.refine(mesh, edges, option=dmesh.RefinementOption.parent_cell)
        if mesh.topology.index_map(2).size_global > 150000:
            raise ValueError("local mesh exceeds bounded CPU cell budget")
    fdim = mesh.topology.dim - 1
    inlet = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0))
    outlet = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], lx))
    walls = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0) | np.isclose(x[1], ly))
    indices = np.concatenate((inlet, outlet, walls))
    tags = np.concatenate((np.full(len(inlet), 11), np.full(len(outlet), 12), np.full(len(walls), 13))).astype(np.int32)
    order = np.argsort(indices)
    facet_tags = dmesh.meshtags(mesh, fdim, indices[order], tags[order])
    W = fem.functionspace(mesh, basix.ufl.mixed_element([
        basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,)),
        basix.ufl.element("Lagrange", mesh.basix_cell(), 1)]))
    V, _ = W.sub(0).collapse()
    inflow, zero = fem.Function(V), fem.Function(V)

    def profile(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * model["umax"] * x[1] * (ly-x[1]) / ly**2
        return values

    inflow.interpolate(profile)
    bcs = [fem.dirichletbc(inflow, fem.locate_dofs_topological((W.sub(0), V), fdim, inlet), W.sub(0)),
           fem.dirichletbc(zero, fem.locate_dofs_topological((W.sub(0), V), fdim, walls), W.sub(0))]
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": config["quadrature_degree"]})
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    x = ufl.SpatialCoordinate(mesh)
    radius = ufl.sqrt((x[0]-model["centre"][0])**2 + (x[1]-model["centre"][1])**2)
    s = (model["radius"] + model["shift"] - radius) / model["epsilon"]
    alpha = model["alpha"] * ufl.conditional(ufl.lt(s, -1), 0,
        ufl.conditional(ufl.gt(s, 1), 1, .5*(1+s+ufl.sin(math.pi*s)/math.pi)))
    rho, mu = model["density"], model["density"] * model["viscosity"]
    v, q = ufl.TestFunctions(W)

    def linear(u, p):
        return (mu*ufl.inner(ufl.grad(u), ufl.grad(v)) - p*ufl.div(v)
                + q*ufl.div(u) + rho*alpha*ufl.inner(u, v))*dx

    ut, pt = ufl.TrialFunctions(W)
    options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
               "ksp_error_if_not_converged": True}
    t0 = time.perf_counter()
    stokes = petsc.LinearProblem(linear(ut, pt), ufl.inner(fem.Constant(mesh, np.zeros(2, dtype=PETSc.ScalarType)), v)*dx,
                                bcs=bcs, petsc_options_prefix="brinkman_stokes_", petsc_options=options)
    initial = stokes.solve()
    initial.x.scatter_forward()
    stokes_time = time.perf_counter()-t0
    solution = fem.Function(W)
    solution.x.array[:] = initial.x.array
    u, p = ufl.split(solution)
    # Advective continuum form, independently discretised with weak divergence.
    # Its difference from conservative form must vanish with mesh refinement.
    F = linear(u, p) + rho*ufl.inner(ufl.dot(ufl.grad(u), u), v)*dx
    if config.get("advection_form", "advective") == "conservative":
        F += rho*ufl.div(u)*ufl.inner(u, v)*dx
    problem = petsc.NonlinearProblem(F, solution, bcs=bcs, J=ufl.derivative(F, solution),
        petsc_options_prefix="brinkman_newton_", petsc_options={**options,
        "snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-10,
        "snes_atol": 1e-11, "snes_max_it": 40, "snes_error_if_not_converged": True})
    history = []

    def monitor(_snes, iteration, norm):
        row = dict(iteration=int(iteration), residual_l2=float(norm))
        history.append(row)
        print(json.dumps(row), flush=True)

    problem.solver.setMonitor(monitor)
    t0 = time.perf_counter()
    normal_report = None
    if "normal_constraint" in model:
        from fe_normal_constraint import constrained_newton
        normal_report = constrained_newton(problem, solution, mesh, v, dx, model, model["normal_constraint"], bcs, output)
        history = [{"iteration": row["iteration"], "residual_l2": row["residual_l2"]} for row in normal_report["history"]]
    else:
        problem.solve()
    solve_time = time.perf_counter()-t0
    if normal_report is None and problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("Newton solve did not converge")
    solution.x.scatter_forward()
    uh, ph = solution.sub(0).collapse(), solution.sub(1).collapse()

    def integral(expr):
        return evidence._assemble_scalar(fem.form(expr), comm)

    scale = 2 / (rho*model["umean"]**2*2*model["radius"])
    forces = [integral(rho*alpha*uh[i]*dx) for i in (0, 1)]
    brinkman_forces = list(forces)
    if normal_report is not None:
        forces = [forces[i]+normal_report["constraint_force"][i] for i in (0, 1)]
    centre, r = model["centre"], model["radius"]
    front = evidence._probe(ph, [centre[0]-r, centre[1]], mesh)
    rear = evidence._probe(ph, [centre[0]+r, centre[1]], mesh)
    # Match the physical radial offsets and polynomial degree of the JAX pressure
    # estimator, while evaluating the independent FE field directly (not MAC interpolation).
    offsets = np.array(source["surface_measurement"]["offsets_over_h"])*model["reference_h"]
    traces = []
    for sign in (-1, 1):
        values = [evidence._probe(ph, [centre[0]+sign*(r+d), centre[1]], mesh) for d in offsets]
        traces.append(float(np.polynomial.polynomial.polyfit(offsets, values,
                      source["surface_measurement"]["pressure_polynomial_degree"])[0]))
    normal = ufl.FacetNormal(mesh)
    inlet_flux = -integral(ufl.dot(uh, normal)*ds(11))
    outlet_flux = integral(ufl.dot(uh, normal)*ds(12))
    area = integral(1.0*dx)
    div_rms = math.sqrt(max(0., integral(ufl.div(uh)**2*dx))/area)
    D = fem.functionspace(mesh, ("DG", 1))
    divergence = fem.Function(D)
    divergence.interpolate(fem.Expression(ufl.div(uh), D.element.interpolation_points))
    # Save native DOFs and their coordinates; serial uniform triangle topology is
    # reproducible from the pinned configuration. No enormous visualisation files.
    np.savez_compressed(output / "fields.npz", velocity=uh.x.array, pressure=ph.x.array,
                        velocity_coordinates=uh.function_space.tabulate_dof_coordinates(),
                        pressure_coordinates=ph.function_space.tabulate_dof_coordinates())
    metrics = dict(force_x=forces[0], force_y=forces[1], drag_coefficient=scale*forces[0],
                   lift_coefficient=scale*forces[1], pressure_front=front, pressure_rear=rear,
                   pressure_difference=front-rear, pressure_difference_radial_extrapolated=traces[0]-traces[1],
                   integrated_alpha=integral(alpha*dx))
    if normal_report is not None:
        metrics.update(normal_constraint=normal_report, brinkman_force_x=brinkman_forces[0], brinkman_force_y=brinkman_forces[1],
                       constraint_force_x=normal_report["constraint_force"][0], constraint_force_y=normal_report["constraint_force"][1])
    if "reaction_pressure_diagnostic" in config:
        import sys
        sys.path.insert(0,str(ROOT))
        from jax_cad.reaction_pressure_trace import observe_continuous
        with np.load(output/"normal_constraints.npz",allow_pickle=False) as saved:
            metrics["reaction_pressure_diagnostic"]=observe_continuous(
                lambda xy:evidence._probe(ph,xy,mesh),points=saved["points"],normals=saved["normals"],
                integrated_reaction=saved["integrated_reaction"],h=model["reference_h"],centre=centre,radius=r)
    return dict(case=config["case"], run_id=config["run_id"],
        formulation=dict(model=model, geometry="full rectangle including solid", momentum="rho*(grad(u)u)+rho*alpha*u-mu*div(grad(u))+grad(p)=0",
                         outlet="natural (mu*grad(u)-pI)n=0", force=("rho integral(alpha*u) plus integrated normal multipliers" if normal_report is not None else "rho integral(alpha*u) over full rectangle"),
                         pressure_difference="direct finite-field nominal probes; NOT sharp or JAX extrapolated trace",
                         pressure_difference_radial_extrapolated="same physical offsets/degree as source, evaluated in FE field",
                         alpha_sampling="continuous compact-cosine expression at FE quadrature points",
                         velocity_element="P2", pressure_element="P1", quadrature_degree=config["quadrature_degree"],
                         advection_form=config.get("advection_form", "advective")),
        physical_tags=dict(inlet=11, outlet=12, walls=13, cylinder=None),
        mesh={**config["mesh"], "cells": mesh.topology.index_map(2).size_global,
              "mixed_dofs": W.dofmap.index_map.size_global*W.dofmap.index_map_bs},
        metrics=metrics, reference={"kind": "same finite model; no sharp benchmark acceptance"},
        original_interval_membership={}, relative_errors={"drag_coefficient": None},
        conservation=dict(inlet_flux=inlet_flux, outlet_flux=outlet_flux, fluid_area=area,
                          relative_inlet_outlet_mismatch=abs(inlet_flux-outlet_flux)/abs(inlet_flux),
                          divergence_rms=div_rms, divergence_max_at_dg1_nodes=float(np.max(np.abs(divergence.x.array)))),
        nonlinear_convergence=dict(converged=True, iterations=normal_report["iterations"] if normal_report is not None else problem.solver.getIterationNumber(),
                                   snes_reason_code=None if normal_report is not None else problem.solver.getConvergedReason(),
                                   method="augmented-normal-constraint-newton" if normal_report is not None else "petsc-snes-newtonls",
                                   final_residual_l2=history[-1]["residual_l2"], history=history),
        timing_seconds=dict(stokes_initialisation=stokes_time, nonlinear_solve=solve_time, total=time.perf_counter()-started),
        environment=dict(hostname=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, dolfinx=dolfinx.__version__, gmsh=gmsh.__version__, petsc4py=petsc4py.__version__,
                         mpi_size=comm.size, mpi_rank=comm.rank, git_commit=snapshot["commit"], git_branch=snapshot["branch"],
                         git_repository=snapshot["repository"], git_dirty=snapshot["dirty"], source_snapshot=snapshot,
                         reproducibility_environment={k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XDG_CACHE_HOME", "XDG_CONFIG_HOME") if k in os.environ}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config, source_path, source, model = load_model(args.config)
    evidence._check_runtime_versions(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing nonempty output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = (Path(__file__).resolve(), Path(evidence.__file__).resolve(), args.config.resolve(), source_path)
    if "normal_constraint" in model:
        required += (ROOT/"tools/fe_normal_constraint.py", Path(model["normal_constraint_source"]))
    if "fine_reference_authorization" in model:
        required += (ROOT/"tools/run_normal_cylinder.py",)
    if "reaction_pressure_diagnostic" in config:
        required += (ROOT/"jax_cad/reaction_pressure_trace.py",ROOT/"jax_cad/native_pressure_trace.py")
    observed = evidence._runtime_option_paths()
    snapshot = evidence._git_snapshot(required, observed)
    if snapshot["dirty"] is not False:
        raise RuntimeError("finite-model solve requires tracked clean source")
    start = datetime.now(timezone.utc).isoformat()
    result = solve(config, source, model, snapshot, args.output_dir)
    if evidence._git_snapshot(required, observed) != snapshot:
        raise RuntimeError("source changed during solve")
    path = evidence._write_experiment_bundle(args.output_dir, args.config, config, result,
        started_utc=start, ended_utc=datetime.now(timezone.utc).isoformat(), record_overrides=dict(
            experiment_id="dolfinx_normal_constraint_diagnostic" if "normal_constraint" in model else "dolfinx_finite_brinkman_diagnostic",
            reference_comparison=dict(paper_eligible=False, boundary_conditions_exact=True,
                                      quantity_definition_exact=False, geometry_exact=True,
                                      reason="Independent continuous finite mask; normal-convolution constraints included when configured. No sharp-wall acceptance. MAC interpolation and FE quadrature require refinement; direct pressure is not radial extrapolation."),
            objective=dict(name="finite_model_total_reaction_drag" if "normal_constraint" in model else "finite_brinkman_volume_drag", value=result["metrics"]["drag_coefficient"],
                           units="dimensionless", normalization="2F/(rho*Umean^2*diameter)"),
            notes="Independent fixed-physical-model diagnostic. P2/P1 is weakly divergence-free; inspect divergence and mesh convergence. Not paper evidence of sharp-cylinder convergence."))
    record = json.loads(path.read_text())
    record["artifacts"].append(evidence._artifact(args.output_dir/"fields.npz", args.output_dir, "application/octet-stream", "other"))
    if "normal_constraint" in model:
        record["artifacts"].append(evidence._artifact(args.output_dir/"normal_constraints.npz", args.output_dir, "application/octet-stream", "other"))
    evidence._write_json(path, record)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
