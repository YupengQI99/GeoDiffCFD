#!/usr/bin/env python3
"""Captured, guarded opt-in cylinder diagnostic; never writes manuscript data."""
from __future__ import annotations

import argparse
import csv
import copy
import faulthandler
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FORCE_PROTOCOL = "mac-brinkman-plus-normal-reaction-prototype-v1"
FINE_AUTHORIZATION = dict(date="2026-09-19", scope="single-segmented-monitored-verification",
                          gpu_preallocation_fraction=.70, gpu_memory_stop_mib=22000,
                          minimum_disk_free_gib=30)
FIXED_MODEL_FINE_AUTHORIZATION = dict(
    date="2026-09-19", scope="single-fixed-physical-model-880x164-control",
    gpu_preallocation_fraction=.70, gpu_memory_stop_mib=22000,
    minimum_disk_free_gib=30)
FIXED_MODEL_FINE_PARENT = "configs/forward_verification/diagnostics/cylinder_normal_continuum_medium.json"
FIXED_MODEL_FINE_PARENT_SHA256 = "11bf5d641732fc588bd97acadb07d316f62258934f73c64db9e8c709378a773a"


def validate_fixed_model_fine(config):
    """The separately approved control is not a reusable fine-grid permission."""
    control = config.get("fixed_model_control", {})
    if (config.get("fixed_model_fine_authorization") != FIXED_MODEL_FINE_AUTHORIZATION
            or "fine_authorization" in config
            or config.get("grid") != dict(nx=880, ny=164, lx=2.2, ly=.41)
            or control.get("reference_normal_config") != FIXED_MODEL_FINE_PARENT
            or control.get("reference_normal_config_sha256") != FIXED_MODEL_FINE_PARENT_SHA256
            or config.get("normal_constraint", {}).get("markers") != 48
            or config.get("pressure_preconditioner") != "constant-poisson-dct"):
        raise ValueError("fixed-model fine approval freezes medium physics, 48 markers and 880x164")
    _, parent = fixed_model_reference(config)
    if parent["grid"] != dict(nx=440, ny=82, lx=2.2, ly=.41):
        raise ValueError("fixed-model fine reference must be the frozen medium model")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if "execution_scope" in config:
        raise ValueError("FE-only physical-model source is not a GPU run configuration")
    if config.get("schema_version") != "normal-cylinder-diagnostic-config-1.0":
        raise ValueError("wrong constrained cylinder schema")
    if config.get("experiment_id") != "normal_constraint_cylinder_diagnostic" or config.get("study") != {"development_only": True, "paper_eligible": False}:
        raise ValueError("constrained cylinder is development-only")
    source_path = (ROOT/config["source_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or hashlib.sha256(source_path.read_bytes()).hexdigest() != config["source_config_sha256"]:
        raise ValueError("source configuration hash mismatch")
    source = json.loads(source_path.read_text())
    if "execution_scope" in source:
        raise ValueError("FE-only physical-model source is not a GPU run configuration")
    fine = "fine_authorization" in config
    fixed_fine = "fixed_model_fine_authorization" in config
    if fixed_fine:
        validate_fixed_model_fine(config)
    if fine and (config["fine_authorization"] != FINE_AUTHORIZATION or config.get("grid") != dict(nx=880,ny=164,lx=2.2,ly=.41)):
        raise ValueError("fine authorization is limited to the one frozen 880x164 case")
    limit_x,limit_y,limit_markers = (880,164,128) if fine or fixed_fine else (440,82,64)
    resolved = resolve_physical_source(config, source)
    if config.get("grid") != resolved["grid"] or resolved["grid"]["nx"] > limit_x or resolved["grid"]["ny"] > limit_y:
        raise ValueError("grid must match the frozen bounded source")
    normal = config["normal_constraint"]
    if normal.get("force_protocol") != FORCE_PROTOCOL or normal.get("kernel") != "tensor-cubic-bspline-native-mac":
        raise ValueError("unrecognised normal constraint/force protocol")
    if type(normal.get("markers")) is not int or not 8 <= normal["markers"] <= limit_markers:
        raise ValueError("marker count outside explicit scope")
    if config.get("pressure_preconditioner", "none") not in ("none", "constant-poisson-dct"):
        raise ValueError("unrecognised pressure preconditioner")
    if fine and (normal["markers"] != 96 or config.get("pressure_preconditioner") != "constant-poisson-dct" or "planar_calibration_control" not in config):
        raise ValueError("fine verification freezes 96 markers and the continuum-model control")
    if fine and "fixed_model_control" in config:
        raise ValueError("original fine authority does not authorize a fixed-model control")
    if normal.get("nominal_surface_shift") != 0 or normal.get("angle_origin") != 0:
        raise ValueError("no reference-fitted surface shift/rotation allowed")
    if config.get("gates") != dict(divergence_max=1e-8, normal_velocity_max=1e-10, mass_flux_mismatch=1e-9, cv_relative_closure=1e-5):
        raise ValueError("conservation gates must not be relaxed")
    if type(config.get("max_steps")) is not int or not 1 <= config["max_steps"] <= resolved["solver"]["max_steps"]:
        raise ValueError("step cap outside original bounded scope")
    if type(config.get("chunk_steps")) is not int or not 1 <= config["chunk_steps"] <= 1000:
        raise ValueError("bounded checkpoint interval required")
    if config["max_steps"] % config["chunk_steps"]:
        raise ValueError("max_steps must be a multiple of chunk_steps")
    return config, source_path, source


def resolve_physical_source(config, source):
    """Opt-in consistency control, NOT a silently substituted MAC calibration.

    The original source still passes all production physical/stability checks.
    Only its mask shift is replaced by an independently integrated continuum
    planar root. The resolved source records both roots and the model change.
    """
    resolved = copy.deepcopy(source)
    if "planar_calibration_control" not in config:
        if "fixed_model_control" in config:
            raise ValueError("fixed-model control requires the existing continuous calibration reference")
        return resolved
    control = config["planar_calibration_control"]
    expected = dict(method="continuous-planar-tangential-shear-zero-displacement",
                    reference_data="none", purpose="same-continuum-model-discretization-only",
                    paper_eligible=False)
    if control != expected:
        raise ValueError("unapproved planar calibration control")
    from jax_cad.continuum_mask_calibration import continuum_planar_displacement
    grid, physics, ib = source["grid"], source["physics"], source["immersed_boundary"]
    h = max(grid["lx"]/grid["nx"], grid["ly"]/grid["ny"])
    q = (physics["viscosity"]/ib["alpha_max"])**.5/h
    if abs(q-.5) > 1e-12 or ib["epsilon_over_h"] != 1.5:
        raise ValueError("continuum control is frozen to q=.5, epsilon/h=1.5")
    calibration = continuum_planar_displacement(q, ib["epsilon_over_h"])
    resolved["immersed_boundary"]["mask_midpoint_shift_over_h"] = calibration["zero_displacement_shift_over_reference_h"]
    resolved["immersed_boundary"]["reference_free_calibration"] = dict(
        **control, integration=calibration,
        baseline_discrete_shift_over_h=ib["mask_midpoint_shift_over_h"],
        baseline_discrete_calibration=ib["reference_free_calibration"],
        satisfies_original_single_phase_discrete_zero=False)
    if "fixed_model_control" in config:
        resolved = refine_fixed_model(config, resolved)
    return resolved


def fixed_model_reference(config):
    control=config["fixed_model_control"]
    if set(control)!={"method","refinement_factor","reference_normal_config","reference_normal_config_sha256","paper_eligible"} or control["method"]!="fixed-physical-mask-kernel-and-markers" or type(control["refinement_factor"]) is not int or control["refinement_factor"]!=2 or control["paper_eligible"] is not False:
        raise ValueError("unapproved fixed-model refinement control")
    path=(ROOT/control["reference_normal_config"]).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=control["reference_normal_config_sha256"]:
        raise ValueError("fixed-model reference configuration hash mismatch")
    parent=json.loads(path.read_text())
    if any(key in parent for key in ("fixed_model_control", "execution_scope", "fine_authorization", "fixed_model_fine_authorization")):
        raise ValueError("fixed-model control requires an original coarse/medium GPU reference")
    for key in ("source_config","source_config_sha256","normal_constraint","planar_calibration_control","gates"):
        if config.get(key)!=parent.get(key):
            raise ValueError(f"fixed-model refinement must preserve reference {key}")
    return path,parent


def refine_fixed_model(config, resolved):
    """Change ONLY discretisation; preserve all physical lengths and reactions."""
    path,parent=fixed_model_reference(config)
    g=resolved["grid"]; h=max(g["lx"]/g["nx"],g["ly"]/g["ny"])
    if parent["grid"]!=g:
        raise ValueError("reference grid must identify the unchanged physical model")
    result=copy.deepcopy(resolved)
    result["grid"].update(nx=2*g["nx"],ny=2*g["ny"])
    ib=result["immersed_boundary"]
    for key in ("epsilon_over_h","mask_midpoint_shift_over_h","delta_b_over_h"):
        ib[key]*=2
    ib["alpha_max_h_squared_over_nu"]/=4
    result["solver"]["dt"]/=4
    result["solver"]["max_steps"]*=4
    phase=result["phase_lock"]
    phase["cylinder_centre_over_h"]=[2*x for x in phase["cylinder_centre_over_h"]]
    phase["cylinder_radius_over_h"]*=2
    measurement=result["surface_measurement"]
    for key in ("offsets_over_h","offset_sensitivity_shifts_over_h"):
        measurement[key]=[2*x for x in measurement[key]]
    result["fixed_model_resolution"]=dict(reference_grid=g,reference_h=h,kernel_ratio=2,
        kernel_widths=[g["lx"]/g["nx"],g["ly"]/g["ny"]],reference_normal_config=str(path),
        physical_model_unchanged=True,calibration_reference_h=h,
        note="calibration metadata uses original physical h; *_over_h values use refined numerical h")
    return result


def write_checkpoint(output, fields, *, completed, initial_rms, threshold, model):
    """Replace only this run's checkpoint atomically, retaining restart metadata."""
    import numpy as np
    temporary = Path(output)/"fields.next.npz"
    fingerprint = hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()
    np.savez_compressed(temporary, **fields,
        checkpoint_schema=np.asarray("normal-response-checkpoint-1.0"),
        completed_iterations=np.asarray(completed), initial_residual_rms=np.asarray(initial_rms),
        convergence_threshold=np.asarray(threshold), model_sha256=np.asarray(fingerprint))
    temporary.replace(Path(output)/"fields.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    config, source_path, source = load_config(args.config)
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise ValueError("need new external output directory")
    if os.environ.get("JAX_PLATFORMS") != "cuda" or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") != "0.70":
        raise RuntimeError("use the guarded GPU launcher with the fixed memory cap")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, str(ROOT))
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from jax_cad.benchmarks import DFG_2D1
    from jax_cad.differentiable.mac_channel import (
        ChannelParams, cell_centres, create_channel_grid, create_initial_state,
        interpolate_alpha_to_faces, residual_rms, steady_residual, flux_diagnostics,
    )
    from jax_cad.differentiable.mac_normal_constraint import NormalProjection, ellipse_markers, require_valid
    from jax_cad.differentiable.mac_normal_response import (
        build_normal_response, solve_normal_chunk, normal_sample, body_reaction_components,
        normal_control_volume_balance,
    )
    from jax_cad.differentiable.smooth_heaviside import smooth_heaviside
    from jax_cad.verification import MACStaggeredControlVolume, parabolic_inlet_face_averages, dfg_circle_exterior_trace
    from jax_cad.run_artifacts import write_json, artifact_record, no_gradient_record, peak_memory_record
    from capture_compute_environment import capture
    import dolfinx_dfg2d1 as evidence
    spec = importlib.util.spec_from_file_location("normal_diagnostic_base", ROOT/"examples/30_cylinder_dfg2d1.py")
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    base._load_config(source_path)  # retain original physical/stability checks
    source = resolve_physical_source(config, source)
    required = (Path(__file__).resolve(), ROOT/"jax_cad/differentiable/mac_normal_response.py",
                ROOT/"jax_cad/differentiable/mac_normal_constraint.py", args.config.resolve(), source_path)
    if "planar_calibration_control" in config:
        required += (ROOT/"jax_cad/continuum_mask_calibration.py",)
    if config.get("pressure_preconditioner", "none") != "none":
        required += (ROOT/"jax_cad/differentiable/mac_pressure_preconditioner.py", ROOT/"jax_cad/differentiable/mac_channel.py")
    if "fixed_model_control" in config:
        required += (fixed_model_reference(config)[0],)
    snapshot = evidence._git_snapshot(required)
    if snapshot["dirty"] is not False:
        raise RuntimeError("clean committed source required")
    output.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    environment = capture("float64")
    environment["source_snapshot"] = snapshot
    write_json(output/"environment.json", environment)
    write_json(output/"config.json", {**config, "resolved_base": source})
    # A native compilation/initialisation stall must not be an opaque empty run.
    # SIGUSR1 requests a stack dump without terminating this owned child. This
    # diagnostic does not change numerical kernels, tolerances or GPU settings.
    stack_file = (output/"python_stacks.log").open("x")
    faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
    progress_started = time.perf_counter()
    def progress(stage):
        write_json(output/"progress.json", dict(stage=stage, elapsed_seconds=time.perf_counter()-progress_started,
                                                pid=os.getpid(), git_commit=snapshot["commit"]))
    progress("construct-grid")
    g = source["grid"]; physics = source["physics"]; ib = source["immersed_boundary"]; s = source["solver"]
    grid = create_channel_grid(g["nx"], g["ny"], g["lx"], g["ly"])
    h = max(grid.dx, grid.dy)
    cx, cy = physics["cylinder_centre"]; radius = physics["cylinder_radius"]
    x, y = cell_centres(grid, dtype=jnp.float64)
    progress("construct-mask")
    alpha = ib["alpha_max"]*(1-smooth_heaviside(jnp.sqrt((x-cx)**2+(y-cy)**2)-radius-ib["mask_midpoint_shift_over_h"]*h, ib["epsilon_over_h"]*h))
    progress("construct-inlet")
    inlet = parabolic_inlet_face_averages(grid, physics["maximum_inlet_velocity"], dtype=jnp.float64)
    params = ChannelParams(viscosity=physics["viscosity"], dt=s["dt"], pressure_tolerance=s["pressure_tolerance"],
        pressure_atol=s["pressure_atol"], pressure_maxiter=s["pressure_maxiter"], include_advection=s["include_advection"],
        advection_scheme=s["advection_scheme"], nonlinear_rtol=s["nonlinear_rtol"], nonlinear_atol=s["nonlinear_atol"],
        nonlinear_min_steps=s["nonlinear_min_steps"], pressure_preconditioner=config.get("pressure_preconditioner", "none"))
    au, av = interpolate_alpha_to_faces(alpha, grid)
    mu, mv = 1/(1+params.dt*au), 1/(1+params.dt*av)
    pressure_preflight = None
    if params.pressure_preconditioner != "none":
        from jax_cad.differentiable.mac_pressure_preconditioner import manufactured_preflight
        progress("manufactured-pressure-preflight")
        check = jax.jit(lambda a,b: manufactured_preflight(a,b,grid,params))(mu,mv)
        jax.block_until_ready(check)
        pressure_preflight = dict(max_relative_solution_error=float(check[0]), max_fresh_relative_residual=float(check[1]))
        write_json(output/"pressure_preflight.json", pressure_preflight)
        if not all(np.isfinite(list(pressure_preflight.values()))) or max(pressure_preflight.values()) >= 1e-8:
            raise RuntimeError("same-operator manufactured pressure preflight failed")
    progress("construct-markers")
    allow_fine = "fine_authorization" in config or "fixed_model_fine_authorization" in config
    markers = ellipse_markers(jnp.array([cx, cy]), jnp.array([radius, radius]), 0., config["normal_constraint"]["markers"], allow_fine=allow_fine)
    prep_started = time.perf_counter()
    progress("compile-and-build-normal-response")
    # Stage primitive kernels and the pressure-column loop separately. This
    # preserves the equations while exposing the precise slow compilation site.
    kernel_ratio=source.get("fixed_model_resolution",{}).get("kernel_ratio",1)
    response = build_normal_response(markers, mu, mv, grid, params, column_batching="map", allow_fine=allow_fine,kernel_ratio=kernel_ratio)
    jax.block_until_ready(response)
    preparation_seconds = time.perf_counter()-prep_started
    prep = dict(column_batching="map", preparation_staging="separate-kernels", valid=bool(response.valid), relative_minimum_eigenvalue=float(response.relative_minimum_eigenvalue),
                kernel_ratio=kernel_ratio,kernel_widths=[kernel_ratio*grid.dx,kernel_ratio*grid.dy],
                pressure_preconditioner=params.pressure_preconditioner, manufactured_pressure_preflight=pressure_preflight,
                relative_asymmetry=float(response.relative_asymmetry), pressure_residual_relative_max=float(response.pressure_residual_relative_max),
                basis_divergence_max=float(response.basis_divergence_max), preparation_seconds=preparation_seconds,
                basis_bytes=sum(a.size*a.dtype.itemsize for a in (response.velocity_u, response.velocity_v, response.pressure_impulse)))
    write_json(output/"preparation.json", prep)
    progress("response-prepared")
    print(json.dumps(dict(preparation=prep)), flush=True)
    if not response.valid:
        raise RuntimeError("projected boundary response failed rank/pressure/divergence gate")
    state = create_initial_state(grid, inlet, dtype=jnp.float64)
    result = NormalProjection(state, jnp.zeros(len(markers.weights)), jnp.zeros_like(state.u), jnp.zeros_like(state.v),
        normal_sample(state.u, state.v, response.stencil), jnp.array(True), response.relative_minimum_eigenvalue, jnp.array(0.), jnp.array(0.))
    rms0 = float(residual_rms(steady_residual(state, alpha, inlet, grid, params)))
    threshold = max(params.nonlinear_atol, params.nonlinear_rtol*rms0)
    kernel = jax.jit(lambda r, a, i, b, t, n: solve_normal_chunk(r, a, i, grid, params, b, config["chunk_steps"], t, n))
    completed = 0; chunks = []; metrics_row = None
    solve_started = time.perf_counter()
    history_path = output/"residual_history.csv"
    with history_path.open("x", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["iteration", "residual_rms", "residual_max", "divergence_rms", "divergence_max", "normal_velocity_max", "mass_flux_mismatch", "constraint_backward_error", "stationarity_max"])
        while completed < config["max_steps"]:
            if config["max_steps"]-completed < config["chunk_steps"]:
                raise ValueError("max_steps must be a multiple of chunk_steps")
            t0 = time.perf_counter()
            progress(f"solve-chunk-after-{completed}")
            result, history, count, metrics_row = kernel(result, alpha, inlet, response, threshold, completed)
            jax.block_until_ready(result)
            count = int(count); duration = time.perf_counter()-t0
            require_valid(result)
            rows = np.asarray(history)[:count]
            if not np.isfinite(rows).all():
                raise RuntimeError("nonfinite constrained residual history")
            writer.writerows([completed+i+1, *row.tolist()] for i, row in enumerate(rows)); history_file.flush()
            completed += count
            forces = body_reaction_components(result, alpha, grid)
            coefficient_scale = 2/(physics["mean_inlet_velocity"]**2*2*radius)
            row = dict(iterations=completed, residual_rms=float(metrics_row[0]), divergence_max=float(metrics_row[3]),
                       normal_velocity_max=float(metrics_row[4]), total_drag=float(forces["total"][0]*coefficient_scale),
                       total_lift=float(forces["total"][1]*coefficient_scale), seconds=duration)
            chunks.append(row)
            print(json.dumps(row), flush=True)
            write_checkpoint(output, dict(u_staggered=np.asarray(result.state.u), v_staggered=np.asarray(result.state.v),
                                p=np.asarray(result.state.p), alpha=np.asarray(alpha), reaction_u=np.asarray(result.reaction_u),
                                reaction_v=np.asarray(result.reaction_v), normal_traction=np.asarray(result.normal_traction),
                                marker_points=np.asarray(markers.points), marker_normals=np.asarray(markers.normals), marker_weights=np.asarray(markers.weights)),
                             completed=completed, initial_rms=rms0, threshold=threshold,
                             model=dict(resolved_base=source, normal_constraint=config["normal_constraint"], pressure_preconditioner=params.pressure_preconditioner))
            if completed >= params.nonlinear_min_steps and float(metrics_row[0]) <= threshold:
                break
            if count == 0:
                raise RuntimeError("constrained solve made no progress")
    solve_seconds = time.perf_counter()-solve_started
    progress("postprocess")
    state = result.state
    forces = body_reaction_components(result, alpha, grid)
    # Control volumes contain the UNION of mask support and all constraint kernels.
    supports = [np.asarray(au) != 0, np.asarray(av) != 0]
    for support, indices, weights in zip(supports, (response.stencil.indices_u, response.stencil.indices_v), (response.stencil.weights_u, response.stencil.weights_v)):
        support.ravel()[np.asarray(indices)[np.asarray(weights) != 0]] = True
    bounds = []
    for support in supports:
        indices = np.argwhere(support)
        bounds.append((indices.min(axis=0), indices.max(axis=0)))
    balances = []
    for margin in (1, 2, 3):
        values = []
        for low, high in bounds:
            values.extend([int(low[0]-margin), int(high[0]+margin), int(low[1]-margin), int(high[1]+margin)])
        cv = MACStaggeredControlVolume(*values)
        balance = normal_control_volume_balance(result, alpha, inlet, grid, params, cv)
        entry = {key: np.asarray(value).tolist() for key, value in balance.items()}
        entry.update(margin=margin, bounds=values, relative_closure=float(jnp.linalg.norm(balance["inferred_total"]-balance["captured_total"])/jnp.maximum(jnp.linalg.norm(balance["global_total"]), 1.e-30)))
        balances.append(entry)
    measurement = source["surface_measurement"]
    trace = dfg_circle_exterior_trace(state, grid, cx, cy, radius, h, jnp.asarray(measurement["offsets_over_h"]),
        quadrature_points=measurement["quadrature_points"], pressure_degree=measurement["pressure_polynomial_degree"],
        tangential_velocity_degree=measurement["tangential_velocity_polynomial_degree"],
        minimum_fluid_signed_distance=max(2*kernel_ratio*h, (ib["epsilon_over_h"]+ib["mask_midpoint_shift_over_h"])*h))
    # theta=0 is rear, theta=pi is front, matching the original trace convention.
    dp = float(trace.pressure[len(trace.pressure)//2]-trace.pressure[0])
    scale = 2/(physics["mean_inlet_velocity"]**2*2*radius)
    numerical = dict(total_drag_coefficient=float(forces["total"][0]*scale), total_lift_coefficient=float(forces["total"][1]*scale),
                     brinkman_drag_component=float(forces["brinkman"][0]*scale), brinkman_lift_component=float(forces["brinkman"][1]*scale),
                     constraint_drag_component=float(forces["constraint"][0]*scale), constraint_lift_component=float(forces["constraint"][1]*scale),
                     pressure_difference=dp)
    errors = {name: abs(numerical[key]/DFG_2D1["reference_"+name]-1) for name, key in (("drag_coefficient", "total_drag_coefficient"), ("lift_coefficient", "total_lift_coefficient"), ("pressure_difference", "pressure_difference"))}
    divrms, divmax, influx, outflux, mismatch = map(float, flux_diagnostics(state, grid))
    converged = float(metrics_row[0]) <= threshold and completed >= params.nonlinear_min_steps
    gates = config["gates"]
    checks = dict(primal_converged=converged, divergence=divmax < gates["divergence_max"], normal=float(metrics_row[4]) < gates["normal_velocity_max"],
                  mass_flux=mismatch < gates["mass_flux_mismatch"], cv=max(b["relative_closure"] for b in balances) < gates["cv_relative_closure"],
                  algebraic_cv=max(abs(x) for b in balances for x in b["algebraic_closure"]) < 1.e-12,
                  complete_reaction_support=max(abs(x) for b in balances for x in b["uncaptured_total"]) < 1.e-12)
    stability = base._stability_metrics(state, grid, params, ib["alpha_max"], limits=s["stability_limits"])
    checks["stability"] = stability["all_within_configured_limits"] is True
    metrics = dict(force_protocol=FORCE_PROTOCOL, preparation=prep, numerical=numerical, relative_errors=errors,
                   convergence=dict(converged=converged, iterations=completed, initial_residual_rms=rms0, threshold=threshold, final_residual_rms=float(metrics_row[0]), chunks=chunks),
                   conservation=dict(divergence_rms=divrms, divergence_max=divmax, inlet_flux=influx, outlet_flux=outflux, mass_flux_mismatch=mismatch,
                                     normal_velocity_max=float(metrics_row[4])), control_volumes=balances, checks=checks, stability=stability,
                   paper_eligible=False, sharp_cylinder_acceptance=False)
    write_json(output/"metrics.json", metrics)
    progress("complete")
    faulthandler.unregister(signal.SIGUSR1)
    stack_file.close()
    if evidence._git_snapshot(required) != snapshot:
        raise RuntimeError("source changed during constrained cylinder solve")
    success = all(checks.values())
    peak_host, peak_gpu, memory_method = peak_memory_record()
    record = dict(schema_version="1.0", experiment_id=config["experiment_id"], run_id=config["run_id"], git=environment["git"],
        command=dict(argv=sys.argv, working_directory=str(ROOT)), timestamps=dict(start_utc=started_utc, end_utc=datetime.now(timezone.utc).isoformat(), status="success" if success else "partial", exit_code=0 if success else 1),
        configuration={**config, "resolved_base": source, "requires_gradient": False,
          "reference_comparison": dict(boundary_conditions_exact=False, quantity_definition_exact=False, paper_eligible=False,
             reason="Normal interpolation constraints plus finite tangential penalisation; new total-reaction protocol requires independent and mesh validation.")},
        hardware=environment["hardware"], software=environment["software"],
        primal=dict(converged=converged, iterations=completed, residual_norm=float(metrics_row[0]), divergence_rms=divrms, divergence_max=divmax,
                    mass_flux_mismatch=mismatch, failure_reason=None if converged else "nonlinear step cap before tolerance"),
        objective=dict(name="relative_total_reaction_drag_error", value=errors["drag_coefficient"], units="dimensionless", normalization="DFG reference C_D=5.57953523384"),
        gradient=no_gradient_record("Forward diagnostic; separate small-grid geometry-gradient audits, not a cylinder gradient claim."),
        timing=dict(compile_seconds=preparation_seconds+chunks[0]["seconds"], forward_samples_seconds=[solve_seconds], gradient_samples_seconds=[], optimizer_step_samples_seconds=[]),
        memory=dict(peak_host_bytes=peak_host, peak_accelerator_bytes=peak_gpu, measurement_method=memory_method), artifacts=[],
        notes="Opt-in normal constraint, full Brinkman plus multiplier reaction, no paper promotion; success means numerical guards passed, not sharp DFG acceptance.")
    for filename, media, role in (("config.json", "application/json", "config"), ("environment.json", "application/json", "environment"),
                                 ("preparation.json", "application/json", "table"), ("metrics.json", "application/json", "table"),
                                 ("progress.json", "application/json", "other"), ("python_stacks.log", "text/plain", "other"),
                                 ("fields.npz", "application/x-npz", "field"), ("residual_history.csv", "text/csv", "residual-history")):
        record["artifacts"].append(artifact_record(output/filename, output, media_type=media, role=role))
    if pressure_preflight is not None:
        record["artifacts"].append(artifact_record(output/"pressure_preflight.json", output, media_type="application/json", role="table"))
    write_json(output/"result.json", record)
    print(json.dumps(dict(final=numerical, relative_errors=errors, checks=checks)), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
