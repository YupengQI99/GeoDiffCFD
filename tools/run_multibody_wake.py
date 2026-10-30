#!/usr/bin/env python3
"""Captured three-body DEVELOPMENT driver; launch through guard_multibody_wake."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multibody_wake_config import validate_config, envelope_report
from multibody_wake_continuation import optimization_context


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",required=True)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); output=a.output_dir
    if os.environ.get("JAX_PLATFORMS")!="cuda" or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")!="0.70":
        raise RuntimeError("captured CUDA supervisor required")
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    c=json.loads(a.config.read_text()); validate_config(c)
    context=optimization_context(c,ROOT) if a.mode=="optimize" else None
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_cad.run_artifacts import provenance_record
    provenance=provenance_record()
    if jax.default_backend()!="gpu": raise RuntimeError("CUDA required")
    write_json(output/"environment.json",provenance); write_json(output/"config.json",c)
    write_json(output/"geometry_envelope.json",envelope_report(c))
    if a.mode=="audit":
        import pytest
        return int(pytest.main(["-q","-s","tests/test_multibody_wake.py","--junitxml",str(output/"junit.xml")]))
    from jax_cad.differentiable.mac_channel import ChannelParams,create_channel_grid,parabolic_inlet
    from jax_cad.differentiable.wake_design import initial_flow,make_wake_kernels,evaluate_trajectory
    from jax_cad.differentiable.multibody_wake import geometry_fields,body_level_set_cotangent
    from jax_cad.unified.geometry import fourier_shape_area
    from jax_cad.optimization.projected_descent import projected_gradient_descent,projected_gradient
    grid=create_channel_grid(**c["grid"])
    params=ChannelParams(c["viscosity"],c["dt"],include_advection=True,advection_scheme=c["advection_scheme"],
                        pressure_tolerance=1e-11,pressure_atol=1e-13,pressure_maxiter=1000,
                        pressure_preconditioner="constant-poisson-dct")
    inlet=parabolic_inlet(grid,c["mean_inlet"],dtype=jnp.float64); state0=initial_flow(grid,inlet)
    kernels=make_wake_kernels(grid,params,chunk_steps=c["chunk_steps"],force_scale=2/(c["mean_inlet"]**2*c["reference_diameter"]))
    fields=lambda z:geometry_fields(z,grid,c)
    material=lambda z:fields(z)[0].momentum_resistance
    evaluations=[]
    def evaluate(theta,gradient=False):
        index=len(evaluations); label=f"evaluation-{index:03d}"; directory=output/label; directory.mkdir()
        z=jnp.asarray(theta,dtype=jnp.float64)
        if not np.isfinite(z).all() or np.max(np.abs(z))>1.05: raise ValueError("geometry outside certified probe box")
        started=time.monotonic(); alpha,pullback=jax.vjp(material,z); last=[-1]
        def progress(stage,step,value,row):
            if step in (0,c["n_steps"]) or abs(step-last[0])>=max(c["chunk_steps"],c["n_steps"]//12):
                event=dict(evaluation=index,stage=stage,step=step,total_steps=c["n_steps"],
                           accumulated_objective=value,elapsed_seconds=time.monotonic()-started)
                print(json.dumps(event),flush=True); write_json(output/"progress.json",event); last[0]=step
        result=evaluate_trajectory(alpha,inlet,state0,kernels,n_steps=c["n_steps"],chunk_steps=c["chunk_steps"],
                                   window_start_step=c["window_start_step"],gradient=gradient,progress=progress)
        grad=np.asarray(pullback(result["material_gradient"])[0]) if gradient else None
        if index==0 and gradient and context and context["parent"]:
            np.testing.assert_allclose(result["value"],context["parent"]["value"],rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(grad,context["parent"]["gradient"],rtol=1e-8,atol=1e-10)
        q,phi,chi,coefficients=fields(z)
        areas=[float(fourier_shape_area(coefficients[i],base_radius=b["radius"],modes=c["geometry"]["modes"],n_quadrature=4096))
               for i,b in enumerate(c["geometry"]["bodies"])]
        references=[np.pi*b["radius"]**2 for b in c["geometry"]["bodies"]]
        np.testing.assert_allclose(areas,references,rtol=1e-12,atol=1e-14)
        record=dict(theta=np.asarray(z).tolist(),body_coefficients=np.asarray(coefficients).tolist(),
                    objective=result["value"],gradient=None if grad is None else grad.tolist(),
                    geometric_areas=areas,reference_areas=references,
                    diffuse_areas=np.asarray(jnp.sum(1-chi,axis=(1,2))*grid.dx*grid.dy).tolist(),
                    diffuse_area=float(jnp.sum(1-q.fluid_fraction)*grid.dx*grid.dy),
                    seconds=time.monotonic()-started,max_diagnostics=result["history"].max(axis=0).tolist(),
                    paper_eligible=False,directory=label)
        write_json(directory/"metrics.json",record)
        with (directory/"history.csv").open("w",newline="") as f:
            writer=csv.writer(f); writer.writerow(["time","Cd","Cl","pressure_residual","divergence_max","flux_mismatch","cfl","max_velocity"])
            for i,row in enumerate(result["history"]): writer.writerow([(i+1)*c["dt"],*row])
        sensitivity={}
        if gradient:
            sensitivity=dict(material_cotangent=np.asarray(result["material_gradient"]),
                body_level_set_cotangent=np.asarray(body_level_set_cotangent(result["material_gradient"],phi,chi,c)))
        samples=result["snapshots"]
        np.savez_compressed(directory/"fields.npz",steps=np.array([s[0] for s in samples]),times=np.array([s[0]*c["dt"] for s in samples]),
            u=np.stack([s[1].u for s in samples]),v=np.stack([s[1].v for s in samples]),p=np.stack([s[1].p for s in samples]),
            phi=np.asarray(q.level_set),body_phi=np.asarray(phi),body_chi=np.asarray(chi),alpha=np.asarray(alpha),**sensitivity)
        evaluations.append(record); write_json(output/"evaluations.json",evaluations)
        print(json.dumps(dict(completed_evaluation=index,**record)),flush=True)
        return result["value"],grad

    theta=np.asarray(c["theta"],dtype=float)
    summary=dict(schema="multibody-wake-result-1",mode=a.mode,provenance=provenance,paper_eligible=False,
        scope="three-body finite-Brinkman finite-time native shape design; not sharp-wall acceptance")
    if a.mode=="forward": summary["value"]=evaluate(theta)[0]
    elif a.mode=="gradient":
        value,gradient=evaluate(theta,True)
        direction=np.asarray(c["taylor_direction"],dtype=float); direction/=np.linalg.norm(direction)
        eps=np.asarray(c.get("taylor_steps",[.008,.004,.002,.001])); errors=[]
        for e in eps:
            v,_=evaluate(theta+e*direction); errors.append(abs(v-value-e*float(gradient@direction)))
        rates=np.log(np.array(errors[:-1])/errors[1:])/np.log(eps[:-1]/eps[1:])
        fd_step=float(c.get("fd_step",1e-4)); checks=[]
        directions=[("all_bodies",direction)]
        if c.get("per_body_fd",True):
            for body in range(3):
                d=np.zeros(12); d[4*body:4*body+4]=direction[4*body:4*body+4]; d/=np.linalg.norm(d)
                directions.append((f"body_{body+1}",d))
        for label,d in directions:
            plus,_=evaluate(theta+fd_step*d); minus,_=evaluate(theta-fd_step*d)
            fd=(plus-minus)/(2*fd_step); ad=float(gradient@d)
            checks.append(dict(label=label,direction=d.tolist(),directional_ad=ad,directional_fd=fd,
                relative_difference=abs(fd-ad)/max(abs(fd),abs(ad),1e-8)))
        passed=bool(min(rates[-2:])>1.8 and max(r["relative_difference"] for r in checks)<.005)
        summary["gradient_audit"]=dict(value=value,gradient=gradient.tolist(),direction=direction.tolist(),epsilons=eps.tolist(),
            remainders=errors,rates=rates.tolist(),fd_step=fd_step,finite_difference_checks=checks,passed=passed)
        if not passed:
            summary["evaluations"]=evaluations; write_json(output/"result.json",summary)
            raise RuntimeError("full-trajectory twelve-variable gradient gate failed")
    elif a.mode=="optimize":
        opt=projected_gradient_descent(lambda z:evaluate(z,True),lambda z:evaluate(z)[0],theta,
            lower=-1.,upper=1.,initial_step=context["initial_step"],max_iterations=c["max_optimization_steps"],max_backtracks=3,
            relative_gradient_tolerance=.02,reference_gradient_norm=context["reference_gradient_norm"])
        value,gradient=evaluate(opt.coefficients,True)
        reference=context["reference_gradient_norm"]
        if reference is None: reference=opt.initial_projected_gradient_norm
        final_pg=float(np.linalg.norm(projected_gradient(opt.coefficients,gradient,-1.,1.)))
        converged=bool(final_pg<=.02*reference)
        summary["optimization"]=dict(coefficients=opt.coefficients.tolist(),trace=opt.trace,
            converged=converged,termination="projected_gradient_tolerance" if converged else opt.termination,
            optimizer_loop_termination=opt.termination,final_value=value,final_gradient=gradient.tolist(),
            final_projected_gradient_norm=final_pg,reference_gradient_norm=reference,
            relative_gradient_tolerance=.02,absolute_gradient_tolerance=.02*reference,
            initial_step=context["initial_step"],accepted_steps_before=context["accepted_steps_before"],
            total_accepted_steps=context["accepted_steps_before"]+sum(t["accepted"] for t in opt.trace),
            continuation=c.get("continuation"))
    else: raise ValueError("unknown mode")
    summary["evaluations"]=evaluations; write_json(output/"result.json",summary)
    return 0


if __name__=="__main__": raise SystemExit(main())
