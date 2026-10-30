#!/usr/bin/env python3
"""Read-only audit of archived wake histories, geometry pullback and guard hashes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def audit(directory):
    config=json.loads((directory/"config.json").read_text())
    result=json.loads((directory/"result.json").read_text())
    guard_dir=directory.with_name(directory.name+".guard")
    guard=json.loads((guard_dir/"guard.json").read_text())
    assert guard["exit_code"]==0 and guard["stop_reason"] is None and guard["source_unchanged"]
    assert result["paper_eligible"] is False
    for relative,digest in json.loads((guard_dir/"artifacts.json").read_text()).items():
        assert hashlib.sha256((directory.parent/relative).read_bytes()).hexdigest()==digest,relative
    g=config["grid"]; dx=g["lx"]/g["nx"]; dy=g["ly"]/g["ny"]
    x,y=np.meshgrid((np.arange(g["nx"])+.5)*dx,(np.arange(g["ny"])+.5)*dy,indexing="ij")
    angle=np.arctan2(y-config["center"][1],x-config["center"][0])
    quadrature=2*np.pi*(np.arange(1024)+.5)/1024
    basis=np.array([np.cos(2*quadrature),np.cos(4*quadrature)])
    max_pullback_error=0.; pullback_checks=0
    for record in result["evaluations"]:
        d=directory/record["directory"]
        h=np.genfromtxt(d/"history.csv",delimiter=",",names=True)
        assert len(h)==config["n_steps"] and np.isfinite(h.tolist()).all()
        np.testing.assert_allclose(h["time"],np.arange(1,len(h)+1)*config["dt"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(h["Cd"][config["window_start_step"]:].mean(),record["objective"],rtol=1e-12)
        assert math.isclose(record["geometric_area"],np.pi*config["radius"]**2,rel_tol=1e-12)
        assert h["divergence_max"].max()<1e-7 and h["flux_mismatch"].max()<1e-8 and h["pressure_residual"].max()<1e-8
        with np.load(d/"fields.npz") as z:
            div=np.diff(z["u"],axis=1)/dx+np.diff(z["v"],axis=2)/dy
            assert np.max(abs(div))<1e-7
            np.testing.assert_allclose(z["times"],h["time"][z["steps"]-1],atol=1e-12)
            if "level_set_cotangent" in z:
                pullback_checks+=1
                # Independent analytic chain rule, not another call to jax.vjp.
                theta=np.array(record["theta"]); exponential=np.exp(2*theta@basis)
                normal=.5*np.log(exponential.mean())
                normal_derivative=(basis*exponential).mean(axis=1)/exponential.mean()
                local_basis=np.array([np.cos(2*angle),np.cos(4*angle)])
                radius=config["radius"]*np.exp(np.einsum("i,ijk->jk",theta,local_basis)-normal)
                dphi=-radius*(local_basis-normal_derivative[:,None,None])
                analytic=np.sum(dphi*z["level_set_cotangent"],axis=(1,2))
                np.testing.assert_allclose(analytic,record["gradient"],rtol=1e-10,atol=1e-10)
                max_pullback_error=max(max_pullback_error,float(np.max(abs(analytic-record["gradient"]))))
    if "gradient_audit" in result:
        t=result["gradient_audit"]
        rates=np.log(np.array(t["remainders"][:-1])/t["remainders"][1:])/np.log(np.array(t["epsilons"][:-1])/t["epsilons"][1:])
        np.testing.assert_allclose(rates,t["rates"],rtol=1e-12)
        assert t["passed"] and min(rates[-2:])>1.8 and t["relative_difference"]<.005
    if "optimization" in result:
        opt=result["optimization"]
        for record in opt["trace"]:
            assert np.max(np.abs(record["coefficients"]))<=.22+1e-12
            if record["accepted"]:
                assert record["realised_change"]<=1e-4*record["linear_prediction"]<0
        assert np.max(np.abs(opt["coefficients"]))<=.22+1e-12
        assert opt["final_value"]<result["evaluations"][0]["objective"]
    return dict(directory=str(directory),passed=True,evaluations=len(result["evaluations"]),
                analytic_geometry_pullback_checks=pullback_checks,
                analytic_geometry_pullback_max_error=max_pullback_error if pullback_checks else None,paper_eligible=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("directories",type=Path,nargs="+")
    p.add_argument("--output",type=Path,help="optional new JSON audit report")
    args=p.parse_args()
    if args.output and args.output.exists(): raise FileExistsError("new report path required")
    reports=[]
    for directory in args.directories:
        report=audit(directory); reports.append(report); print(json.dumps(report),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=True,paper_eligible=False,reports=reports),indent=2)+"\n")
