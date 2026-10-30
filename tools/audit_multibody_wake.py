#!/usr/bin/env python3
"""Audit archived multibody geometry, native forces and full transient derivatives."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory, *, calibrated_offset=False):
    c=json.loads((directory/"config.json").read_text()); r=json.loads((directory/"result.json").read_text())
    if not calibrated_offset and (c.get('wall_shift',0) or c.get('normalize_level_set',False)):
        raise ValueError('corrected mapping requires the separately registered sensitivity auditor')
    guard=directory.with_name(directory.name+".guard")
    safety=json.loads((guard/"guard.json").read_text())
    assert safety["source_unchanged"] and safety["exit_code"]==0 and safety["stop_reason"] is None
    assert r["paper_eligible"] is False and r["provenance"]["backend"]=="gpu"
    for name,digest in json.loads((guard/"artifacts.json").read_text()).items():
        assert hashlib.sha256((directory.parent/name).read_bytes()).hexdigest()==digest,name
    geom=c["geometry"]; grid=c["grid"]; dx=grid["lx"]/grid["nx"]; dy=grid["ly"]/grid["ny"]
    x,y=np.meshgrid((np.arange(grid["nx"])+.5)*dx,(np.arange(grid["ny"])+.5)*dy,indexing="ij")
    quadrature=2*np.pi*(np.arange(1024)+.5)/1024
    basis=np.stack([np.cos(m*quadrature) for m in geom["modes"]]+[np.sin(m*quadrature) for m in geom["modes"]])
    fine_angle=2*np.pi*(np.arange(8192)+.5)/8192
    fine_basis=np.stack([np.cos(m*fine_angle) for m in geom["modes"]]+[np.sin(m*fine_angle) for m in geom["modes"]])
    pullback_error=0.; pullback_count=0; max_force_error=0.
    for e in r["evaluations"]:
        path=directory/e["directory"]; h=np.genfromtxt(path/"history.csv",delimiter=",",names=True)
        assert len(h)==c["n_steps"] and np.isfinite(h.tolist()).all()
        np.testing.assert_allclose(h["time"],np.arange(1,len(h)+1)*c["dt"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(h["Cd"][c["window_start_step"]:].mean(),e["objective"],rtol=1e-12)
        assert h["divergence_max"].max()<1e-7 and h["pressure_residual"].max()<1e-8 and h["flux_mismatch"].max()<1e-8
        z=np.array(e["theta"]).reshape(3,4); assert np.max(abs(z))<=1.05
        coeff=np.array([b["base_coefficients"] for b in geom["bodies"]],dtype=float)
        coeff[:,[0,3,1,4]]+=z*geom["design_scales"]
        np.testing.assert_allclose(coeff,e["body_coefficients"],atol=1e-14)
        phis=[]; effective_phis=[]; jacobians=[]; areas=[]
        for i,b in enumerate(geom["bodies"]):
            angle=np.arctan2(y-b["center"][1],x-b["center"][0])
            local=np.stack([np.cos(m*angle) for m in geom["modes"]]+[np.sin(m*angle) for m in geom["modes"]])
            exponent=np.exp(2*coeff[i]@basis); normal=.5*np.log(exponent.mean())
            radius=b["radius"]*np.exp(np.einsum("i,ijk->jk",coeff[i],local)-normal)
            phis.append(np.hypot(x-b["center"][0],y-b["center"][1])-radius)
            if calibrated_offset:
                fp=sum(m*(-coeff[i,k]*np.sin(m*angle)+coeff[i,k+3]*np.cos(m*angle)) for k,m in enumerate(geom['modes']))
                effective_phis.append(phis[-1]/np.sqrt(1+fp**2)+c['wall_shift'] if c['normalize_level_set'] else phis[-1]+c['wall_shift'])
            dc=(basis*exponent).mean(axis=1)/exponent.mean()
            jacobians.append((-radius*(local-dc[:,None,None]))[[0,3,1,4]]*np.array(geom["design_scales"])[:,None,None])
            areas.append(np.pi*np.mean((b["radius"]*np.exp(coeff[i]@fine_basis-normal))**2))
        reference=np.array([np.pi*b["radius"]**2 for b in geom["bodies"]])
        np.testing.assert_allclose(areas,reference,atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(areas,e["geometric_areas"],atol=1e-14,rtol=1e-12)
        phi=np.array(phis); effective=np.array(effective_phis) if calibrated_offset else phi
        s=np.clip(effective/c["epsilon"],-1,1)
        chi=.5+s*(15/16+s*s*(-5/8+s*s*3/16)); alpha=c["alpha_max"]*(1-np.prod(chi,axis=0))
        assert np.max(np.sum(effective<c["epsilon"],axis=0))==1
        np.testing.assert_allclose(np.sum(1-chi,axis=(1,2))*dx*dy,e["diffuse_areas"],atol=1e-12)
        with np.load(path/"fields.npz") as f:
            assert all(np.isfinite(f[k]).all() for k in f.files)
            np.testing.assert_allclose(f["body_phi"],phi,atol=1e-13,rtol=1e-12)
            np.testing.assert_allclose(f["body_chi"],chi,atol=1e-12)
            np.testing.assert_allclose(f["alpha"],alpha,atol=1e-10)
            np.testing.assert_allclose(f["phi"],effective.min(axis=0),atol=1e-13)
            np.testing.assert_allclose(f["times"],h["time"][f["steps"]-1],atol=1e-12)
            divergence=np.diff(f["u"],axis=1)/dx+np.diff(f["v"],axis=2)/dy
            assert np.max(abs(divergence))<1e-7
            # Independently integrate the native face-centred reaction.
            au=np.concatenate([alpha[:1],.5*(alpha[:-1]+alpha[1:]),alpha[-1:]],axis=0)
            av=np.concatenate([alpha[:,:1],.5*(alpha[:,:-1]+alpha[:,1:]),alpha[:,-1:]],axis=1)
            au[[0,-1]]*=.5; av[:,[0,-1]]*=.5
            scale=2/(c["mean_inlet"]**2*c["reference_diameter"])*dx*dy
            for force,key,state in ((au,"Cd","u"),(av,"Cl","v")):
                value=np.sum(f[state]*force,axis=(1,2))*scale
                actual=h[key][f["steps"]-1]
                np.testing.assert_allclose(value,actual,rtol=1e-10,atol=1e-10)
                max_force_error=max(max_force_error,float(np.max(abs(value-actual))))
            if "material_cotangent" in f:
                assert not calibrated_offset, 'this registration is forward-only, not a gradient audit'
                d=phi/c["epsilon"]; delta=np.where(abs(d)<1,(15/16)*(1-d*d)**2/c["epsilon"],0)
                others=np.stack([np.prod(chi[[j for j in range(3) if j!=i]],axis=0) for i in range(3)])
                lam=-c["alpha_max"]*delta*others*f["material_cotangent"]
                np.testing.assert_allclose(lam,f["body_level_set_cotangent"],rtol=1e-9,atol=1e-10)
                gradient=np.sum(np.array(jacobians)*lam[:,None,:,:],axis=(2,3)).reshape(-1)
                np.testing.assert_allclose(gradient,e["gradient"],rtol=1e-9,atol=1e-9)
                pullback_error=max(pullback_error,float(np.max(abs(gradient-e["gradient"]))))
                pullback_count+=1
    if "gradient_audit" in r:
        a=r["gradient_audit"]; e=r["evaluations"]; eps=np.array(a["epsilons"]); grad=np.array(e[0]["gradient"])
        errors=abs(np.array([v["objective"] for v in e[1:5]])-e[0]["objective"]-eps*(grad@a["direction"]))
        np.testing.assert_allclose(errors,a["remainders"],atol=1e-12,rtol=1e-8)
        rates=np.log(errors[:-1]/errors[1:])/np.log(eps[:-1]/eps[1:]); assert min(rates[-2:])>1.8
        for j,t in enumerate(a["finite_difference_checks"]):
            ad=float(grad@t["direction"]); fd=(e[5+2*j]["objective"]-e[6+2*j]["objective"])/(2*a["fd_step"])
            assert abs(ad-fd)/max(abs(ad),abs(fd),1e-8)<.005
            np.testing.assert_allclose(fd,t["directional_fd"],rtol=1e-10,atol=1e-10)
        assert a["passed"]
    if "optimization" in r:
        opt=r["optimization"]
        assert np.max(np.abs(opt["coefficients"]))<=1+1e-12
        assert opt["final_value"]<r["evaluations"][0]["objective"]
        for t in opt["trace"]:
            assert np.max(np.abs(t["coefficients"]))<=1+1e-12
            if t["accepted"]: assert t["realised_change"]<=1e-4*t["linear_prediction"]<0
    return dict(directory=str(directory),passed=True,evaluations=len(r["evaluations"]),
                analytic_pullback_checks=pullback_count,analytic_pullback_max_error=pullback_error if pullback_count else None,
                raw_field_force_max_error=max_force_error,paper_eligible=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("directories",type=Path,nargs="+"); p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.output and args.output.exists(): raise FileExistsError("new report required")
    records=[]
    for directory in args.directories:
        record=audit(directory); records.append(record); print(json.dumps(record),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=True,paper_eligible=False,records=records),indent=2)+"\n")
