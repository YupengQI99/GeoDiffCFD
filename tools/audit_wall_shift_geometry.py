"""CPU-only fixed-design geometry prerequisite; a negative result blocks GPU.

The T1 geometry formula below was confirmed in the user's CPU specification.
The user replaced the initial proposed band gate with a near-interface gate
and a frozen np.gradient endpoint comparison; the full band is report-only.
No flow solve, output selection, reference-force fitting or GPU launch occurs.
"""
import copy
import json
import math
from pathlib import Path
if __name__ == '__main__':
    import os
    import resource
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OPENBLAS_NUM_THREADS')!='1':
        raise RuntimeError('CPU-only, single-thread numerical environment required')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
import numpy as np
from multibody_penalty_config import ROOT, ANCHOR, ANCHOR_SHA, FINAL, FINAL_SHA, pinned

WALL_SHIFT=.0031020
EPSILON=.0075
FD_REFERENCES={
    'initial': [((.9877,1.0135,134),(.9918,1.0549,136),(.9603,1.1656,762)),
                ((.9819,1.0127,152),(.9817,1.0725,124),(.9397,1.2605,800)),
                ((.9882,1.0137,137),(.9936,1.0503,133),(.9603,1.1539,769))],
    # Final body 1's missing band entry was separately confirmed by the user.
    'final10': [((.9858,1.0185,138),(.9876,1.0706,130),(.9405,1.2477,764)),
                ((.9733,1.0356,145),(.9461,1.1278,135),(.8587,1.4019,815)),
                ((.9851,1.0152,134),(.9934,1.0575,128),(.9584,1.1817,752))]}
FD_REGIONS=('nominal_interface','corrected_interface','corrected_band')


def reference_checks(shape, bodies):
    """Fail closed on missing references; distinguish mismatch from absence."""
    checks=[]
    for i,b in enumerate(bodies):
        regions=[]
        for key,expected in zip(FD_REGIONS,FD_REFERENCES[shape][i]):
            v=b['finite_difference'][key]
            difference=None if expected is None else max(abs(v['minimum']-expected[0]),abs(v['maximum']-expected[1]))
            regions.append(dict(region=key,observed=v,reference=expected,
                reference_available=expected is not None,maximum_endpoint_difference=difference,
                endpoint_gate=expected is not None and difference<=.01,
                exact_count_gate=expected is not None and v['count']==expected[2]))
        nominal=b['finite_difference']['nominal_interface']
        checks.append(dict(name=b['name'],nominal_interface_gate=nominal['minimum']>=.95 and nominal['maximum']<=1.05,
            reference_complete=all(r['reference_available'] for r in regions),regions=regions,
            reference_endpoint_gate=all(r['endpoint_gate'] for r in regions),
            reference_count_gate=all(r['exact_count_gate'] for r in regions)))
    return checks


def candidate_config(final=True,normalized=True,wall_shift=WALL_SHIFT):
    c=copy.deepcopy(pinned(ANCHOR,ANCHOR_SHA))
    f=pinned(FINAL,FINAL_SHA)['optimization']
    assert f['converged'] and f['total_accepted_steps']==10 and c['theta']==f['coefficients']
    c.update(grid=dict(nx=880,ny=164,lx=2.2,ly=.41),alpha_max=2560.,epsilon=EPSILON,
        wall_shift=wall_shift,normalize_level_set=normalized,dt=.000125,n_steps=48000,window_start_step=32000,
        theta=f['coefficients'] if final else [0.]*12)
    return c


def body_parameters(c):
    g=c['geometry'];coeff=np.array([b['base_coefficients'] for b in g['bodies']],float)
    coeff[:,[0,3,1,4]]+=np.array(c['theta']).reshape(3,4)*np.array(g['design_scales'])
    return coeff,np.array(g['modes'])


def quantities(r,t,coeff,m,R0):
    angles=2*np.pi*(np.arange(1024)+.5)/1024
    fq=np.sum(coeff[:3]*np.cos(angles[:,None]*m)+coeff[3:]*np.sin(angles[:,None]*m),axis=-1)
    log_scale=.5*np.log(np.exp(2*fq).mean())
    p=t[...,None]*m
    f=np.sum(coeff[:3]*np.cos(p)+coeff[3:]*np.sin(p),axis=-1)
    fp=np.sum(m*(-coeff[:3]*np.sin(p)+coeff[3:]*np.cos(p)),axis=-1)
    fpp=np.sum(-m*m*(coeff[:3]*np.cos(p)+coeff[3:]*np.sin(p)),axis=-1)
    R=R0*np.exp(f-log_scale);q=np.sqrt(1+fp*fp);phi=r-R
    dn=phi/q
    radial=1/q
    angular=-R*fp/q-phi*fp*fpp/q**3
    norm=np.sqrt(radial**2+(angular/r)**2)
    return dict(phi=phi,normalized=dn,gradient_norm=norm,radius=R,
        normalization=q,log_scale=float(log_scale),first=fp,second=fpp)


def geometry_audit(c):
    g=c['grid'];dx=g['lx']/g['nx'];dy=g['ly']/g['ny'];coeff,m=body_parameters(c)
    x,y=np.meshgrid((np.arange(g['nx'])+.5)*dx,(np.arange(g['ny'])+.5)*dy,indexing='ij')
    bodies=[];bounds=[];walls=[];pairs=[]
    for i,b in enumerate(c['geometry']['bodies']):
        r=np.hypot(x-b['center'][0],y-b['center'][1]);t=np.arctan2(y-b['center'][1],x-b['center'][0])
        a=quantities(r,t,coeff[i],m,b['radius']);bands={}
        for name,mask in [('nominal_band',abs(a['normalized'])<=c['epsilon']),
                          ('corrected_band',abs(a['normalized']+c['wall_shift'])<=c['epsilon'])]:
            v=a['gradient_norm'][mask]
            bands[name]=dict(count=int(mask.sum()),minimum=float(v.min()),maximum=float(v.max()),
                quantiles_1_5_50_95_99=np.quantile(v,[.01,.05,.5,.95,.99]).tolist(),
                fraction_outside_10_percent=float(np.mean(abs(v-1)>.1)),
                within_10_percent_diagnostic_only=bool(v.min()>=.9 and v.max()<=1.1))
        # Analytic conservative bound at THIS fixed design, including modified support.
        maximum_q=math.sqrt(1+sum(m*(abs(coeff[i,:3])+abs(coeff[i,3:])))**2) if c['normalize_level_set'] else 1.
        bound=b['radius']*math.exp(sum(abs(coeff[i]))-a['log_scale'])+max(0,c['epsilon']-c['wall_shift'])*maximum_q
        bounds.append(bound);cx,cy=b['center']
        walls.append(min(cx,g['lx']-cx,cy,g['ly']-cy)-bound)
        for j in range(i):pairs.append(math.dist(b['center'],c['geometry']['bodies'][j]['center'])-bound-bounds[j])
        gx,gy=np.gradient(a['normalized'],dx,dy)
        fd=np.hypot(gx,gy);near={}
        for name,mask in [('nominal_interface',abs(a['normalized'])<=max(dx,dy)/2),
                          ('corrected_interface',abs(a['normalized']+c['wall_shift'])<=max(dx,dy)/2),
                          ('corrected_band',abs(a['normalized']+c['wall_shift'])<=c['epsilon'])]:
            near[name]=dict(minimum=float(fd[mask].min()),maximum=float(fd[mask].max()),count=int(mask.sum()))
        bodies.append(dict(name=b['name'],bands=bands,finite_difference=near))
    envelope=dict(support_radius_bounds=bounds,minimum_wall_gap=min(walls),minimum_pair_gap=min(pairs),
        passed=min(walls)>=.02 and min(pairs)>=.04,
        scope='conservative analytic bound of corrected support for the pinned fixed design, not a design-box certificate')
    return dict(bodies=bodies,envelope=envelope,
        full_band_within_10_percent_diagnostic_only=all(b['bands']['corrected_band']['within_10_percent_diagnostic_only'] for b in bodies))


def report():
    rows=[]
    for final in (False,True):
        for normalized in (False,True):
            c=candidate_config(final,normalized)
            shape='final10' if final else 'initial';audit=geometry_audit(c)
            checks=reference_checks(shape,audit['bodies'])
            audit['user_registered_fd_checks']=checks
            audit['user_registered_fd_gate']=all(x['nominal_interface_gate'] and x['reference_endpoint_gate'] and x['reference_count_gate'] for x in checks)
            rows.append(dict(shape=shape,normalized=normalized,config=c,audit=audit))
    x,y=np.meshgrid((np.arange(880)+.5)*.0025,(np.arange(164)+.5)*.0025,indexing='ij')
    circle=np.hypot(x-.2,y-.2)-.05;gx,gy=np.gradient(circle,.0025,.0025)
    selected=abs(circle)<=.0025/2;norm=np.hypot(gx,gy)[selected]
    circle_check=dict(minimum=float(norm.min()),maximum=float(norm.max()),passed=bool(norm.min()>=.99 and norm.max()<=1.01))
    return dict(schema='wall-shift-cpu-geometry-audit-1',paper_eligible=False,rows=rows,
        circle_gradient_check=circle_check,
        geometric_gates_passed=circle_check['passed'] and all(r['audit']['user_registered_fd_gate'] and r['audit']['envelope']['passed'] for r in rows),
        gpu_authorized_by_this_report=False,
        reason='Geometry checks alone do not authorize GPU: also require frozen unit tests, exact stage registration and hardware preflight. Stage 2 settings await user completion.')


if __name__=='__main__':
    import argparse
    import importlib.util
    import subprocess
    from multibody_penalty_config import EXTERNAL
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(EXTERNAL):raise ValueError('new external diagnostic JSON required')
    spec=importlib.util.spec_from_file_location('calibration',ROOT/'jax_cad/brinkman_mask_calibration.py')
    cal=importlib.util.module_from_spec(spec);spec.loader.exec_module(cal)
    result=report();c=cal.calibrate_planar_mask_shift(.25,epsilon_over_h=3,profile='compact-quintic')
    face=cal.calibrate_planar_mask_shift(.25,epsilon_over_h=3,profile='compact-quintic',sampling='normal-face-average-sensitivity')
    result.update(calibration=dict(inward_shift_m=-c.shift_over_h*.0025,normal_face_inward_m=-face.shift_over_h*.0025,
        frozen_experiment_shift_m=WALL_SHIFT,details=c._asdict(),reference_data='none'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        registered_gate=dict(nominal_minimum=.95,nominal_maximum=1.05,circle_minimum=.99,circle_maximum=1.01,
            fd_endpoint_tolerance=.01,fd_cell_counts='exact match',missing_reference='blocks GPU',
            whole_band='distribution report-only; implementation reference comparison still required'),
        execution=dict(device='CPU; NumPy only',address_space_limit_gib=4,cpu_seconds_limit=45,affinity=sorted(os.sched_getaffinity(0))))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(calibration=result['calibration'],
        geometry=[dict(shape=r['shape'],normalized=r['normalized'],envelope=r['audit']['envelope']) for r in result['rows']],
        gpu_authorized=False),indent=2))
