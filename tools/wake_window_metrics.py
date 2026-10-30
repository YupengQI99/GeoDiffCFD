"""Read-only NumPy diagnostics; a screening index is not a stability proof."""
import json
from pathlib import Path
import numpy as np


def window_metrics(directory):
    directory=Path(directory)
    c=json.loads((directory/'config.json').read_text())
    path=directory/'evaluation-000'
    h=np.genfromtxt(path/'history.csv',delimiter=',',names=True)
    start=c['window_start_step'];w=h[start:]
    assert len(h)==c['n_steps'] and len(w)>0
    np.testing.assert_allclose(h['time'],np.arange(1,len(h)+1)*c['dt'],rtol=0,atol=1e-12)
    speed=float(w['max_velocity'].max());factor=c['dt']/(2*c['viscosity'])
    with np.load(path/'fields.npz') as f:
        mask=f['steps']>start
        u=f['u'][mask];v=f['v'][mask]
        # Component maxima may occur at different points/times. This is only
        # an upper bound across SAVED snapshots, not the full time history.
        bound=float(np.max(abs(u))**2+np.max(abs(v))**2)
        centre=(.5*(u[:,:-1]+u[:,1:]))**2+(.5*(v[:,:,:-1]+v[:,:,1:]))**2
        centre_max=float(np.sqrt(centre.max()))
    return dict(dt=c['dt'],n_steps=c['n_steps'],window_start_step=start,window_count=len(w),
        window_start_time=start*c['dt'],end_time=len(h)*c['dt'],
        max_velocity_component_all=float(h['max_velocity'].max()),max_velocity_component_window=speed,
        component_screening_index=factor*speed**2,
        steps_component_index_above_one=int(np.count_nonzero(factor*w['max_velocity']**2>1)),
        window_max_cfl=float(w['cfl'].max()),snapshot_component_upper_index=factor*bound,
        snapshot_cell_center_max_speed=centre_max,snapshot_cell_center_index=factor*centre_max**2,
        snapshot_count=int(mask.sum()),meaning='component screening index and snapshot-only bounds; not a theorem for the projected variable-coefficient solver')


def compare_values(initial_old,final_old,initial_new,final_new):
    if not initial_old>final_old or not initial_old>0 or not initial_new>0:
        raise ValueError('positive baseline objective and baseline improvement required')
    old=100*(1-final_old/initial_old);new=100*(1-final_new/initial_new)
    bound=.1*(initial_old-final_old)
    changes=[initial_new-initial_old,final_new-final_old]
    return dict(J_initial_original=initial_old,J_final_original=final_old,
        J_initial_dt2=initial_new,J_final_dt2=final_new,gain_original_percent=old,gain_dt2_percent=new,
        gain_change_percentage_points=new-old,objective_changes=changes,
        objective_relative_changes_percent=[100*changes[0]/initial_old,100*changes[1]/final_old],
        individual_objective_change_limit=bound,
        gain_gate=abs(new-old)<=1,individual_objective_gate=all(abs(x)<=bound for x in changes))
