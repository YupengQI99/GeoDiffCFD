"""Frozen period/drift extraction and benchmark thresholds; no fitted reference."""
import numpy as np
from scipy.signal import find_peaks

REFERENCE_INTERVALS={'Cd_max':(3.22,3.24),'Cl_max':(.99,1.01),'St':(.295,.305)}


def relative_error(value,interval):
    if value is None:return None
    lo,hi=interval;mid=.5*(lo+hi)
    return dict(reference_interval=list(interval),reference_midpoint=mid,
        signed_midpoint_percent=100*(value/mid-1),absolute_midpoint_percent=100*abs(value/mid-1),
        outside_interval_percent=100*max(lo-value,0.,value-hi)/mid,
        signed_endpoint_percent_range=[100*(value/hi-1),100*(value/lo-1)])


def cycles(times,lift):
    times=np.asarray(times);lift=np.asarray(lift)
    amplitude=float(np.ptp(lift))
    if amplitude<1e-6:return dict(minimum_times=[],periods=[],St=None,period_cv=None,complete_period_count=0,reason='no resolved lift oscillation')
    dt=float(np.median(np.diff(times)))
    peaks,_=find_peaks(-lift,prominence=.1*amplitude,distance=max(1,round(.05/dt)))
    positions=[]
    for k in peaks:
        denominator=lift[k-1]-2*lift[k]+lift[k+1]
        shift=.5*(lift[k-1]-lift[k+1])/denominator if denominator else 0.
        positions.append(float(times[k]+np.clip(shift,-.5,.5)*dt))
    periods=np.diff(positions)
    valid=len(periods)>=2
    return dict(minimum_times=positions,periods=periods.tolist(),
        St=float(.1/periods.mean()) if valid else None,
        period_cv=float(periods.std()/periods.mean()) if valid else None,
        complete_period_count=len(periods),reason=None if valid else 'fewer than two complete lift cycles')


def signal_summary(times,drag,lift):
    result=cycles(times,lift)
    result.update(Cd_max=float(np.max(drag)),Cd_min=float(np.min(drag)),Cd_time_mean=float(np.mean(drag)),
        Cl_max=float(np.max(lift)),Cl_min=float(np.min(lift)),Cl_time_mean=float(np.mean(lift)),Cl_amplitude=float(np.ptp(lift)))
    return result


def analyze_history(history,start):
    h=history[start:];t=h['time'];d=h['Cd'];l=h['Cl']
    if len(h)<10 or not np.isfinite(np.c_[t,d,l]).all():raise ValueError('invalid complete window')
    full=signal_summary(t,d,l);mid=len(h)//2
    first=signal_summary(t[:mid],d[:mid],l[:mid]);second=signal_summary(t[mid:],d[mid:],l[mid:])
    errors={k:relative_error(full[k],v) for k,v in REFERENCE_INTERVALS.items()}
    drift={}
    for key in ('Cd_max','Cd_time_mean','St'):
        a,b,scale=first[key],second[key],full[key]
        drift[key]=None if a is None or b is None or scale is None else abs(b-a)/max(abs(scale),1e-12)
    drift_max=None if any(x is None for x in drift.values()) else max(drift.values())
    drift['Cl_amplitude_report_only']=abs(second['Cl_amplitude']-first['Cl_amplitude'])/max(full['Cl_amplitude'],1e-12)
    enough=full['complete_period_count']>=3 and min(first['complete_period_count'],second['complete_period_count'])>=2
    gates=dict(Cd_max=errors['Cd_max']['absolute_midpoint_percent']<=5,
        St=errors['St'] is not None and errors['St']['absolute_midpoint_percent']<=3,
        half_window_drift=drift_max is not None and drift_max<.01,cycles_available=enough)
    return dict(full_window=full,first_half=first,second_half=second,errors=errors,
        relative_half_window_drift=drift,maximum_acceptance_drift=drift_max,gates=gates,
        passed_benchmark_metrics=all(gates.values()),
        method='lift minima with 10% full-window prominence, 0.05 s minimum spacing and quadratic substep interpolation; St=0.1/mean(period); half windows analyzed independently')
