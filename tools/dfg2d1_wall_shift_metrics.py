"""Frozen means, point/interval errors and strict drag-drift citation rule."""
import numpy as np

REFERENCES={'Cd':(5.57953523384,(5.57,5.59)),'Cl':(.010618948146,(.0104,.0110))}
CD_DRIFT_LIMIT=1e-4


def quantity_statistics(values,reference,interval):
    v=np.asarray(values,dtype=float)
    if len(v)!=4000 or not np.isfinite(v).all():raise ValueError('complete 4000-sample steady window required')
    full=float(v.mean());first=float(v[:2000].mean());last=float(v[2000:].mean())
    drift=None if full==0 else abs(last-first)/abs(full)
    low,high=interval
    return dict(mean_14_16=full,mean_14_15=first,mean_15_16=last,relative_half_window_drift=drift,
        drift_undefined_reason='zero full-window mean' if full==0 else None,
        reference_point=reference,reference_interval=list(interval),signed_point_error=full-reference,
        signed_point_error_percent=100*(full/reference-1),absolute_point_error_percent=100*abs(full/reference-1),
        inside_reference_interval=low<=full<=high,
        outside_interval_error_percent=100*max(low-full,0.,full-high)/abs(reference),
        signed_interval_endpoint_error_percent=[100*(full/high-1),100*(full/low-1)])


def analyze_history(h):
    if len(h)!=32000:raise ValueError('exact 32000-step production history required')
    np.testing.assert_allclose(h['time'],.0005*np.arange(1,32001),rtol=0,atol=1e-12)
    values={key:quantity_statistics(h[key][28000:],ref,interval) for key,(ref,interval) in REFERENCES.items()}
    drift=values['Cd']['relative_half_window_drift']
    return dict(quantities=values,window_steps=[28001,32000],window_seconds=[14.,16.],
        half_window_sample_counts=[2000,2000],Cd_drift_limit=CD_DRIFT_LIMIT,
        Cd_citation_drift_passed=drift is not None and drift<CD_DRIFT_LIMIT,
        rule='Only the strict Cd drift <1e-4 determines this citation-drift flag; it is not an accuracy or paper-eligibility gate.',
        corroboration_only=True,permits_stage2=False,paper_eligible=False)
