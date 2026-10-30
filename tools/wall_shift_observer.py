"""Two explicitly named core masks; no extra force or flow-model operations."""
import numpy as np

LABELS=['nominal_body1','corrected_body1']
COLUMNS=[f'{s}_{k}' for s in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for k in LABELS]


def masks(nominal,effective,c):
    cells=np.stack([nominal<=-.015,effective<=-.015])
    nx,ny=nominal.shape;dx=c['grid']['lx']/nx;dy=c['grid']['ly']/ny
    k=round(c['center'][0]/dx)
    if not 0<k<nx or abs(k*dx-c['center'][0])>1e-12:raise ValueError('unaligned core section')
    planes=np.zeros((2,nx+1,ny),bool);planes[:,k]=cells[:,k-1]&cells[:,k]
    counts=cells.sum(axis=(1,2));widths=planes.sum(axis=(1,2))*dy
    if np.any(counts==0) or np.any(widths==0):raise ValueError('empty core')
    return cells,planes,dict(labels=LABELS,margin_m=.015,cell_counts=counts.tolist(),
        areas=(counts*dx*dy).tolist(),centre_u_face_indices=[k,k],section_widths=widths.tolist(),
        definition='nominal phi <= -0.015; corrected phi_eff <= -0.015; both adjacent cells for native-face section')


def statistics(history,start):
    values=np.asarray(history)[start:,7:]
    if values.shape[1]!=6 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('invalid corrected core history')
    means=values.mean(axis=0)
    return dict(labels=LABELS,sample_count=len(values),mean_speed_ratio=means[:2].tolist(),
        rms_speed_ratio=np.sqrt(means[2:4]).tolist(),mean_absolute_flux_ratio=means[4:].tolist(),
        max_absolute_flux_ratio=values[:,4:].max(axis=0).tolist())


def measured_radius(chi,c):
    nx,ny=chi.shape;dx=c['grid']['lx']/nx;dy=c['grid']['ly']/ny
    x=(np.arange(nx)+.5)*dx;y=(np.arange(ny)+.5)*dy;radii=[]
    for j in range(ny):
        row=chi[:,j]-.5
        for i in np.flatnonzero(row[:-1]*row[1:]<0):
            crossing=x[i]-row[i]*dx/(row[i+1]-row[i])
            radii.append(float(np.hypot(crossing-c['center'][0],y[j]-c['center'][1])))
    if not radii:raise ValueError('no half-fraction crossings')
    expected=c['radius']-c['wall_shift']
    return dict(expected_m=expected,minimum_m=min(radii),maximum_m=max(radii),
        mean_m=float(np.mean(radii)),crossing_count=len(radii),
        max_error_m=float(np.max(abs(np.array(radii)-expected))),method='linear interpolation along cell-centred x rows')
