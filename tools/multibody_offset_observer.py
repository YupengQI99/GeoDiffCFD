"""Passive six-core observations; nominal radial and actual corrected masks."""
import numpy as np
from audit_wall_shift_geometry import body_parameters

LABELS=[f'{kind}_body{i+1}' for i in range(3) for kind in ('nominal','corrected')]
COLUMNS=[f'{s}_{k}' for s in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for k in LABELS]


def effective_fields(nominal,c):
    g=c['grid'];dx=g['lx']/g['nx'];dy=g['ly']/g['ny']
    x,y=np.meshgrid((np.arange(g['nx'])+.5)*dx,(np.arange(g['ny'])+.5)*dy,indexing='ij')
    coeff,modes=body_parameters(c);fields=[]
    for i,b in enumerate(c['geometry']['bodies']):
        angle=np.arctan2(y-b['center'][1],x-b['center'][0])
        fp=sum(m*(-coeff[i,k]*np.sin(m*angle)+coeff[i,k+3]*np.cos(m*angle)) for k,m in enumerate(modes))
        fields.append(nominal[i]/np.sqrt(1+fp**2)+c['wall_shift'] if c['normalize_level_set'] else nominal[i]+c['wall_shift'])
    return np.array(fields)


def core_masks(phi,c):
    nominal=np.asarray(phi);effective=effective_fields(nominal,c)
    cells=np.stack([p<=-.015 for i in range(3) for p in (nominal[i],effective[i])])
    _,nx,ny=nominal.shape;dx=c['grid']['lx']/nx;dy=c['grid']['ly']/ny
    planes=np.zeros((6,nx+1,ny),bool);indices=[]
    for i,b in enumerate(c['geometry']['bodies']):
        k=round(b['center'][0]/dx)
        if not 0<k<nx or abs(k*dx-b['center'][0])>1e-12:raise ValueError('unaligned core section')
        planes[2*i:2*i+2,k]=cells[2*i:2*i+2,k-1]&cells[2*i:2*i+2,k]
        indices.extend([k,k])
    counts=cells.sum(axis=(1,2));widths=planes.sum(axis=(1,2))*dy
    if np.any(counts==0) or np.any(widths==0):raise ValueError('empty core')
    return cells,planes,dict(labels=LABELS,margin_m=.015,cell_counts=counts.tolist(),areas=(counts*dx*dy).tolist(),
        centre_u_face_indices=indices,section_widths=widths.tolist(),
        definition='nominal radial phi_i <= -0.015; actual effective phi_i <= -0.015; adjacent-cell intersection at native centre u-face')


def statistics(history,start):
    values=np.asarray(history)[start:,7:]
    if values.shape[1]!=18 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('invalid six-core history')
    means=values.mean(axis=0)
    return dict(labels=LABELS,sample_count=len(values),mean_speed_ratio=means[:6].tolist(),
        rms_speed_ratio=np.sqrt(means[6:12]).tolist(),mean_absolute_flux_ratio=means[12:].tolist(),
        max_absolute_flux_ratio=values[:,12:].max(axis=0).tolist())
