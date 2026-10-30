"""Diagnostic exterior pressure reconstruction using the known normal reaction.

On each native horizontal ray the discrete momentum equation contains
G_x p + r_x = q_x. Subtracting the outward cumulative integral of r_x removes
its regularised pressure jump: G_x p_corrected = G_x p + r_x exactly.
Each exterior patch keeps the original pressure where r_x vanishes. The
remaining q_x contains viscosity, advection AND Brinkman forcing; none are
omitted or refitted. Smooth extrapolation of this corrected field is a
candidate, not an assumption that q_x is accurate or a sharp-wall certificate.
Only the two horizontal cylinder probes are implemented, not general traction.
"""
from __future__ import annotations

import numpy as np

from jax_cad.native_pressure_trace import polynomial_intercept


def cubic_kernel(t):
    a=np.abs(np.asarray(t,dtype=float))
    return np.where(a<1,2/3-a*a+.5*a**3,np.where(a<2,(2-a)**3/6,0.))


def cubic_cdf(t):
    t=np.asarray(t,dtype=float)
    central=.5+2*t/3-t**3/3+np.sign(t)*t**4/8
    return np.where(t<=-2,0.,np.where(t<-1,(t+2)**4/24,
        np.where(t<=1,central,np.where(t<2,1-(2-t)**4/24,1.))))


def continuous_correction(points, normals, integrated_reaction, query, *, h, sign):
    """Independent analytic integral of the continuous FE cubic kernel load."""
    points=np.asarray(points); normals=np.asarray(normals)
    strength=np.asarray(integrated_reaction); query=np.asarray(query)
    if points.ndim!=2 or points.shape[1]!=2 or normals.shape!=points.shape or strength.shape!=(len(points),):
        raise ValueError("paired finite FE normal-reaction geometry required")
    if query.ndim!=2 or query.shape[1]!=2 or sign not in (-1,1) or not np.isfinite(h) or h<=0:
        raise ValueError("valid exterior ray and spacing required")
    if not all(np.isfinite(a).all() for a in (points,normals,strength,query)):
        raise ValueError("nonfinite continuous normal reaction")
    tx=(query[:,None,0]-points[None,:,0])/h
    ty=(query[:,None,1]-points[None,:,1])/h
    factor=cubic_cdf(tx) if sign==-1 else cubic_cdf(tx)-1
    return (factor*cubic_kernel(ty)/h)@(strength*normals[:,0])


def observe_continuous(probe, *, points, normals, integrated_reaction, h, centre, radius):
    """Same near-column/transverse observation of an independent FE solution."""
    if not np.allclose(np.asarray(centre)/h,np.round(np.asarray(centre)/h),atol=1e-12,rtol=0) or not np.isclose(radius/h,round(radius/h),atol=1e-12,rtol=0):
        raise ValueError("frozen whole-cell phase required for continuous comparison")
    offsets=np.arange(.5,5.)*h
    sides=[]
    for sign in (-1,1):
        query=np.array([[centre[0]+sign*(radius+s),centre[1]+dy] for s in offsets for dy in (-.5*h,.5*h)])
        raw=np.array([probe(xy.tolist()) for xy in query])
        correction=continuous_correction(points,normals,integrated_reaction,query,h=h,sign=sign)
        values=(raw+correction).reshape(5,2).mean(axis=1)
        sides.append(dict(**polynomial_intercept(offsets,values,degree=2),query=query.tolist(),
            raw_samples=raw.tolist(),added_correction=correction.tolist()))
    return dict(protocol="continuous-normal-reaction-pressure-reconstruction-v1",
        pressure_difference=sides[0]["intercept"]-sides[1]["intercept"],front=sides[0],rear=sides[1],
        brinkman_reaction_removed=False,replaces_primary_measurement=False,paper_eligible=False)


def cumulative_corrections(reaction_u, *, dx):
    """Integrate native face reaction toward the left/right exterior gauges."""
    r = np.asarray(reaction_u, dtype=float)
    if r.ndim != 2 or not np.isfinite(r).all() or not np.isfinite(dx) or dx <= 0:
        raise ValueError("finite native face reaction and positive spacing required")
    if np.any(r[[0, -1]] != 0):
        raise ValueError("normal reaction must not reach the outer x boundaries")
    left = dx*np.cumsum(r, axis=0)[:-1]
    right = -dx*np.cumsum(r[::-1], axis=0)[::-1][1:]
    return left, right


def corrected_pressure_trace(p, reaction_u, *, dx, dy, centre, radius):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.shape(reaction_u) != (p.shape[0]+1, p.shape[1]):
        raise ValueError("finite paired pressure-cell and x-reaction-face arrays required")
    if not np.isfinite([dx, dy, radius]).all() or min(dx, dy, radius) <= 0:
        raise ValueError("positive finite geometry required")
    if np.shape(centre) != (2,) or not np.isfinite(centre).all():
        raise ValueError("finite two-dimensional centre required")
    nx, ny = p.shape
    if nx > 880 or ny > 164:
        raise ValueError("observer exceeds bounded archive scope")
    yc = centre[1]/dy-.5
    j = int(np.floor(yc)); fraction = yc-j
    if j < 1 or j+1 >= ny-1:
        raise ValueError("transverse interpolation must be interior")
    x = (np.arange(nx)+.5)*dx
    corrections = cumulative_corrections(reaction_u, dx=dx)
    sides = []
    for sign, correction in zip((-1, 1), corrections):
        offsets = sign*(x-(centre[0]+sign*radius))
        allowed = np.flatnonzero((offsets > 0) & (np.arange(nx) > 0) & (np.arange(nx) < nx-1))
        indices = allowed[np.argsort(offsets[allowed])][:5]
        if len(indices) != 5 or np.any(np.abs(np.diff(indices)) != 1):
            raise ValueError("five consecutive exterior native columns required")
        def sample(a):
            return (1-fraction)*a[indices, j]+fraction*a[indices, j+1]
        values, added = sample(p), sample(correction)
        fit = polynomial_intercept(offsets[indices], values+added, degree=2)
        gradient_defect = np.diff(correction, axis=0)/dx-np.asarray(reaction_u)[1:-1]
        sides.append(dict(**fit, raw_samples=values.tolist(), added_correction=added.tolist(),
            column_indices=indices.tolist(), row_indices=[j,j+1], y_fraction=float(fraction),
            offsets_over_h=(offsets[indices]/max(dx,dy)).tolist(),
            gradient_identity_max_defect=float(np.max(np.abs(gradient_defect)))))
    return dict(protocol="native-normal-reaction-pressure-reconstruction-v1",
                pressure_difference=sides[0]["intercept"]-sides[1]["intercept"],
                front=sides[0], rear=sides[1],
                reaction_convention="normal-constraint reaction ON BODY; fluid sees minus",
                brinkman_reaction_removed=False, replaces_primary_measurement=False,
                paper_eligible=False, uses_reference_data=False)
