"""Development-only force-free native-pressure trace, not a force correction.

The old five subcell samples span only 1.2 h and can reuse the same pressure
DOFs. This observer uses five DISTINCT, closest admissible cell columns and a
quadratic consistency model. It rejects all cells adjacent to structural body
forcing support. It is specific to the two horizontal DFG pressure probes.
"""
from __future__ import annotations

import numpy as np


def polynomial_intercept(offsets, values, *, degree=2):
    offsets = np.asarray(offsets, dtype=float)
    values = np.asarray(values, dtype=float)
    if offsets.ndim != 1 or values.shape != offsets.shape or offsets.size != 5:
        raise ValueError("exactly five paired samples required")
    if degree not in (1, 2):
        raise ValueError("only preregistered degree one or two")
    if not np.isfinite(offsets).all() or not np.isfinite(values).all() or np.min(offsets) <= 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("finite positive distinct increasing offsets required")
    design = np.polynomial.polynomial.polyvander(offsets/offsets[-1], degree)
    weights = np.linalg.pinv(design)[0]
    l1 = float(np.abs(weights).sum())
    condition = float(np.linalg.cond(design))
    if l1 > 25 or condition > 1000:
        raise ValueError("pressure extrapolation amplification exceeds declared diagnostic cap")
    return dict(intercept=float(weights@values), weights=weights.tolist(),
                weight_l1=l1, condition=condition, degree=degree)


def structural_support(alpha, marker_points, marker_normals, *, dx, dy):
    """Exact nonzero SUPPORT of existing cell-average alpha and cubic B^T.

    Do not detect support from a converged reaction: cancellation/zero velocity
    does not make a geometrically forced face a pure-fluid face.
    """
    alpha = np.asarray(alpha)
    points, normals = np.asarray(marker_points), np.asarray(marker_normals)
    if alpha.ndim != 2 or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("finite nonnegative cell mask required")
    if points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("paired finite marker geometry required")
    if not np.isfinite([dx,dy]).all() or not 0 < dx or not 0 < dy:
        raise ValueError("positive spacing required")
    nx, ny = alpha.shape
    su = np.zeros((nx+1, ny), dtype=bool)
    sv = np.zeros((nx, ny+1), dtype=bool)
    su[:-1] |= alpha != 0; su[1:] |= alpha != 0
    sv[:, :-1] |= alpha != 0; sv[:, 1:] |= alpha != 0
    for component, support, shift in ((0, su, (0., .5)), (1, sv, (.5, 0.))):
        for point, normal in zip(points, normals):
            if normal[component] == 0:
                continue
            coordinate = point/np.array([dx, dy])-shift
            start = np.floor(coordinate).astype(int)-1
            for a in range(4):
                for b in range(4):
                    i, j = start+[a, b]
                    if 0 <= i < support.shape[0] and 0 <= j < support.shape[1] and np.all(np.abs(coordinate-[i, j]) < 2):
                        support[i, j] = True
    return su, sv


def pressure_trace(p, support_u, support_v, *, dx, dy, centre, radius):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or min(dx, dy, radius) <= 0 or not np.isfinite([dx,dy,radius]).all():
        raise ValueError("finite 2-D pressure and positive spacings/radius required")
    if np.shape(centre) != (2,) or not np.isfinite(centre).all():
        raise ValueError("finite two-dimensional centre required")
    nx, ny = p.shape
    if np.shape(support_u) != (nx+1, ny) or np.shape(support_v) != (nx, ny+1):
        raise ValueError("native force-support shapes do not match pressure")
    if np.asarray(support_u).dtype != bool or np.asarray(support_v).dtype != bool:
        raise ValueError("structural support must be Boolean, not force values")
    unsafe = support_u[:-1] | support_u[1:] | support_v[:, :-1] | support_v[:, 1:]
    cy = centre[1]/dy-.5
    j0 = int(np.floor(cy)); fraction = cy-j0
    if j0 < 1 or j0+1 >= ny-1:
        raise ValueError("probe interpolation must be interior")
    x = (np.arange(nx)+.5)*dx
    sides = []
    for sign in (-1, 1):
        wall = centre[0]+sign*radius
        offsets = sign*(x-wall)
        allowed = (offsets > 0) & ~unsafe[:, j0] & ~unsafe[:, j0+1]
        allowed[[0, nx-1]] = False
        indices = np.flatnonzero(allowed)
        indices = indices[np.argsort(offsets[indices])][:5]
        if len(indices) != 5 or np.any(np.abs(np.diff(indices)) != 1):
            raise ValueError("need five consecutive force-free native pressure columns")
        values = (1-fraction)*p[indices, j0]+fraction*p[indices, j0+1]
        selected = offsets[indices]
        record = polynomial_intercept(selected, values)
        sides.append(dict(**record, column_indices=indices.tolist(), row_indices=[j0,j0+1],
                          y_fraction=float(fraction), x=x[indices].tolist(),
                          offsets_over_h=(selected/max(dx,dy)).tolist(), samples=values.tolist()))
    return dict(protocol="force-free-native-quadratic-pressure-diagnostic-v1",
                pressure_front=sides[0]["intercept"], pressure_rear=sides[1]["intercept"],
                pressure_difference=sides[0]["intercept"]-sides[1]["intercept"],
                front=sides[0], rear=sides[1], paper_eligible=False,
                replaces_primary_measurement=False, uses_reference_data=False)
