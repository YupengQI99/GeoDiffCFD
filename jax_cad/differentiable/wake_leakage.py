"""Passive core diagnostics for fixed-design sensitivity, not a new flow model."""
import numpy as np
import jax
import jax.numpy as jnp
from .mac_channel import step
from jax_cad.verification import brinkman_force

def labels_for_bodies(count):
    if count not in (1,3): raise ValueError('only registered one/three-body diagnostics')
    return [f'{core}_body{i+1}' for core in ('case','common') for i in range(count)]


def columns_for_bodies(count):
    return [f'{stat}_{label}' for stat in ('speed_mean','speed_squared_mean','absolute_flux_ratio') for label in labels_for_bodies(count)]


LABELS = labels_for_bodies(3)
COLUMNS = columns_for_bodies(3)


def core_masks(phi, config):
    """Radial-field margins; common margin is fixed across A/B/C, not an SDF."""
    phi = np.asarray(phi)
    body_count = phi.shape[0]
    labels = labels_for_bodies(body_count)
    if len(config['geometry']['bodies'])!=body_count: raise ValueError('core/body count mismatch')
    cells = np.concatenate([phi <= -config['epsilon'], phi <= -.015])
    nx, ny = phi.shape[1:]
    dx = config['grid']['lx']/nx
    dy = config['grid']['ly']/ny
    planes = np.zeros((2*body_count, nx+1, ny), dtype=bool)
    indices = []
    for i in range(2*body_count):
        cx = config['geometry']['bodies'][i % body_count]['center'][0]
        k = round(cx/dx)
        if not 0 < k < nx or abs(k*dx-cx) > 1e-12:
            raise ValueError('body centre must coincide with an internal native u face')
        planes[i, k] = cells[i, k-1] & cells[i, k]
        indices.append(k)
    counts = cells.sum(axis=(1,2))
    widths = planes.sum(axis=(1,2))*dy
    if np.any(counts == 0) or np.any(widths == 0):
        raise ValueError('empty solid core or centre cross section')
    metadata = dict(labels=labels, cell_counts=counts.tolist(), areas=(counts*dx*dy).tolist(),
                    centre_u_face_indices=indices, section_widths=widths.tolist(),
                    case_margin=config['epsilon'], common_margin=.015,
                    definition='phi_i <= -margin; radial field, not Euclidean distance; native face has both adjacent cells in core')
    return cells, planes, metadata


def observe(state, cells, planes, inlet, *, dy, mean_inlet):
    u = .5*(state.u[:-1]+state.u[1:])
    v = .5*(state.v[:,:-1]+state.v[:,1:])
    speed2 = (u*u+v*v)/mean_inlet**2
    count = jnp.sum(cells, axis=(1,2))
    mean = jnp.sum(cells*jnp.sqrt(speed2), axis=(1,2))/count
    square = jnp.sum(cells*speed2, axis=(1,2))/count
    flux = jnp.sum(planes*jnp.abs(state.u), axis=(1,2))*dy/(jnp.sum(inlet)*dy)
    return jnp.concatenate([mean, square, flux])


def make_observed_forward(grid, params, *, chunk_steps, force_scale, mean_inlet):
    """Same step/force as wake_design, plus passive reductions; masks are runtime args."""
    def forward(state, alpha, inlet, weights, active, cells, planes):
        history = jnp.zeros((chunk_steps, 7+3*cells.shape[0]), dtype=state.u.dtype)
        def body(carry):
            k, current, total, history = carry
            following, d = step(current, alpha, inlet, grid, params)
            forces = force_scale*jnp.asarray(brinkman_force(following, alpha, grid))
            cfl = params.dt*(jnp.max(jnp.abs(following.u))/grid.dx+jnp.max(jnp.abs(following.v))/grid.dy)
            speed = jnp.maximum(jnp.max(jnp.abs(following.u)), jnp.max(jnp.abs(following.v)))
            base = jnp.array([forces[0],forces[1],d.pressure_residual_rel,d.divergence_max,d.mass_flux_mismatch,cfl,speed])
            metrics = jnp.concatenate([base, observe(following,cells,planes,inlet,dy=grid.dy,mean_inlet=mean_inlet)])
            return k+1, following, total+weights[k]*forces[0], history.at[k].set(metrics)
        _, final, value, history = jax.lax.while_loop(lambda c: c[0]<active,body,
            (jnp.array(0),state,jnp.array(0.,dtype=state.u.dtype),history))
        return final,value,history
    return jax.jit(forward)


def statistics(history, start):
    values = np.asarray(history)[start:,7:]
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('invalid leakage history')
    means = values.mean(axis=0)
    count=values.shape[1]//3
    if values.shape[1] not in (6,18): raise ValueError('wrong core diagnostic width')
    return dict(labels=labels_for_bodies(count//2), sample_count=len(values),
                mean_speed_ratio=means[:count].tolist(), rms_speed_ratio=np.sqrt(means[count:2*count]).tolist(),
                mean_absolute_flux_ratio=means[2*count:3*count].tolist(),
                max_absolute_flux_ratio=values[:,2*count:3*count].max(axis=0).tolist())
