"""Joint MAC/normal constraints, complete force accounting and shape derivatives."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve

from jax_cad.differentiable.mac_channel import (
    ChannelParams, ChannelState, apply_velocity_boundaries, create_channel_grid,
    create_initial_state, divergence, interpolate_alpha_to_faces, parabolic_inlet,
    pressure_gradient, project_velocity, conservative_advection, laplacian_u, laplacian_v,
)
from jax_cad.differentiable.mac_normal_constraint import (
    NormalMarkers, dense_mac_layout, ellipse_markers, normal_interpolation,
    project_velocity_normal, require_valid, spread_normal_reaction, step_normal,
    steady_residual_normal, total_body_reaction,
)
from jax_cad.verification import brinkman_force

jax.config.update("jax_enable_x64", True)


def problem():
    grid = create_channel_grid(16, 12, 1.6, 1.2)
    layout = dense_mac_layout(grid)
    markers = ellipse_markers(jnp.array([.78, .61]), jnp.array([.21, .18]), .17, 12)
    params = ChannelParams(.01, .003, pressure_tolerance=1.e-13, pressure_atol=1.e-14,
                           pressure_maxiter=1000)
    u = .1*jnp.sin(jnp.arange(204).reshape((17, 12))*.13)
    v = .08*jnp.cos(jnp.arange(208).reshape((16, 13))*.11)
    mu, mv = .7+.1*jnp.cos(u), .6+.1*jnp.sin(v)
    inlet = jnp.zeros(12, dtype=jnp.float64)
    return grid, layout, markers, params, u, v, mu, mv, inlet


def joined(u, v):
    return jnp.concatenate((u.ravel(), v.ravel()))


def report(name, **values):
    print(json.dumps(dict(check=name, **values), sort_keys=True), flush=True)


def test_discrete_divergence_and_weighted_gradient_including_half_outlet():
    g, layout, _, _, u, v, _, _, inlet = problem()
    u, v = apply_velocity_boundaries(u, v, inlet, g)
    p = jnp.sin(jnp.arange(192)).reshape((16, 12))
    np.testing.assert_allclose(layout.divergence_matrix @ joined(u, v), divergence(u, v, g).ravel(), atol=1.e-14)
    grad = joined(*pressure_gradient(p, g))
    lhs = (layout.volume_weights*grad)[layout.free]
    rhs = -g.dx*g.dy*(layout.divergence_matrix.T @ p.ravel())[layout.free]
    np.testing.assert_allclose(lhs, rhs, atol=1.e-14)
    np.testing.assert_allclose(layout.volume_weights[16*12:17*12], .5*g.dx*g.dy)


def test_interpolation_linear_reproduction_and_work_force_torque_adjoint():
    g, layout, markers, _, u, v, *_ = problem()
    b, valid = normal_interpolation(markers, layout, g)
    assert valid
    x, y = markers.points.T
    xu, yu = layout.face_points_u.T
    xv, yv = layout.face_points_v.T
    q = jnp.concatenate((.3+.2*xu-.1*yu, -.2+.3*xv+.4*yv))
    expected = (.3+.2*x-.1*y)*markers.normals[:, 0]+(-.2+.3*x+.4*y)*markers.normals[:, 1]
    np.testing.assert_allclose(b @ q, expected, atol=1.e-14)
    traction = jnp.sin(jnp.arange(12)*.31)+.4
    ru, rv = spread_normal_reaction(traction, markers, layout, g)
    spread = joined(ru, rv)
    work_grid = jnp.dot(layout.volume_weights*joined(u, v), spread)
    work_markers = jnp.dot(markers.weights*traction, b @ joined(u, v))
    np.testing.assert_allclose(work_grid, work_markers, atol=1.e-15)
    wu, wv = (layout.volume_weights*spread)[:204], (layout.volume_weights*spread)[204:]
    force = jnp.array([jnp.sum(wu), jnp.sum(wv)])
    marker_force = markers.weights[:, None]*traction[:, None]*markers.normals
    np.testing.assert_allclose(force, jnp.sum(marker_force, axis=0), atol=1.e-14)
    np.testing.assert_allclose(jnp.dot(xv, wv)-jnp.dot(yu, wu),
                               jnp.sum(x*marker_force[:, 1]-y*marker_force[:, 0]), atol=1.e-14)


def test_joint_projection_constraints_energy_idempotence_and_independent_kkt():
    g, layout, markers, params, u, v, mu, mv, inlet = problem()
    result = require_valid(project_velocity_normal(u, v, mu, mv, inlet, g, params, layout, markers))
    assert np.max(np.abs(divergence(result.state.u, result.state.v, g))) < 2.e-12
    assert np.max(np.abs(result.normal_velocity)) < 2.e-13
    assert result.backward_error < 1.e-14
    assert result.stationarity_max < 1.e-15
    # A separate indefinite KKT solve (not the Schur factorisation under test).
    b, _ = normal_interpolation(markers, layout, g)
    c = np.concatenate((g.dx*np.asarray(layout.divergence_matrix), np.asarray(b)))
    free = np.asarray(layout.free)
    cf = c[:, free]
    u0, v0 = apply_velocity_boundaries(u, v, inlet, g)
    q0 = np.asarray(joined(u0, v0))
    mass = np.asarray(layout.volume_weights/joined(mu, mv))[free]
    kkt = np.block([[np.diag(mass), cf.T], [cf, np.zeros((c.shape[0], c.shape[0]))]])
    reference = solve(kkt, np.r_[mass*q0[free], np.zeros(c.shape[0])], assume_a="sym")[:len(free)]
    q = np.asarray(joined(result.state.u, result.state.v))
    np.testing.assert_allclose(q[free], reference, atol=2.e-12)
    assert np.linalg.norm(q) > .01  # not an implementation that returns zero
    np.testing.assert_allclose(np.sum(mass*q0[free]**2),
        np.sum(mass*q[free]**2)+np.sum(mass*(q-q0)[free]**2), atol=1.e-14)
    again = require_valid(project_velocity_normal(result.state.u, result.state.v, mu, mv, inlet, g, params, layout, markers))
    np.testing.assert_allclose(joined(again.state.u, again.state.v), q, atol=2.e-13)
    report("joint_projection", divergence_max=float(jnp.max(jnp.abs(divergence(result.state.u, result.state.v, g)))),
           normal_velocity_max=float(jnp.max(jnp.abs(result.normal_velocity))),
           relative_minimum_eigenvalue=float(result.relative_minimum_eigenvalue),
           backward_error=float(result.backward_error), kkt_velocity_max_difference=float(np.max(np.abs(q[free]-reference))))


def test_balanced_discrete_pressure_normal_load_recovers_equilibrium_and_reaction():
    g, layout, markers, params, u, v, mu, mv, inlet = problem()
    p = .01*jnp.sin(jnp.arange(192)*.17).reshape((16, 12))
    traction = .02*jnp.cos(jnp.arange(12)*.4)
    ru, rv = spread_normal_reaction(traction, markers, layout, g)
    gu, gv = pressure_gradient(p, g)
    ustar, vstar = params.dt*mu*(gu+ru), params.dt*mv*(gv+rv)
    unconstrained, _ = project_velocity(ustar, vstar, mu, mv, inlet, g, params)
    b, _ = normal_interpolation(markers, layout, g)
    assert np.max(np.abs(b @ joined(unconstrained.u, unconstrained.v))) > 1.e-5
    result = require_valid(project_velocity_normal(ustar, vstar, mu, mv, inlet, g, params, layout, markers))
    assert np.max(np.abs(joined(result.state.u, result.state.v))) < 1.e-13
    np.testing.assert_allclose(result.state.p, p, atol=1.e-12)
    np.testing.assert_allclose(result.normal_traction, traction, atol=1.e-12)
    report("balanced_discrete_load", unconstrained_normal_max=float(jnp.max(jnp.abs(b @ joined(unconstrained.u, unconstrained.v)))),
           constrained_velocity_max=float(jnp.max(jnp.abs(joined(result.state.u, result.state.v)))),
           traction_max_error=float(jnp.max(jnp.abs(result.normal_traction-traction))))


def test_rectangular_cells_preserve_pressure_scaling_and_both_constraints():
    _, _, _, params, u, v, mu, mv, inlet = problem()
    g = create_channel_grid(16, 12, 1.6, .96)
    layout = dense_mac_layout(g)
    markers = ellipse_markers(jnp.array([.78, .48]), jnp.array([.2, .14]), .1, 12)
    pressure = .02*jnp.cos(jnp.arange(192)*.17).reshape((16, 12))
    traction = .02*jnp.sin(jnp.arange(12)*.4)
    ru, rv = spread_normal_reaction(traction, markers, layout, g)
    gu, gv = pressure_gradient(pressure, g)
    result = require_valid(project_velocity_normal(params.dt*mu*(gu+ru), params.dt*mv*(gv+rv),
                                                    mu, mv, inlet, g, params, layout, markers))
    assert np.max(np.abs(joined(result.state.u, result.state.v))) < 1.e-13
    np.testing.assert_allclose(result.state.p, pressure, atol=1.e-12)
    np.testing.assert_allclose(result.normal_traction, traction, atol=1.e-12)


def test_nonzero_inlet_step_momentum_force_accounting_and_residual():
    g, layout, markers, params, *_ = problem()
    params = params._replace(include_advection=True, advection_scheme="central")
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    previous = create_initial_state(g, inlet, dtype=jnp.float64)
    alpha = jnp.full((16, 12), 2., dtype=jnp.float64)
    # Nontrivial transport and three joint steps; this is NOT steady CFD.
    for index in range(3):
        result = require_valid(step_normal(previous, alpha, inlet, g, params, layout, markers))
        state = result.state
        assert np.max(np.abs(divergence(state.u, state.v, g))) < 3.e-12
        assert np.max(np.abs(result.normal_velocity)) < 3.e-13
        np.testing.assert_allclose(jnp.sum(state.u[-1])*g.dy, jnp.sum(inlet)*g.dy, atol=2.e-14)
        assert np.linalg.norm(state.u) > .5
        au, av = interpolate_alpha_to_faces(alpha, g)
        gu, gv = pressure_gradient(state.p, g)
        advu, advv = conservative_advection(previous.u, previous.v, g, scheme="central")
        # Independent split-step momentum, including acceleration and reaction.
        ru = (state.u-previous.u)/params.dt+advu-params.viscosity*laplacian_u(previous.u, g)+au*state.u+gu+result.reaction_u
        rv = (state.v-previous.v)/params.dt+advv-params.viscosity*laplacian_v(previous.v, g)+av*state.v+gv+result.reaction_v
        assert np.max(np.abs(joined(ru, rv)[layout.free])) < 3.e-10
        fluid_terms = joined(ru-au*state.u-result.reaction_u, rv-av*state.v-result.reaction_v)
        # Momentum over ALL free native control volumes; inlet alpha reaction
        # belongs to the prescribed boundary and is accounted for separately.
        weighted = np.asarray(fluid_terms*layout.volume_weights).copy()
        fixed = np.ones(weighted.size, bool); fixed[np.asarray(layout.free)] = False
        weighted[fixed] = 0
        from_momentum = -np.array([weighted[:204].sum(), weighted[204:].sum()])
        total = np.asarray(total_body_reaction(result, alpha, layout, g))
        inlet_reaction = float(jnp.sum(au[0]*inlet)*.5*g.dx*g.dy)
        np.testing.assert_allclose(total, from_momentum+np.array([inlet_reaction, 0]), atol=3.e-12)
        constraint = np.sum(np.asarray(markers.weights[:, None]*result.normal_traction[:, None]*markers.normals), axis=0)
        legacy = np.array(brinkman_force(state, alpha, g))
        np.testing.assert_allclose(total, legacy+constraint, atol=2.e-13)
        assert np.linalg.norm(constraint) > 1.e-3  # omitting it is detectable
        residual, normal = steady_residual_normal(result, alpha, inlet, g, params)
        assert np.max(np.abs(normal)) < 3.e-13
        assert residual.continuity.shape == (16, 12)
        report("nonzero_inlet_step", step=index+1, divergence_max=float(jnp.max(jnp.abs(residual.continuity))),
               normal_velocity_max=float(jnp.max(jnp.abs(normal))), split_momentum_max=float(jnp.max(jnp.abs(joined(ru, rv)[layout.free]))),
               brinkman_force=legacy.tolist(), constraint_force=constraint.tolist(), total_body_force=total.tolist(),
               force_balance_error=float(np.max(np.abs(total-from_momentum-np.array([inlet_reaction, 0])))))
        previous = state


def test_independent_steady_stokes_reference_is_a_fixed_point_of_new_step():
    """Nonzero inlet, viscous/penalised body, no advection: NOT a DFG run.

    An independent full saddle solve supplies the steady reference; the new
    timestepping projection must leave it invariant with the same pressure,
    reaction and total body force. No convergence is inferred from three steps.
    """
    g, layout, _, params, *_ = problem()
    markers = ellipse_markers(jnp.array([.78, .61]), jnp.array([.2, .2]), 0., 12)
    inlet = parabolic_inlet(g, .1, dtype=jnp.float64)
    x, y = jnp.meshgrid((jnp.arange(16)+.5)*g.dx, (jnp.arange(12)+.5)*g.dy, indexing="ij")
    alpha = 15*(1-jnp.tanh((jnp.sqrt((x-.78)**2+(y-.61)**2)-.2)/.06))
    au, av = interpolate_alpha_to_faces(alpha, g)
    free = np.asarray(layout.free)
    boundary = jnp.zeros(412, dtype=jnp.float64).at[:12].set(inlet)
    def momentum(q):
        u, v = q[:204].reshape((17, 12)), q[204:].reshape((16, 13))
        return joined(au*u-params.viscosity*laplacian_u(u, g),
                      av*v-params.viscosity*laplacian_v(v, g))
    operator = np.asarray(jax.jacfwd(momentum)(jnp.zeros_like(boundary)))
    wf = np.asarray(layout.volume_weights)[free]
    stiffness = wf[:, None]*operator[np.ix_(free, free)]
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1.e-14)
    b, _ = normal_interpolation(markers, layout, g)
    c = np.concatenate((g.dx*np.asarray(layout.divergence_matrix), np.asarray(b)))
    cf = c[:, free]
    rhs = np.r_[-wf*np.asarray(momentum(boundary))[free], -c @ np.asarray(boundary)]
    matrix = np.block([[stiffness, cf.T], [cf, np.zeros((len(c), len(c)))]])
    solution = solve(matrix, rhs, assume_a="sym")
    q = boundary.at[free].set(solution[:len(free)])
    multipliers = solution[len(free):]
    pressure = -g.dx*multipliers[:192]/(g.dx*g.dy)
    traction = multipliers[192:]/np.asarray(markers.weights)
    initial = ChannelState(q[:204].reshape((17, 12)), q[204:].reshape((16, 13)),
                           jnp.asarray(pressure.reshape((16, 12))))
    result = require_valid(step_normal(initial, alpha, inlet, g, params, layout, markers))
    velocity_change = float(jnp.max(jnp.abs(joined(result.state.u, result.state.v)-q)))
    assert velocity_change < 3.e-13
    np.testing.assert_allclose(result.state.p.ravel(), pressure, atol=3.e-11)
    np.testing.assert_allclose(result.normal_traction, traction, atol=3.e-11)
    residual, normal = steady_residual_normal(result, alpha, inlet, g, params)
    momentum_max = float(jnp.max(jnp.abs(joined(residual.momentum_u, residual.momentum_v)[free])))
    assert momentum_max < 3.e-10
    assert np.linalg.norm(initial.u) > .5
    assert np.max(np.abs(normal)) < 3.e-13
    report("independent_steady_stokes", velocity_fixed_point_error=velocity_change,
           momentum_residual_max=momentum_max, normal_velocity_max=float(jnp.max(jnp.abs(normal))),
           divergence_max=float(jnp.max(jnp.abs(residual.continuity))),
           total_body_force=np.asarray(total_body_reaction(result, alpha, layout, g)).tolist())


@pytest.mark.parametrize("failure", ["duplicate", "outside", "nonunit", "negative_weight", "negative_mobility"])
def test_invalid_or_redundant_constraints_fail_closed(failure):
    g, layout, markers, params, u, v, mu, mv, inlet = problem()
    if failure == "duplicate":
        markers = NormalMarkers(*(jnp.concatenate((a, a[:1]), axis=0) for a in markers))
    elif failure == "outside":
        markers = markers._replace(points=markers.points.at[0, 0].set(.05))
    elif failure == "nonunit":
        markers = markers._replace(normals=markers.normals*2)
    elif failure == "negative_weight":
        markers = markers._replace(weights=-markers.weights)
    else:
        mu = -mu
    result = project_velocity_normal(u, v, mu, mv, inlet, g, params, layout, markers)
    assert not result.valid
    assert np.isnan(np.asarray(result.state.p)).all()
    with pytest.raises(ValueError, match="rank-deficient"):
        require_valid(result)
    report("expected_invalid_constraint", cause=failure, rejected=not bool(result.valid))


def test_dense_scope_limit():
    with pytest.raises(ValueError, match="256"):
        dense_mac_layout(create_channel_grid(440, 82, 2.2, .41))


def test_geometry_jvp_vjp_and_fd_include_normals_positions_and_weights():
    g, layout, _, params, u, v, mu, mv, inlet = problem()
    def objective(theta):
        markers = ellipse_markers(theta[:2], theta[2:4], theta[4], 12)
        result = project_velocity_normal(u, v, mu, mv, inlet, g, params, layout, markers)
        kinetic = jnp.sum(layout.volume_weights*joined(result.state.u, result.state.v)**2)
        # Traction-square includes arclength: its derivatives must include S.
        return kinetic+.001*jnp.sum(markers.weights*result.normal_traction**2)
    theta = jnp.array([.78, .61, .21, .18, .17])
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_grad(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    direction = jnp.array([.2, -.3, .1, -.15, .4])
    _, tangent = jax.jvp(objective, (theta,), (direction,))
    np.testing.assert_allclose(tangent, gradient @ direction, rtol=2.e-10, atol=1.e-12)
    epsilon = 2.e-6
    differences = []
    for i in range(5):
        delta = jnp.eye(5)[i]*epsilon
        estimate = (value_and_grad(theta+delta)[0]-value_and_grad(theta-delta)[0])/(2*epsilon)
        np.testing.assert_allclose(gradient[i], estimate, rtol=2.e-5, atol=2.e-9)
        differences.append(float(abs(gradient[i]-estimate)/max(abs(float(estimate)), 1.e-14)))
    remainders = [float(abs(objective(theta+h*direction)-value-h*(gradient @ direction)))
                  for h in (.002, .001, .0005)]
    assert min(np.log2(np.array(remainders[:-1])/remainders[1:])) > 1.8
    report("shape_derivatives", objective=float(value), gradient=np.asarray(gradient).tolist(),
           finite_difference_relative_errors=differences, jvp_vjp_difference=float(abs(tangent-gradient @ direction)),
           taylor_remainders=remainders, taylor_orders=np.log2(np.array(remainders[:-1])/remainders[1:]).tolist())
