"""Unified differentiable design-program API."""

from jax_cad.unified.geometry import (
    fourier_area_log_normalization,
    fourier_boundary_radius,
    fourier_log_radius,
    fourier_mode_sobolev_energy,
    fourier_shape_area,
    make_periodic_fourier_material_map,
    periodic_fourier_level_set,
)
from jax_cad.unified.flagship import FourierObstacleBenchmark
from jax_cad.unified.implicit import (
    FixedPointDiagnostics,
    fixed_point_adjoint,
    make_implicit_affine_solver,
    make_implicit_fixed_point_solver,
    tree_l2_norm,
)
from jax_cad.unified.maps import (
    circle_sdf_periodic,
    make_periodic_circle_brinkman_map,
    make_sampled_sdf_offset_map,
    periodic_delta,
    sdf_to_brinkman,
)
from jax_cad.unified.materials import (
    MaterialFields,
    build_cone_filter_kernel,
    compact_delta,
    compact_heaviside,
    fields_from_sdf,
    heaviside_projection,
    make_density_material_map,
    make_periodic_circle_material_map,
    make_sampled_sdf_material_map,
    periodic_density_filter,
    ramp_resistance,
    smoothed_heaviside,
)
from jax_cad.unified.problem import DifferentiableProgram, ProgramAux, compose_program
from jax_cad.unified.verification import (
    TaylorTestResult,
    directional_taylor_test,
    normalize_direction,
    tree_add_scaled,
    tree_vdot,
)

__all__ = [
    "DifferentiableProgram",
    "FixedPointDiagnostics",
    "FourierObstacleBenchmark",
    "MaterialFields",
    "ProgramAux",
    "TaylorTestResult",
    "build_cone_filter_kernel",
    "circle_sdf_periodic",
    "compact_delta",
    "compact_heaviside",
    "compose_program",
    "directional_taylor_test",
    "fields_from_sdf",
    "fixed_point_adjoint",
    "fourier_area_log_normalization",
    "fourier_boundary_radius",
    "fourier_log_radius",
    "fourier_mode_sobolev_energy",
    "fourier_shape_area",
    "heaviside_projection",
    "make_density_material_map",
    "make_implicit_affine_solver",
    "make_implicit_fixed_point_solver",
    "make_periodic_circle_brinkman_map",
    "make_periodic_circle_material_map",
    "make_periodic_fourier_material_map",
    "make_sampled_sdf_material_map",
    "make_sampled_sdf_offset_map",
    "normalize_direction",
    "periodic_delta",
    "periodic_density_filter",
    "periodic_fourier_level_set",
    "ramp_resistance",
    "sdf_to_brinkman",
    "smoothed_heaviside",
    "tree_add_scaled",
    "tree_l2_norm",
    "tree_vdot",
]
