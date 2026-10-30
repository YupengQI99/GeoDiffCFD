"""Tests for core data structures."""

import numpy as np
import pytest

from jax_cad.core.grid import GridSpec
from jax_cad.core.geometry_pack import PatchType


class TestGridSpec:
    def test_from_bounds_shape(self):
        g = GridSpec.from_bounds((0, 0, 0), (1, 1, 1), 0.1)
        assert g.shape == (10, 10, 10)

    def test_from_bounds_origin(self):
        g = GridSpec.from_bounds((-1, -2, -3), (1, 2, 3), 0.5)
        np.testing.assert_array_equal(g.origin, [-1, -2, -3])

    def test_cell_centers_shape(self):
        g = GridSpec.from_bounds((0, 0, 0), (1, 1, 1), 0.25)
        assert g.cell_centers.shape == (4, 4, 4, 3)

    def test_cell_centers_values(self):
        g = GridSpec.from_bounds((0, 0, 0), (1, 1, 1), 0.5)
        # First cell center should be at (0.25, 0.25, 0.25)
        np.testing.assert_allclose(g.cell_centers[0, 0, 0], [0.25, 0.25, 0.25])

    def test_face_centers_shape(self):
        g = GridSpec.from_bounds((0, 0, 0), (1, 1, 1), 0.5)
        assert g.face_centers(0).shape == (3, 2, 2, 3)  # Nx+1, Ny, Nz
        assert g.face_centers(1).shape == (2, 3, 2, 3)  # Nx, Ny+1, Nz
        assert g.face_centers(2).shape == (2, 2, 3, 3)  # Nx, Ny, Nz+1

    def test_upper(self):
        g = GridSpec.from_bounds((0, 0, 0), (1, 1, 1), 0.1)
        np.testing.assert_allclose(g.upper, [1, 1, 1])

    def test_from_mesh_bounds(self):
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.3]])
        g = GridSpec.from_mesh_bounds(bounds, spacing=0.1, padding=2.0)
        assert g.origin[0] < 0.0  # padded below
        assert g.upper[0] > 1.0   # padded above


class TestPatchType:
    def test_enum_values(self):
        assert PatchType.INLET.value == "inlet"
        assert PatchType.WALL.value == "wall"
