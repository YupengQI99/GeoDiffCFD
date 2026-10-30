"""Structured grid specification for background SDF computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Uniform structured grid specification.

    Attributes:
        origin: (3,) lower corner of the domain.
        spacing: Uniform cell size h.
        shape: (Nx, Ny, Nz) number of cells in each direction.
    """

    origin: np.ndarray  # (3,)
    spacing: float
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz)

    # -- derived properties --------------------------------------------------

    @property
    def upper(self) -> np.ndarray:
        """Upper corner of the domain."""
        return self.origin + np.array(self.shape) * self.spacing

    @property
    def cell_centers(self) -> np.ndarray:
        """Cell-center coordinates, shape (Nx, Ny, Nz, 3)."""
        nx, ny, nz = self.shape
        x = self.origin[0] + (np.arange(nx) + 0.5) * self.spacing
        y = self.origin[1] + (np.arange(ny) + 0.5) * self.spacing
        z = self.origin[2] + (np.arange(nz) + 0.5) * self.spacing
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([xx, yy, zz], axis=-1)

    @property
    def cell_centers_flat(self) -> np.ndarray:
        """Flattened cell centers, shape (Nx*Ny*Nz, 3)."""
        return self.cell_centers.reshape(-1, 3)

    def face_centers(self, axis: int) -> np.ndarray:
        """Face-center coordinates for faces normal to *axis*.

        Returns array with shape that has +1 in the *axis* dimension:
          axis=0 → (Nx+1, Ny, Nz, 3)
          axis=1 → (Nx, Ny+1, Nz, 3)
          axis=2 → (Nx, Ny, Nz+1, 3)
        """
        nx, ny, nz = self.shape
        sizes = [nx, ny, nz]
        sizes[axis] += 1

        coords = []
        for d in range(3):
            if d == axis:
                c = self.origin[d] + np.arange(sizes[d]) * self.spacing
            else:
                c = self.origin[d] + (np.arange(sizes[d]) + 0.5) * self.spacing
            coords.append(c)

        grids = np.meshgrid(*coords, indexing="ij")
        return np.stack(grids, axis=-1)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_bounds(
        cls,
        lower: Tuple[float, float, float],
        upper: Tuple[float, float, float],
        spacing: float,
    ) -> GridSpec:
        """Create a grid from bounding box corners and cell spacing.

        The grid is expanded so that *upper* is fully contained.
        """
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
        return cls(origin=lo, spacing=float(spacing), shape=shape)

    @classmethod
    def from_mesh_bounds(
        cls,
        mesh_bounds: np.ndarray,
        spacing: float,
        padding: float = 3.0,
    ) -> GridSpec:
        """Create a grid from a mesh AABB with padding (in multiples of h).

        Args:
            mesh_bounds: (2, 3) array [[xmin,ymin,zmin],[xmax,ymax,zmax]].
            spacing: Cell size h.
            padding: Number of cells of padding around the mesh.
        """
        lo = mesh_bounds[0] - padding * spacing
        hi = mesh_bounds[1] + padding * spacing
        return cls.from_bounds(tuple(lo), tuple(hi), spacing)
