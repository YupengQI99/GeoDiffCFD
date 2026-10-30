"""Data structures for the preprocessed geometry bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from jax_cad.core.grid import GridSpec


class PatchType(Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    UNKNOWN = "unknown"


@dataclass
class FittedPlane:
    """Result of SVD plane fitting for a planar patch."""

    normal: np.ndarray   # (3,) unit outward normal
    point: np.ndarray    # (3,) centroid
    max_error: float     # max distance of any vertex from the plane


@dataclass
class PatchInfo:
    """Metadata for a single STL patch (inlet, outlet, wall, etc.)."""

    name: str
    patch_type: PatchType
    face_indices: np.ndarray           # indices into merged mesh faces
    vertex_indices: np.ndarray         # indices into merged mesh vertices
    area: float
    centroid: np.ndarray               # (3,)
    fitted_plane: Optional[FittedPlane] = None


@dataclass
class QualityReport:
    """Diagnostics for the SDF / mesh quality."""

    is_watertight: bool
    num_fluid_components: int          # should be 1
    num_solid_components: int
    plane_fit_errors: Dict[str, float]  # patch_name → max_error
    total_inlet_area: float
    total_outlet_area: float
    bbox: np.ndarray                    # (2, 3)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GeometryPack:
    """Complete preprocessed geometry bundle.

    This is the primary output of the preprocessing pipeline.
    It contains everything needed to set up a JAX-Fluids or JAX-FEM simulation.
    """

    grid: GridSpec
    phi: np.ndarray                    # (Nx, Ny, Nz) signed distance, <0 = solid
    patches: List[PatchInfo]
    quality: QualityReport

    # Face-level boundary masks:
    #   key = patch_name, value = dict with 'x':(Nx+1,Ny,Nz), 'y':..., 'z':...
    face_masks: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Optional: the merged trimesh object for reference
    merged_mesh: Optional[object] = None

    def get_patches_by_type(self, patch_type: PatchType) -> List[PatchInfo]:
        return [p for p in self.patches if p.patch_type == patch_type]

    def summary(self) -> str:
        lines = [
            f"GeometryPack:",
            f"  Grid: {self.grid.shape}, h={self.grid.spacing:.4f}",
            f"  Phi range: [{self.phi.min():.4f}, {self.phi.max():.4f}]",
            f"  Patches ({len(self.patches)}):",
        ]
        for p in self.patches:
            plane_str = ""
            if p.fitted_plane is not None:
                n = p.fitted_plane.normal
                plane_str = f", normal=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]"
            lines.append(
                f"    {p.name}: {p.patch_type.value}, "
                f"area={p.area:.4f}{plane_str}"
            )
        lines.append(f"  Quality: watertight={self.quality.is_watertight}, "
                      f"fluid_components={self.quality.num_fluid_components}")
        if self.quality.warnings:
            for w in self.quality.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)
