"""GeoDiffCFD: differentiable flow design with Brinkman penalization (package: jax_cad)."""

__version__ = "0.3.0"

from jax_cad.core.grid import GridSpec
from jax_cad.core.geometry_pack import GeometryPack, PatchInfo, QualityReport

__all__ = ["GridSpec", "GeometryPack", "PatchInfo", "QualityReport"]
