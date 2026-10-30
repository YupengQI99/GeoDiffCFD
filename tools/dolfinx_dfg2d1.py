#!/usr/bin/env python3
"""Boundary-fitted DOLFINx reference solve for the DFG 2D-1 benchmark.

The implementation is deliberately independent of :mod:`jax_cad`.  It uses a
Gmsh rectangle-minus-four-arc-circle mesh, a P2/P1 Taylor--Hood
discretisation, and the DFG nonsymmetric stress ``mu * grad(u) - p * I``.
Run it in the image pinned
by the input configuration, for example::

    docker run --rm \
      -v "$PWD":/work:ro -v /tmp/dolfinx-dfg2d1:/results -w /work \
      dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8 \
      python3 tools/dolfinx_dfg2d1.py \
      --config configs/forward_verification/dolfinx_dfg2d1_smoke.json \
      --output-dir /results/smoke

Only standard-library modules are imported at module load time.  This keeps
configuration validation testable outside the DOLFINx container.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dolfinx-dfg2d1-result-1.0"
CONFIG_SCHEMA_VERSION = "dolfinx-dfg2d1-config-1.0"
IMAGE_DIGEST = "sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8"
EXPECTED_VERSIONS = {
    "dolfinx": "0.11.0.post0",
    "gmsh": "4.15.2",
    "petsc4py": "3.25.1",
}
DFG_ORIGINAL_INTERVALS = {
    "drag_coefficient": (5.5700, 5.5900),
    "lift_coefficient": (0.0104, 0.0110),
    "pressure_difference": (0.1172, 0.1176),
}

# Physical tags are fixed so that a result remains interpretable even without
# the mesh file.  The cylinder normal reported below points from solid to fluid
# and is therefore the negative of the fluid-domain facet normal.
FLUID = 1
INLET = 11
OUTLET = 12
WALLS = 13
CYLINDER = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={os.getcwd()}", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _tracked_tree_is_dirty(
    required_tracked_paths: tuple[Path, ...] = (),
) -> bool | None:
    """Report tracked changes, visible untracked files, and untracked inputs.

    Git-ignored generated outputs do not affect the result.  Formal runs should
    nevertheless write outside the repository so a non-ignored stray source
    file cannot be missed.  Required implicit inputs are checked separately
    because a broad ignore rule (for example ``*.json``) may hide them.
    """

    try:
        repository_root = Path(
            subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={os.getcwd()}",
                    "rev-parse",
                    "--show-toplevel",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
        ).resolve()
        status = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_root}",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if status.strip():
        return True
    for source_path in required_tracked_paths:
        try:
            relative = source_path.resolve().relative_to(repository_root)
        except ValueError:
            return True
        tracked = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_root}",
                "ls-files",
                "--error-unmatch",
                "--",
                relative.as_posix(),
            ],
            cwd=repository_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if tracked != 0:
            return True
    return False


def _runtime_option_paths() -> tuple[Path, ...]:
    """Return known implicit FFCx/PETSc option-file locations."""

    xdg_root = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    )
    return (
        Path.cwd() / "ffcx_options.json",
        xdg_root / "ffcx" / "ffcx_options.json",
        Path.home() / ".petscrc",
    )


def _runtime_option_records() -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in _runtime_option_paths()
        if path.is_file()
    ]


def _git_snapshot(
    required_tracked_paths: tuple[Path, ...],
    observed_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Capture immutable source identity before or after a long solve."""

    repository_root_text = _git("rev-parse", "--show-toplevel")
    repository_root = (
        Path(repository_root_text).resolve() if repository_root_text else None
    )
    input_files: dict[str, str | None] = {}
    for path in required_tracked_paths:
        resolved = path.resolve()
        if repository_root is not None:
            try:
                name = resolved.relative_to(repository_root).as_posix()
            except ValueError:
                name = str(resolved)
        else:
            name = str(resolved)
        input_files[name] = _sha256(resolved)
    for path in observed_paths:
        resolved = path.resolve()
        name = str(resolved)
        input_files[name] = _sha256(resolved) if resolved.is_file() else None
    return {
        "repository": _git("config", "--get", "remote.origin.url"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _tracked_tree_is_dirty(required_tracked_paths),
        "input_file_sha256": input_files,
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _artifact(path: Path, base: Path, media_type: str, role: str) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(base.resolve()).as_posix(),
        "sha256": _sha256(path),
        "media_type": media_type,
        "role": role,
    }


def _no_gradient_record() -> dict[str, Any]:
    return {
        "success": False,
        "backend": "none",
        "parameterization": None,
        "parameter_names": [],
        "values": [],
        "norm": None,
        "adjoint_residual": None,
        "message": "Independent forward reference solve has no design gradient.",
        "taylor_test": None,
    }


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config.get("case") != "dfg-cylinder-2d1-re20":
        raise ValueError("case must be 'dfg-cylinder-2d1-re20'")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", config.get("run_id", "")):
        raise ValueError("run_id must be a portable experiment identifier")

    container = config["container"]
    if container.get("image") != "dolfinx/dolfinx:stable":
        raise ValueError("container.image must record dolfinx/dolfinx:stable")
    digest = str(container.get("digest", ""))
    if digest != IMAGE_DIGEST:
        raise ValueError(
            f"container.digest must be the tested image digest {IMAGE_DIGEST}"
        )
    if container.get("expected_versions") != EXPECTED_VERSIONS:
        raise ValueError(f"container.expected_versions must be {EXPECTED_VERSIONS}")

    geometry = config["geometry"]
    lx = _finite_positive(geometry["length"], "geometry.length")
    ly = _finite_positive(geometry["height"], "geometry.height")
    radius = _finite_positive(geometry["cylinder_radius"], "geometry.cylinder_radius")
    centre = [float(value) for value in geometry["cylinder_centre"]]
    expected = (2.2, 0.41, (0.2, 0.2), 0.05)
    if (
        abs(lx - expected[0]) > 1.0e-14
        or abs(ly - expected[1]) > 1.0e-14
        or tuple(centre) != expected[2]
        or abs(radius - expected[3]) > 1.0e-14
    ):
        raise ValueError("geometry does not match the DFG 2D-1 benchmark")

    physics = config["physics"]
    density = _finite_positive(physics["density"], "physics.density")
    viscosity = _finite_positive(
        physics["kinematic_viscosity"], "physics.kinematic_viscosity"
    )
    mean_velocity = _finite_positive(
        physics["mean_inlet_velocity"], "physics.mean_inlet_velocity"
    )
    maximum_velocity = _finite_positive(
        physics["maximum_inlet_velocity"], "physics.maximum_inlet_velocity"
    )
    if any(
        abs(actual - reference) > 1.0e-14
        for actual, reference in (
            (density, 1.0),
            (viscosity, 0.001),
            (mean_velocity, 0.2),
            (maximum_velocity, 0.3),
        )
    ):
        raise ValueError("physics does not match the DFG 2D-1/Re=20 benchmark")
    if abs(float(physics["reynolds_number"]) - 20.0) > 1.0e-14:
        raise ValueError("physics.reynolds_number must be 20")

    mesh = config["mesh"]
    bulk_size = _finite_positive(mesh["bulk_size"], "mesh.bulk_size")
    cylinder_size = _finite_positive(mesh["cylinder_size"], "mesh.cylinder_size")
    wake_size = _finite_positive(mesh["wake_size"], "mesh.wake_size")
    if cylinder_size > bulk_size or wake_size > bulk_size:
        raise ValueError("local mesh sizes must not exceed mesh.bulk_size")
    study = config.get("study")
    if study is not None:
        if study.get("name") != "grid-refinement":
            raise ValueError("study.name must be 'grid-refinement'")
        if study.get("level") not in {"coarse", "medium", "fine"}:
            raise ValueError("study.level must be coarse, medium, or fine")
        study_value = _finite_positive(study.get("value"), "study.value")
        if abs(study_value - cylinder_size) > 1.0e-14:
            raise ValueError("study.value must equal mesh.cylinder_size")
    _finite_positive(
        mesh["cylinder_refinement_distance"], "mesh.cylinder_refinement_distance"
    )
    wake_box = mesh["wake_box"]
    _finite_positive(wake_box["thickness"], "mesh.wake_box.thickness")
    if not (
        0.0 <= float(wake_box["x_min"]) < float(wake_box["x_max"]) <= lx
        and 0.0 <= float(wake_box["y_min"]) < float(wake_box["y_max"]) <= ly
    ):
        raise ValueError("mesh.wake_box must lie inside the channel")

    element = config["finite_elements"]
    if element != {
        "velocity_family": "Lagrange",
        "velocity_degree": 2,
        "pressure_family": "Lagrange",
        "pressure_degree": 1,
    }:
        raise ValueError("finite_elements must specify continuous P2/P1 Taylor-Hood")

    boundary = config["boundary_conditions"]
    required_boundary = {
        "inlet": "parabolic-dirichlet",
        "walls": "no-slip-dirichlet",
        "cylinder": "no-slip-dirichlet",
        "outlet": "natural-zero-traction",
    }
    if boundary != required_boundary:
        raise ValueError("boundary_conditions do not describe the DFG weak problem")

    force = config["force_measurement"]
    if force.get("stress_tensor") != "nu-grad-u-minus-pI":
        raise ValueError("force_measurement.stress_tensor must be nu-grad-u-minus-pI")
    if force.get("normal") != "cylinder-outward-into-fluid":
        raise ValueError("force_measurement.normal must be cylinder-outward-into-fluid")
    if [float(v) for v in force.get("pressure_probe_front", [])] != [0.15, 0.2]:
        raise ValueError("front pressure probe must be [0.15, 0.2]")
    if [float(v) for v in force.get("pressure_probe_rear", [])] != [0.25, 0.2]:
        raise ValueError("rear pressure probe must be [0.25, 0.2]")
    quadrature_degree = _finite_positive(
        force["quadrature_degree"], "force_measurement.quadrature_degree"
    )
    if not quadrature_degree.is_integer():
        raise ValueError("force_measurement.quadrature_degree must be an integer")

    solver = config["solver"]
    if solver.get("initialisation") != "stokes":
        raise ValueError("solver.initialisation must be 'stokes'")
    if solver.get("nonlinear_method") != "newton-line-search":
        raise ValueError("solver.nonlinear_method must be 'newton-line-search'")
    if solver.get("linear_solver") != "mumps-lu":
        raise ValueError("solver.linear_solver must be 'mumps-lu'")
    for key in ("snes_relative_tolerance", "snes_absolute_tolerance"):
        _finite_positive(solver[key], f"solver.{key}")
    if int(solver["snes_max_iterations"]) < 1:
        raise ValueError("solver.snes_max_iterations must be positive")
    if float(solver["snes_max_iterations"]) != int(solver["snes_max_iterations"]):
        raise ValueError("solver.snes_max_iterations must be an integer")
    if int(solver["reporting_precision"]) < 10:
        raise ValueError("solver.reporting_precision must be at least 10")
    if float(solver["reporting_precision"]) != int(solver["reporting_precision"]):
        raise ValueError("solver.reporting_precision must be an integer")
    reference = config["reference"]
    expected_reference = {
        "url": (
            "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
            "cfdbenchmarking/flow/dfg_benchmark1_re20.html"
        ),
        "drag_coefficient": 5.57953523384,
        "lift_coefficient": 0.010618948146,
        "pressure_difference": 0.11752016697,
    }
    for name, expected_value in expected_reference.items():
        if reference.get(name) != expected_value:
            raise ValueError(f"reference.{name} must be {expected_value!r}")
    if reference.get("paper_eligible") is not False:
        raise ValueError(
            "individual DOLFINx runs must keep reference.paper_eligible=false; "
            "paper eligibility is a convergence-series decision"
        )
    if "pressure_trace_diagnostic" in config:
        from pressure_trace_diagnostic import validate
        validate(config)
    return config


def _build_mesh(
    config: dict[str, Any], comm: Any
) -> tuple[Any, Any, dict[str, list[int]]]:
    """Create and distribute a first-order Gmsh boundary-fitted mesh."""

    import gmsh
    from dolfinx.io import gmsh as gmshio

    rank = comm.rank
    boundary_entities: dict[str, list[int]] = {}
    if rank == 0:
        geometry = config["geometry"]
        mesh_config = config["mesh"]
        lx = float(geometry["length"])
        ly = float(geometry["height"])
        cx, cy = (float(value) for value in geometry["cylinder_centre"])
        radius = float(geometry["cylinder_radius"])

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dolfinx_dfg2d1")
        rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, lx, ly)
        circle_centre = gmsh.model.occ.addPoint(cx, cy, 0.0)
        circle_points = [
            gmsh.model.occ.addPoint(x, y, 0.0)
            for x, y in (
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
                (cx, cy - radius),
            )
        ]
        circle_arcs = [
            gmsh.model.occ.addCircleArc(
                circle_points[index],
                circle_centre,
                circle_points[(index + 1) % len(circle_points)],
            )
            for index in range(len(circle_points))
        ]
        circle_wire = gmsh.model.occ.addWire(circle_arcs)
        disk = gmsh.model.occ.addPlaneSurface([circle_wire])
        cut, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, disk)])
        gmsh.model.occ.synchronize()
        if len(cut) != 1 or cut[0][0] != 2:
            raise RuntimeError(f"unexpected Gmsh boolean-cut result: {cut!r}")
        surface = int(cut[0][1])

        classified = {"inlet": [], "outlet": [], "walls": [], "cylinder": []}
        tolerance = 1.0e-6
        curves = gmsh.model.getBoundary([(2, surface)], oriented=False, recursive=False)
        for dimension, tag in curves:
            if dimension != 1:
                continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dimension, tag)
            if abs(xmin) < tolerance and abs(xmax) < tolerance:
                classified["inlet"].append(tag)
            elif abs(xmin - lx) < tolerance and abs(xmax - lx) < tolerance:
                classified["outlet"].append(tag)
            elif (abs(ymin) < tolerance and abs(ymax) < tolerance) or (
                abs(ymin - ly) < tolerance and abs(ymax - ly) < tolerance
            ):
                classified["walls"].append(tag)
            else:
                # The only remaining boundary of rectangle-minus-disk is the
                # circle; checking its bounding box catches topology mistakes.
                if not (
                    xmin >= cx - radius - tolerance
                    and xmax <= cx + radius + tolerance
                    and ymin >= cy - radius - tolerance
                    and ymax <= cy + radius + tolerance
                ):
                    raise RuntimeError(
                        f"unclassified boundary curve {tag}: {(xmin, ymin, xmax, ymax)}"
                    )
                classified["cylinder"].append(tag)
        if any(not entities for entities in classified.values()):
            raise RuntimeError(f"incomplete boundary classification: {classified}")
        if len(classified["cylinder"]) != 4:
            raise RuntimeError(
                "the cylinder must retain four arcs so both pressure probes are "
                "mesh vertices"
            )
        boundary_entities = {key: sorted(value) for key, value in classified.items()}

        gmsh.model.addPhysicalGroup(2, [surface], FLUID)
        gmsh.model.setPhysicalName(2, FLUID, "fluid")
        for name, tag_value in (
            ("inlet", INLET),
            ("outlet", OUTLET),
            ("walls", WALLS),
            ("cylinder", CYLINDER),
        ):
            gmsh.model.addPhysicalGroup(1, classified[name], tag_value)
            gmsh.model.setPhysicalName(1, tag_value, name)

        bulk_size = float(mesh_config["bulk_size"])
        cylinder_size = float(mesh_config["cylinder_size"])
        wake_size = float(mesh_config["wake_size"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(cylinder_size, wake_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", bulk_size)
        gmsh.option.setNumber(
            "Mesh.Algorithm", int(mesh_config.get("gmsh_algorithm", 6))
        )

        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "EdgesList", classified["cylinder"])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", cylinder_size)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", bulk_size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", radius)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMax", float(mesh_config["cylinder_refinement_distance"])
        )

        wake = mesh_config["wake_box"]
        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", wake_size)
        gmsh.model.mesh.field.setNumber(box, "VOut", bulk_size)
        gmsh.model.mesh.field.setNumber(box, "XMin", float(wake["x_min"]))
        gmsh.model.mesh.field.setNumber(box, "XMax", float(wake["x_max"]))
        gmsh.model.mesh.field.setNumber(box, "YMin", float(wake["y_min"]))
        gmsh.model.mesh.field.setNumber(box, "YMax", float(wake["y_max"]))
        gmsh.model.mesh.field.setNumber(box, "Thickness", float(wake["thickness"]))
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, box])
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(gmsh.model if rank == 0 else None, comm, 0, gdim=2)
    if rank == 0:
        gmsh.finalize()
    boundary_entities = comm.bcast(boundary_entities, root=0)
    if mesh_data.facet_tags is None:
        raise RuntimeError("Gmsh import did not preserve physical facet tags")
    return mesh_data.mesh, mesh_data.facet_tags, boundary_entities


def _global_scalar(value: Any, comm: Any) -> float:
    from mpi4py import MPI

    return float(comm.allreduce(float(value), op=MPI.SUM))


def _assemble_scalar(form: Any, comm: Any) -> float:
    from dolfinx import fem

    return _global_scalar(fem.assemble_scalar(form), comm)


def _probe(function: Any, point_xy: list[float], mesh: Any) -> float:
    """Evaluate a scalar FE function at a point, collectively over MPI ranks."""

    import numpy as np
    from dolfinx import geometry

    point = np.asarray([[point_xy[0], point_xy[1], 0.0]], dtype=mesh.geometry.x.dtype)
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, point)
    colliding = geometry.compute_colliding_cells(mesh, candidates, point)
    local_value: float | None = None
    links = colliding.links(0)
    if len(links):
        evaluated = function.eval(point, np.asarray([links[0]], dtype=np.int32))
        local_value = float(np.real(evaluated.reshape(-1)[0]))
    values = [value for value in mesh.comm.allgather(local_value) if value is not None]
    if not values:
        raise RuntimeError(f"pressure probe {point_xy} was not found in the mesh")
    if max(values) - min(values) > 1.0e-9:
        raise RuntimeError(f"inconsistent distributed pressure probe values: {values}")
    return values[0]


def _nearest_tagged_vertex(
    mesh: Any, facet_tags: Any, tag: int, point_xy: list[float]
) -> dict[str, Any]:
    """Return the globally nearest tagged-boundary geometry node."""

    import numpy as np
    from dolfinx.mesh import entities_to_geometry

    facets = facet_tags.find(tag)
    geometry_nodes = entities_to_geometry(
        mesh, mesh.topology.dim - 1, facets, False
    )
    unique_nodes = np.unique(geometry_nodes)
    target = np.asarray(point_xy, dtype=mesh.geometry.x.dtype)
    if unique_nodes.size:
        coordinates = mesh.geometry.x[unique_nodes, :2]
        distances = np.linalg.norm(coordinates - target[None, :], axis=1)
        index = int(np.argmin(distances))
        local = (
            float(distances[index]),
            float(coordinates[index, 0]),
            float(coordinates[index, 1]),
        )
    else:
        local = (math.inf, math.nan, math.nan)
    candidates = mesh.comm.allgather(local)
    distance, x, y = min(candidates, key=lambda candidate: candidate[0])
    return {"target": point_xy, "actual": [x, y], "distance": distance}


def _mesh_diagnostics(mesh: Any, facet_tags: Any, mixed_space: Any) -> dict[str, Any]:
    import numpy as np
    from mpi4py import MPI

    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    cells_local = cell_map.size_local
    geometry_map = mesh.geometry.dofmaps[0]
    coordinates = mesh.geometry.x
    cell_diameters: list[float] = []
    for cell in range(cells_local):
        points = coordinates[geometry_map[cell]]
        diameter = max(
            float(np.linalg.norm(points[i] - points[j]))
            for i in range(len(points))
            for j in range(i)
        )
        cell_diameters.append(diameter)
    local_min = min(cell_diameters) if cell_diameters else math.inf
    local_max = max(cell_diameters) if cell_diameters else 0.0
    global_min = mesh.comm.allreduce(local_min, op=MPI.MIN)
    global_max = mesh.comm.allreduce(local_max, op=MPI.MAX)
    facet_counts = {
        name: int(mesh.comm.allreduce(len(facet_tags.find(tag)), op=MPI.SUM))
        for name, tag in (
            ("inlet", INLET),
            ("outlet", OUTLET),
            ("walls", WALLS),
            ("cylinder", CYLINDER),
        )
    }
    dof_map = mixed_space.dofmap.index_map
    return {
        "cell_type": str(mesh.topology.cell_name()),
        "geometry_degree": 1,
        "cells_global": int(cell_map.size_global),
        "vertices_global": int(mesh.topology.index_map(0).size_global),
        "mixed_dofs_global": int(dof_map.size_global * mixed_space.dofmap.index_map_bs),
        "minimum_cell_diameter": float(global_min),
        "maximum_cell_diameter": float(global_max),
        "tagged_facets_global": facet_counts,
    }


def _solve(config: dict[str, Any], *, source_snapshot: dict[str, Any]) -> dict[str, Any]:
    import basix.ufl
    import dolfinx
    import gmsh
    import numpy as np
    import petsc4py
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    total_started = time.perf_counter()
    started = time.perf_counter()
    mesh, facet_tags, gmsh_entities = _build_mesh(config, comm)
    mesh_seconds = time.perf_counter() - started

    cell = mesh.basix_cell()
    gdim = mesh.geometry.dim
    velocity_element = basix.ufl.element("Lagrange", cell, 2, shape=(gdim,))
    pressure_element = basix.ufl.element("Lagrange", cell, 1)
    mixed_element = basix.ufl.mixed_element([velocity_element, pressure_element])
    W = fem.functionspace(mesh, mixed_element)
    V, _ = W.sub(0).collapse()

    fdim = mesh.topology.dim - 1
    inlet_facets = facet_tags.find(INLET)
    walls_facets = facet_tags.find(WALLS)
    cylinder_facets = facet_tags.find(CYLINDER)
    inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    zero_dofs = fem.locate_dofs_topological(
        (W.sub(0), V), fdim, np.unique(np.concatenate((walls_facets, cylinder_facets)))
    )

    height = float(config["geometry"]["height"])
    maximum_velocity = float(config["physics"]["maximum_inlet_velocity"])
    inlet_value = fem.Function(V)

    def inlet_profile(x: Any) -> Any:
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4.0 * maximum_velocity * x[1] * (height - x[1]) / (height * height)
        return values

    inlet_value.interpolate(inlet_profile)
    zero_value = fem.Function(V)
    bcs = [
        fem.dirichletbc(inlet_value, inlet_dofs, W.sub(0)),
        fem.dirichletbc(zero_value, zero_dofs, W.sub(0)),
    ]

    rho = float(config["physics"]["density"])
    nu = float(config["physics"]["kinematic_viscosity"])
    mu = rho * nu
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    quadrature_degree = int(config["force_measurement"]["quadrature_degree"])
    force_ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tags,
        metadata={"quadrature_degree": quadrature_degree},
    )
    trial = ufl.TrialFunction(W)
    velocity_trial, pressure_trial = ufl.split(trial)
    velocity_test, pressure_test = ufl.TestFunctions(W)
    stokes_a = (
        mu * ufl.inner(ufl.grad(velocity_trial), ufl.grad(velocity_test))
        - pressure_trial * ufl.div(velocity_test)
        + pressure_test * ufl.div(velocity_trial)
    ) * dx
    zero_force = fem.Constant(mesh, np.zeros(gdim, dtype=PETSc.ScalarType))
    stokes_l = ufl.inner(zero_force, velocity_test) * dx
    linear_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    started = time.perf_counter()
    stokes_problem = petsc.LinearProblem(
        stokes_a,
        stokes_l,
        bcs=bcs,
        petsc_options_prefix="dolfinx_dfg2d1_stokes_",
        petsc_options=linear_options,
    )
    stokes_solution = stokes_problem.solve()
    stokes_solution.x.scatter_forward()
    stokes_seconds = time.perf_counter() - started

    solution = fem.Function(W, name="velocity_pressure")
    solution.x.array[:] = stokes_solution.x.array
    solution.x.scatter_forward()
    velocity, pressure = ufl.split(solution)
    residual = (
        rho * ufl.inner(ufl.dot(ufl.grad(velocity), velocity), velocity_test)
        + mu * ufl.inner(ufl.grad(velocity), ufl.grad(velocity_test))
        - pressure * ufl.div(velocity_test)
        + pressure_test * ufl.div(velocity)
    ) * dx
    jacobian = ufl.derivative(residual, solution)
    solver_config = config["solver"]
    nonlinear_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": float(solver_config["snes_relative_tolerance"]),
        "snes_atol": float(solver_config["snes_absolute_tolerance"]),
        "snes_max_it": int(solver_config["snes_max_iterations"]),
        "snes_error_if_not_converged": True,
        **linear_options,
    }
    nonlinear_problem = petsc.NonlinearProblem(
        residual,
        solution,
        bcs=bcs,
        J=jacobian,
        petsc_options_prefix="dolfinx_dfg2d1_navier_stokes_",
        petsc_options=nonlinear_options,
    )
    residual_history: list[dict[str, float | int]] = []

    def monitor(_snes: Any, iteration: int, norm: float) -> None:
        residual_history.append(
            {"iteration": int(iteration), "residual_l2": float(norm)}
        )

    nonlinear_problem.solver.setMonitor(monitor)
    started = time.perf_counter()
    nonlinear_problem.solve()
    solve_seconds = time.perf_counter() - started
    solution.x.scatter_forward()
    reason_code = int(nonlinear_problem.solver.getConvergedReason())
    if reason_code <= 0:
        raise RuntimeError(f"SNES failed with converged reason {reason_code}")

    velocity_h = solution.sub(0).collapse()
    pressure_h = solution.sub(1).collapse()
    velocity_h.x.scatter_forward()
    pressure_h.x.scatter_forward()
    normal_from_fluid = ufl.FacetNormal(mesh)
    normal_into_fluid = -normal_from_fluid
    identity = ufl.Identity(gdim)
    stress = mu * ufl.grad(velocity_h) - pressure_h * identity
    traction = ufl.dot(stress, normal_into_fluid)
    pressure_traction = ufl.dot(-pressure_h * identity, normal_into_fluid)
    viscous_traction = ufl.dot(mu * ufl.grad(velocity_h), normal_into_fluid)
    force_x = _assemble_scalar(
        fem.form(traction[0] * force_ds(CYLINDER)),
        comm,
    )
    force_y = _assemble_scalar(
        fem.form(traction[1] * force_ds(CYLINDER)),
        comm,
    )
    pressure_force_x = _assemble_scalar(
        fem.form(pressure_traction[0] * force_ds(CYLINDER)),
        comm,
    )
    pressure_force_y = _assemble_scalar(
        fem.form(pressure_traction[1] * force_ds(CYLINDER)),
        comm,
    )
    viscous_force_x = _assemble_scalar(
        fem.form(viscous_traction[0] * force_ds(CYLINDER)),
        comm,
    )
    viscous_force_y = _assemble_scalar(
        fem.form(viscous_traction[1] * force_ds(CYLINDER)),
        comm,
    )
    mean_velocity = float(config["physics"]["mean_inlet_velocity"])
    diameter = 2.0 * float(config["geometry"]["cylinder_radius"])
    coefficient_scale = 2.0 / (rho * mean_velocity * mean_velocity * diameter)
    drag = coefficient_scale * force_x
    lift = coefficient_scale * force_y

    front_point = [
        float(value) for value in config["force_measurement"]["pressure_probe_front"]
    ]
    rear_point = [
        float(value) for value in config["force_measurement"]["pressure_probe_rear"]
    ]
    front_vertex = _nearest_tagged_vertex(
        mesh, facet_tags, CYLINDER, front_point
    )
    rear_vertex = _nearest_tagged_vertex(
        mesh, facet_tags, CYLINDER, rear_point
    )
    if max(front_vertex["distance"], rear_vertex["distance"]) > 1.0e-10:
        raise RuntimeError(
            "nominal pressure probes are not cylinder boundary mesh vertices"
        )
    pressure_front = _probe(pressure_h, front_point, mesh)
    pressure_rear = _probe(pressure_h, rear_point, mesh)
    pressure_difference = pressure_front - pressure_rear

    inlet_flux = -_assemble_scalar(
        fem.form(ufl.dot(velocity_h, normal_from_fluid) * ds(INLET)), comm
    )
    outlet_flux = _assemble_scalar(
        fem.form(ufl.dot(velocity_h, normal_from_fluid) * ds(OUTLET)), comm
    )
    cylinder_flux = _assemble_scalar(
        fem.form(ufl.dot(velocity_h, normal_from_fluid) * ds(CYLINDER)), comm
    )
    walls_flux = _assemble_scalar(
        fem.form(ufl.dot(velocity_h, normal_from_fluid) * ds(WALLS)), comm
    )
    divergence_l2 = math.sqrt(
        max(_assemble_scalar(fem.form(ufl.div(velocity_h) ** 2 * dx), comm), 0.0)
    )
    fluid_area = _assemble_scalar(fem.form(1.0 * dx), comm)
    divergence_space = fem.functionspace(mesh, ("DG", 1))
    divergence_field = fem.Function(divergence_space)
    divergence_expression = fem.Expression(
        ufl.div(velocity_h), divergence_space.element.interpolation_points
    )
    divergence_field.interpolate(divergence_expression)
    divergence_map = divergence_space.dofmap.index_map
    divergence_owned = divergence_map.size_local * divergence_space.dofmap.index_map_bs
    local_divergence_max = (
        float(np.max(np.abs(divergence_field.x.array[:divergence_owned])))
        if divergence_owned
        else 0.0
    )
    divergence_max = float(comm.allreduce(local_divergence_max, op=MPI.MAX))
    strong_momentum = (
        rho * ufl.dot(ufl.grad(velocity_h), velocity_h)
        - mu * ufl.div(ufl.grad(velocity_h))
        + ufl.grad(pressure_h)
    )
    element_momentum_l2 = math.sqrt(
        max(
            _assemble_scalar(
                fem.form(ufl.inner(strong_momentum, strong_momentum) * dx),
                comm,
            ),
            0.0,
        )
    )
    boundary_lengths = {
        name: _assemble_scalar(fem.form(1.0 * ds(tag)), comm)
        for name, tag in (
            ("inlet", INLET),
            ("outlet", OUTLET),
            ("walls", WALLS),
            ("cylinder", CYLINDER),
        )
    }
    expected_inlet_flux = mean_velocity * float(config["geometry"]["height"])
    expected_boundary_lengths = {
        "inlet": float(config["geometry"]["height"]),
        "outlet": float(config["geometry"]["height"]),
        "walls": 2.0 * float(config["geometry"]["length"]),
        "cylinder": 2.0
        * math.pi
        * float(config["geometry"]["cylinder_radius"]),
    }

    reference = config["reference"]
    reference_drag = float(reference["drag_coefficient"])
    reference_lift = float(reference["lift_coefficient"])
    reference_pressure = float(reference["pressure_difference"])
    mesh_info = _mesh_diagnostics(mesh, facet_tags, W)
    first_residual = residual_history[0]["residual_l2"] if residual_history else None
    last_residual = residual_history[-1]["residual_l2"] if residual_history else None
    reduction = (
        float(last_residual) / float(first_residual)
        if first_residual not in (None, 0.0) and last_residual is not None
        else None
    )
    interval_values = {
        "drag_coefficient": drag,
        "lift_coefficient": lift,
        "pressure_difference": pressure_difference,
    }
    interval_membership = {
        name: bool(lower <= interval_values[name] <= upper)
        for name, (lower, upper) in DFG_ORIGINAL_INTERVALS.items()
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "case": config["case"],
        "run_id": config["run_id"],
        "status": "success",
        "resolved_config": config,
        "formulation": {
            "geometry": "boundary-fitted Gmsh rectangle-minus-circle",
            "velocity_element": "continuous Lagrange P2",
            "pressure_element": "continuous Lagrange P1",
            "momentum": "rho * (grad(u) u) - mu * div(grad(u)) + grad(p) = 0",
            "continuity": "div(u) = 0",
            "outlet": "natural (mu * grad(u) - p I) n = 0",
            "force_stress": "mu * grad(u) - p I; mu=rho*nu and rho=1",
            "dynamic_viscosity": mu,
            "force_normal": "cylinder-outward-into-fluid",
            "force_quadrature_degree": quadrature_degree,
            "pressure_probe_definition": "p(0.15,0.2) - p(0.25,0.2)",
            "pressure_probe_boundary_vertices": {
                "front": front_vertex,
                "rear": rear_vertex,
            },
        },
        "physical_tags": {
            "fluid": FLUID,
            "inlet": INLET,
            "outlet": OUTLET,
            "walls": WALLS,
            "cylinder": CYLINDER,
            "gmsh_curve_entities": gmsh_entities,
            "boundary_lengths": boundary_lengths,
        },
        "mesh": {**config["mesh"], **mesh_info},
        "metrics": {
            "force_x": force_x,
            "force_y": force_y,
            "pressure_force_x": pressure_force_x,
            "pressure_force_y": pressure_force_y,
            "viscous_force_x": viscous_force_x,
            "viscous_force_y": viscous_force_y,
            "drag_coefficient": drag,
            "lift_coefficient": lift,
            "pressure_drag_coefficient": coefficient_scale * pressure_force_x,
            "pressure_lift_coefficient": coefficient_scale * pressure_force_y,
            "viscous_drag_coefficient": coefficient_scale * viscous_force_x,
            "viscous_lift_coefficient": coefficient_scale * viscous_force_y,
            "pressure_front": pressure_front,
            "pressure_rear": pressure_rear,
            "pressure_difference": pressure_difference,
        },
        "reference": {
            **reference,
            "original_intervals": DFG_ORIGINAL_INTERVALS,
        },
        "original_interval_membership": interval_membership,
        "relative_errors": {
            "drag_coefficient": abs(drag - reference_drag) / abs(reference_drag),
            "lift_coefficient": abs(lift - reference_lift) / abs(reference_lift),
            "pressure_difference": abs(pressure_difference - reference_pressure)
            / abs(reference_pressure),
        },
        "conservation": {
            "inlet_flux": inlet_flux,
            "outlet_flux": outlet_flux,
            "cylinder_flux": cylinder_flux,
            "walls_flux": walls_flux,
            "net_outward_flux": -inlet_flux + outlet_flux + cylinder_flux + walls_flux,
            "relative_inlet_outlet_mismatch": abs(outlet_flux - inlet_flux)
            / max(abs(inlet_flux), 1.0e-30),
            "expected_inlet_flux": expected_inlet_flux,
            "relative_inlet_flux_error": abs(inlet_flux - expected_inlet_flux)
            / expected_inlet_flux,
            "force_x_decomposition_absolute_error": abs(
                force_x - pressure_force_x - viscous_force_x
            ),
            "force_y_decomposition_absolute_error": abs(
                force_y - pressure_force_y - viscous_force_y
            ),
            "boundary_length_absolute_errors": {
                name: abs(boundary_lengths[name] - expected_length)
                for name, expected_length in expected_boundary_lengths.items()
            },
            "fluid_area": fluid_area,
            "divergence_l2": divergence_l2,
            "divergence_rms": divergence_l2 / math.sqrt(fluid_area),
            "divergence_max_at_dg1_nodes": divergence_max,
            "element_strong_momentum_residual_l2": element_momentum_l2,
        },
        "nonlinear_convergence": {
            "converged": True,
            "snes_reason_code": reason_code,
            "iterations": int(nonlinear_problem.solver.getIterationNumber()),
            "linear_iterations_total": int(
                nonlinear_problem.solver.getLinearSolveIterations()
            ),
            "initial_residual_l2": first_residual,
            "final_residual_l2": last_residual,
            "final_over_initial": reduction,
            "history": residual_history,
        },
        "timing_seconds": {
            "mesh_generation_and_import": mesh_seconds,
            "stokes_initialisation": stokes_seconds,
            "nonlinear_solve": solve_seconds,
            "total_before_serialisation": time.perf_counter() - total_started,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mpi_size": comm.size,
            "mpi_rank": comm.rank,
            "dolfinx": dolfinx.__version__,
            "gmsh": gmsh.__version__,
            "petsc4py": petsc4py.__version__,
            "petsc_scalar_type": str(PETSc.ScalarType),
            "configured_container": config["container"],
            "git_commit": source_snapshot["commit"],
            "git_branch": source_snapshot["branch"],
            "git_repository": source_snapshot["repository"],
            "git_dirty": source_snapshot["dirty"],
            "source_snapshot": source_snapshot,
            "runtime_option_files": _runtime_option_records(),
            "reproducibility_environment": {
                name: os.environ[name]
                for name in (
                    "FFCX_OPTIONS",
                    "XDG_CONFIG_HOME",
                    "PETSC_OPTIONS",
                    "PYTHONPATH",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
                if name in os.environ
            },
        },
    }
    if "pressure_trace_diagnostic" in config:
        from pressure_trace_diagnostic import observe
        result["metrics"]["pressure_trace_diagnostic"] = observe(
            config, lambda point: _probe(pressure_h, point, mesh), pressure_front, pressure_rear)
    return result


def _check_runtime_versions(config: dict[str, Any]) -> None:
    import dolfinx
    import gmsh
    import petsc4py

    actual = {
        "dolfinx": dolfinx.__version__,
        "gmsh": gmsh.__version__,
        "petsc4py": petsc4py.__version__,
    }
    expected = config["container"]["expected_versions"]
    mismatches = {
        name: {"expected": expected.get(name), "actual": value}
        for name, value in actual.items()
        if expected.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"container version mismatch: {mismatches}")


def _write_experiment_bundle(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    solver_result: dict[str, Any],
    *,
    started_utc: str,
    ended_utc: str,
    record_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a checksum-protected ``experiment-result-v1`` directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy = output_dir / "config.json"
    shutil.copyfile(config_path, config_copy)
    environment_path = output_dir / "environment.json"
    _write_json(environment_path, solver_result["environment"])
    metrics_path = output_dir / "metrics.json"
    metrics_document = {
        key: solver_result[key]
        for key in (
            "case",
            "run_id",
            "formulation",
            "physical_tags",
            "mesh",
            "metrics",
            "reference",
            "original_interval_membership",
            "relative_errors",
            "conservation",
            "nonlinear_convergence",
            "timing_seconds",
        )
    }
    _write_json(metrics_path, metrics_document)
    raw_result_path = output_dir / "solver_result.json"
    _write_json(raw_result_path, solver_result)
    history_path = output_dir / "residual_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("iteration", "residual_l2"))
        writer.writeheader()
        writer.writerows(solver_result["nonlinear_convergence"]["history"])
    metrics = solver_result["metrics"]
    stdout_path = output_dir / "stdout.log"
    stdout_path.write_text(
        "In-process summary; not an outer Docker stdout capture.\n"
        "DOLFINx DFG 2D-1: "
        f"Cd={metrics['drag_coefficient']:.16g}, "
        f"Cl={metrics['lift_coefficient']:.16g}, "
        f"dp={metrics['pressure_difference']:.16g}\n",
        encoding="utf-8",
    )
    stderr_path = output_dir / "stderr.log"
    stderr_path.write_text(
        "No in-process exception; outer Docker stderr is not captured here.\n",
        encoding="utf-8",
    )

    environment = solver_result["environment"]
    git_commit = environment.get("git_commit")
    git_repository = environment.get("git_repository")
    git_dirty = environment.get("git_dirty")
    if not git_commit or not git_repository or not isinstance(git_dirty, bool):
        raise RuntimeError("standard bundle requires complete Git provenance")
    conservation = solver_result["conservation"]
    convergence = solver_result["nonlinear_convergence"]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_host_bytes = int(
        peak_rss if sys.platform == "darwin" else peak_rss * 1024
    )
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    tracked_environment = environment["reproducibility_environment"]
    result_record = {
        "schema_version": "1.0",
        "experiment_id": "dolfinx_dfg2d1_independent_reference",
        "run_id": config["run_id"],
        "git": {
            "repository": git_repository,
            "commit": git_commit,
            "branch": environment.get("git_branch"),
            "dirty": git_dirty,
        },
        "command": {
            "argv": [sys.executable, *sys.argv],
            "working_directory": os.getcwd(),
        },
        "timestamps": {
            "start_utc": started_utc,
            "end_utc": ended_utc,
            "status": "success",
            "exit_code": 0,
        },
        "configuration": {
            **config,
            "requires_gradient": False,
            "reference_comparison": {
                "boundary_conditions_exact": True,
                "quantity_definition_exact": True,
                "geometry_exact": False,
                "geometry_convergence_study_member": config.get("study")
                is not None,
                "paper_eligible": False,
                "reason": (
                    "The boundary conditions and DFG traction definition match, "
                    "but an individual polygonal-geometry grid cannot establish "
                    "convergence-series paper eligibility."
                    if config.get("study") is not None
                    else "This smoke mesh is not part of the paper convergence study."
                ),
            },
        },
        "hardware": {
            "hostname": environment["hostname"],
            "platform": environment["platform"],
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count() or 1,
            "process_cpu_affinity": affinity,
            "memory_bytes": _memory_bytes(),
            "device_visibility": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "unset_policy": "runtime default; no accelerator backend used",
            },
            "accelerators": [],
        },
        "software": {
            "python": environment["python"],
            "jax": None,
            "jaxlib": None,
            "numpy": environment["numpy"],
            "scipy": None,
            "jax_cuda_plugin": None,
            "jax_cuda_pjrt": None,
            "backend": "cpu",
            "backend_platform_version": None,
            "cuda_toolkit_nvcc": None,
            "precision": "float64",
            "environment_variables": tracked_environment,
        },
        "primal": {
            "converged": bool(convergence["converged"]),
            "iterations": int(convergence["iterations"]),
            "residual_norm": convergence["final_residual_l2"],
            "divergence_rms": conservation["divergence_rms"],
            "divergence_max": conservation["divergence_max_at_dg1_nodes"],
            "mass_flux_mismatch": conservation[
                "relative_inlet_outlet_mismatch"
            ],
            "failure_reason": None,
        },
        "objective": {
            "name": "relative_error_of_boundary_fitted_drag_against_dfg_reference",
            "value": solver_result["relative_errors"]["drag_coefficient"],
            "units": "dimensionless",
            "normalization": "DFG reference C_D=5.57953523384",
        },
        "gradient": _no_gradient_record(),
        "timing": {
            "compile_seconds": None,
            "forward_samples_seconds": [
                solver_result["timing_seconds"]["nonlinear_solve"]
            ],
            "gradient_samples_seconds": [],
            "optimizer_step_samples_seconds": [],
        },
        "memory": {
            "peak_host_bytes": peak_host_bytes,
            "peak_accelerator_bytes": None,
            "measurement_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        },
        "artifacts": [],
        "notes": (
            "Independent DOLFINx P2/P1 DFG 2D-1 reference. The raw solver "
            "record is retained verbatim. Individual study members remain "
            "paper-ineligible; a separate group-level review must establish "
            "the three-grid polygonal-geometry convergence claim. stdout.log "
            "is an in-process summary, not an outer Docker stream capture; the "
            "host invocation must retain its own stdout/stderr and image inspect."
            if config.get("study") is not None
            else "Independent DOLFINx smoke run; not paper-eligible."
        ),
    }
    # Allow another independent formulation to reuse ONLY the evidence writer.
    # Its identity, comparison semantics and objective must be supplied explicitly.
    if record_overrides is not None:
        if set(record_overrides) != {"experiment_id", "reference_comparison", "objective", "notes"}:
            raise ValueError("incomplete independent-formulation record overrides")
        result_record["configuration"]["reference_comparison"] = record_overrides["reference_comparison"]
        for key in ("experiment_id", "objective", "notes"):
            result_record[key] = record_overrides[key]
    artifact_specs = (
        (config_copy, "application/json", "config"),
        (environment_path, "application/json", "environment"),
        (metrics_path, "application/json", "table"),
        (raw_result_path, "application/json", "other"),
        (history_path, "text/csv", "residual-history"),
        (stdout_path, "text/plain", "log"),
        (stderr_path, "text/plain", "log"),
    )
    result_record["artifacts"] = [
        _artifact(path, output_dir, media_type, role)
        for path, media_type, role in artifact_specs
    ]
    result_path = output_dir / "result.json"
    _write_json(result_path, result_record)
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing JSON output"
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if (
        args.output_dir is not None
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        parser.error(f"refusing to overwrite nonempty output: {args.output_dir}")
    config = _load_config(args.config)
    _check_runtime_versions(config)
    required_paths = [Path(__file__).resolve(), args.config.resolve()]
    if "pressure_trace_diagnostic" in config:
        from pressure_trace_diagnostic import validate
        required_paths.extend([Path(__file__).with_name("pressure_trace_diagnostic.py"), validate(config)])
        if config["pressure_trace_diagnostic"]["degree"] == 2:
            required_paths.append(Path(__file__).resolve().parents[1]/"jax_cad/native_pressure_trace.py")
        if config["pressure_trace_diagnostic"].get("sampling") == "native-reaction-corrected-near-columns":
            required_paths.append(Path(__file__).resolve().parents[1]/"jax_cad/reaction_pressure_trace.py")
    working_ffcx_options = Path.cwd() / "ffcx_options.json"
    if working_ffcx_options.is_file():
        required_paths.append(working_ffcx_options.resolve())
    observed_options = tuple(path.resolve() for path in _runtime_option_paths())
    source_snapshot = _git_snapshot(
        tuple(required_paths), observed_paths=observed_options
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    result = _solve(config, source_snapshot=source_snapshot)
    ended_utc = datetime.now(timezone.utc).isoformat()
    ending_snapshot = _git_snapshot(
        tuple(required_paths), observed_paths=observed_options
    )
    if ending_snapshot != source_snapshot:
        raise RuntimeError(
            "Git/source snapshot changed while the DOLFINx solve was running"
        )
    result["provenance"] = {
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "command": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "wall_seconds": time.perf_counter() - wall_started,
        "source_snapshot_before_and_after": source_snapshot,
        "outer_runtime_attestation": (
            "not available inside the container; retain the host Docker command, "
            "RepoDigest inspection, stdout, and stderr alongside the bundle"
        ),
    }
    precision = int(config["solver"]["reporting_precision"])
    if result["environment"]["mpi_rank"] == 0:
        if args.output_dir is not None:
            destination_path = _write_experiment_bundle(
                args.output_dir,
                args.config,
                config,
                result,
                started_utc=started_utc,
                ended_utc=ended_utc,
            )
        else:
            assert args.output is not None
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, result)
            destination_path = args.output
        metrics = result["metrics"]
        print(
            f"DOLFINx DFG 2D-1: Cd={metrics['drag_coefficient']:.{precision}g}, "
            f"Cl={metrics['lift_coefficient']:.{precision}g}, "
            f"dp={metrics['pressure_difference']:.{precision}g}; "
            f"JSON={destination_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
