"""Immutable definitions for the canonical forward-verification cases.

The tabulated lid-driven-cavity values are from Tables I and II of Ghia,
Ghia, and Shin, *Journal of Computational Physics* 48 (1982), 387--411,
doi:10.1016/0021-9991(82)90058-4.

The cylinder configuration and reference values follow the FEATFLOW/DFG
2D-1 definition based on Schäfer and Turek (1996).
"""

from __future__ import annotations


GHIA_1982_DOI = "10.1016/0021-9991(82)90058-4"

GHIA_RE100_U_Y = (
    0.0000,
    0.0547,
    0.0625,
    0.0703,
    0.1016,
    0.1719,
    0.2813,
    0.4531,
    0.5000,
    0.6172,
    0.7344,
    0.8516,
    0.9531,
    0.9609,
    0.9688,
    0.9766,
    1.0000,
)

GHIA_RE100_U = (
    0.00000,
    -0.03717,
    -0.04192,
    -0.04775,
    -0.06434,
    -0.10150,
    -0.15662,
    -0.21090,
    -0.20581,
    -0.13641,
    0.00332,
    0.23151,
    0.68717,
    0.73722,
    0.78871,
    0.84123,
    1.00000,
)

GHIA_RE100_V_X = (
    0.0000,
    0.0625,
    0.0703,
    0.0781,
    0.0938,
    0.1563,
    0.2266,
    0.2344,
    0.5000,
    0.8047,
    0.8594,
    0.9063,
    0.9453,
    0.9531,
    0.9609,
    0.9688,
    1.0000,
)

GHIA_RE100_V = (
    0.00000,
    0.09233,
    0.10091,
    0.10890,
    0.12317,
    0.16077,
    0.17507,
    0.17527,
    0.05454,
    -0.24533,
    -0.22445,
    -0.16914,
    -0.10313,
    -0.08864,
    -0.07391,
    -0.05906,
    0.00000,
)

GHIA_RE100_PRIMARY_VORTEX = {
    "x": 0.6172,
    "y": 0.7344,
    "streamfunction": -0.103423,
}


DFG_2D1 = {
    "name": "DFG flow around cylinder benchmark 2D-1",
    "reference": "Schäfer and Turek (1996), FEATFLOW benchmark definition",
    "reference_url": (
        "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
        "cfdbenchmarking/flow/dfg_benchmark1_re20.html"
    ),
    "original_reference_doi": "10.1007/978-3-322-89849-4_39",
    "high_precision_reference": "Nabh (1998), high-order numerical reference",
    "domain": (2.2, 0.41),
    "cylinder_centre": (0.2, 0.2),
    "cylinder_radius": 0.05,
    "viscosity": 0.001,
    "maximum_inlet_velocity": 0.3,
    "mean_inlet_velocity": 0.2,
    "reynolds_number": 20.0,
    "pressure_probe_front": (0.15, 0.2),
    "pressure_probe_rear": (0.25, 0.2),
    "reference_drag_coefficient": 5.57953523384,
    "reference_lift_coefficient": 0.010618948146,
    "reference_pressure_difference": 0.11752016697,
    "original_reference_intervals": {
        "drag_coefficient": (5.5700, 5.5900),
        "lift_coefficient": (0.0104, 0.0110),
        "pressure_difference": (0.1172, 0.1176),
        "recirculation_length": (0.0842, 0.0852),
    },
}
