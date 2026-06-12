"""
dqs_sim — Distributed Quantum State (Graph State) Simulator
============================================================
GoSC / Lattice Surgery resource estimator for graph state generation.

Modules
-------
  schedulers  : SubstrateScheduler (Case 1), CZScheduler (Case 2),
                CZSchedulerFixedAncilla (Case 3),
                MPPSchedulerLimited (Case 4), CZSchedulerLimited (Case 5)
  cost        : ResourceReport, compute_report()
  topology    : SwitchTopology, placement strategies
  visualizer  : show_gantt, show_heatmaps, show_2d_layout,
                show_qubit_occupancy, show_qubit_occupancy_case2,
                show_qubit_occupancy_case3, show_comparison_dashboard,
                show_swap_overhead, show_topology_layout,
                show_heatmaps_limited
"""

from dqs_sim.schedulers import (
    CZScheduler,
    CZSchedulerFixedAncilla,
    CZSchedulerLimited,
    MPPSchedulerLimited,
    SubstrateScheduler,
)
from dqs_sim.cost import ResourceReport, compute_report
from dqs_sim.topology import SwitchTopology

__all__ = [
    "SubstrateScheduler",
    "CZScheduler",
    "CZSchedulerFixedAncilla",
    "CZSchedulerLimited",
    "MPPSchedulerLimited",
    "ResourceReport",
    "compute_report",
    "SwitchTopology",
]
