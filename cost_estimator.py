"""
cost_estimator.py
=================
Physical and logical resource estimation for graph state generation
via GoSC (Game of Surface Code / Lattice Surgery).

Physical model
--------------
Each ELU is a distance-d rotated surface code patch.
  Physical qubits per ELU = 2d² - 1  (data + syndrome qubits)

Case 1 (MPP / stabilizer)
  - Data ELUs        : |V|
  - Ancilla ELUs     : one per generator measured simultaneously
  - Peak ancilla     : max measurements active in a single Tock
  - Total Tocks      : len(scheduler.schedule)

Case 2 (CZ gates, unlimited ancillas)
  - Data ELUs        : |V|
  - Ancilla ELUs     : one per CZ gate running in parallel
  - Peak ancilla     : max edges in any CZ-round colour class
  - Total Tocks      : 2 × χ'(G)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate_scheduler import CZScheduler, SubstrateScheduler


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def physical_qubits_per_ELU(d: int) -> int:
    """Physical qubits in a distance-d rotated surface code patch: 2d² - 1."""
    return 2 * d * d - 1


# ---------------------------------------------------------------------------
# Resource report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResourceReport:
    """Full resource estimate for both Case 1 and Case 2."""

    # ── Graph ────────────────────────────────────────────────────────────────
    graph_type:  str
    graph_nodes: int
    graph_edges: int
    max_degree:  int

    # ── Code ─────────────────────────────────────────────────────────────────
    code_distance: int

    # ── Case 1 (MPP / stabilizer) ────────────────────────────────────────────
    mis_size:                int
    nodes_measured_case1:    int
    tocks_case1:             int
    peak_ancilla_ELUs_case1: int   # max simultaneous measurements in one Tock

    # ── Case 2 (CZ gates, unlimited ancillas) ───────────────────────────────
    chromatic_index:         int   # χ'(G)
    tocks_case2:             int   # = 2 × χ'(G)
    vizing_class:            int   # 1 → χ'= Δ,  2 → χ'= Δ+1
    peak_ancilla_ELUs_case2: int   # max simultaneous CZ gates in one round

    # ── Physical (derived) ──────────────────────────────────────────────────
    phys_per_ELU:                int   # 2d² - 1
    data_ELUs:                   int   # = |V|
    total_phys_qubits_case1:     int   # (data + peak_anc) × phys_per_ELU
    total_phys_qubits_case2:     int

    # ── Comparison ───────────────────────────────────────────────────────────
    tock_ratio_c2_c1: float   # tocks_case2 / tocks_case1

    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Results exported → {path}")

    def print_table(self) -> None:
        W = 36
        sep_row = ("─" * (W - 2), "─" * 8, "─" * 8)

        rows: list[tuple] = [
            ("Graph topology",           self.graph_type,              ""),
            ("Nodes  |V|",               self.graph_nodes,             self.graph_nodes),
            ("Edges  |E|",               self.graph_edges,             self.graph_edges),
            ("Max degree  Δ(G)",         self.max_degree,              self.max_degree),
            ("Code distance  d",         self.code_distance,           self.code_distance),
            ("Phys. qubits / ELU (2d²-1)", self.phys_per_ELU,         self.phys_per_ELU),
            sep_row,
            ("Data ELUs",                self.data_ELUs,               self.data_ELUs),
            ("Peak ancilla ELUs",        self.peak_ancilla_ELUs_case1, self.peak_ancilla_ELUs_case2),
            ("MIS size  (Case 1 only)",  self.mis_size,                "—"),
            ("Vizing class  (Case 2)",   "—",                          self.vizing_class),
            ("Chromatic index  χ'(G)",   "—",                          self.chromatic_index),
            sep_row,
            ("Total Tocks",              self.tocks_case1,             self.tocks_case2),
            ("Total physical qubits",    f"{self.total_phys_qubits_case1:,}",
                                         f"{self.total_phys_qubits_case2:,}"),
            ("Tock ratio  C2 / C1",      "—",                          f"{self.tock_ratio_c2_c1:.2f}×"),
        ]

        print("\n" + "╔" + "═" * W + "╦" + "═" * 10 + "╦" + "═" * 10 + "╗")
        print(f"║  {'Metric':<{W - 2}}║{'Case 1':^10}║{'Case 2':^10}║")
        print("╠" + "═" * W + "╬" + "═" * 10 + "╬" + "═" * 10 + "╣")
        for label, c1, c2 in rows:
            if str(label).startswith("─"):
                print("╟" + "─" * W + "╫" + "─" * 10 + "╫" + "─" * 10 + "╢")
            else:
                print(f"║  {str(label):<{W - 2}}║{str(c1):^10}║{str(c2):^10}║")
        print("╚" + "═" * W + "╩" + "═" * 10 + "╩" + "═" * 10 + "╝\n")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def compute_report(
    scheduler: "SubstrateScheduler",
    cz: "CZScheduler",
    code_distance: int,
    graph_type: str = "custom",
) -> ResourceReport:
    """
    Build a ResourceReport from already-run SubstrateScheduler and CZScheduler.
    Both `.run()` must have been called before this function.

    Parameters
    ----------
    scheduler      : finished SubstrateScheduler (Case 1)
    cz             : finished CZScheduler (Case 2)
    code_distance  : surface code distance d
    graph_type     : label string for display
    """
    G       = scheduler.G
    n       = len(G.nodes)
    e       = len(G.edges)
    max_deg = max((d for _, d in G.degree()), default=0)
    phys    = physical_qubits_per_ELU(code_distance)

    peak_anc_c1 = max((len(tock) for tock in scheduler.schedule), default=0)
    peak_anc_c2 = max((len(r)    for r    in cz.cz_rounds),       default=0)

    tocks_c1    = len(scheduler.schedule)
    tocks_c2    = 2 * cz.chromatic_index
    chi_prime   = cz.chromatic_index
    delta       = cz.max_degree
    ratio       = tocks_c2 / tocks_c1 if tocks_c1 > 0 else float("inf")

    return ResourceReport(
        graph_type               = graph_type,
        graph_nodes              = n,
        graph_edges              = e,
        max_degree               = max_deg,
        code_distance            = code_distance,
        mis_size                 = len(scheduler.mis),
        nodes_measured_case1     = len(scheduler.to_measure),
        tocks_case1              = tocks_c1,
        peak_ancilla_ELUs_case1  = peak_anc_c1,
        chromatic_index          = chi_prime,
        tocks_case2              = tocks_c2,
        vizing_class             = 1 if chi_prime == delta else 2,
        peak_ancilla_ELUs_case2  = peak_anc_c2,
        phys_per_ELU             = phys,
        data_ELUs                = n,
        total_phys_qubits_case1  = (n + peak_anc_c1) * phys,
        total_phys_qubits_case2  = (n + peak_anc_c2) * phys,
        tock_ratio_c2_c1         = ratio,
    )
