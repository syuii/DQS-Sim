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

Case 3 (CZ gates, k fixed ancilla ELUs)
  - Data ELUs        : |V|
  - Ancilla ELUs     : k  (fixed pool)
  - Each colour class split into sub-batches of ≤ k edges
  - Total Tocks      : 2 × Σᵢ ⌈|Cᵢ|/k⌉
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqs_sim.schedulers import (
        CZScheduler, CZSchedulerFixedAncilla, CZSchedulerLimited,
        MPPSchedulerLimited, SubstrateScheduler,
    )


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

    # ── Case 3 (CZ gates, k fixed ancilla ELUs) — optional ──────────────────
    ancilla_k:               int   | None = None
    n_batches_case3:         int   | None = None   # Σᵢ ⌈|Cᵢ|/k⌉
    tocks_case3:             int   | None = None   # 2 × n_batches
    peak_ancilla_ELUs_case3: int   | None = None   # = k
    total_phys_qubits_case3: int   | None = None
    tock_ratio_c3_c1:        float | None = None   # tocks_case3 / tocks_case1

    # ── Case 4 (MPP + limited connectivity) — optional ──────────────────────
    topology_name:           str   | None = None
    n_elu_slots:             int   | None = None
    placement_strategy:      str   | None = None
    tocks_case4:             int   | None = None
    swap_tocks_case4:        int   | None = None
    mpp_tocks_case4:         int   | None = None
    tock_ratio_c4_c1:        float | None = None

    # ── Case 5 (CZ + limited connectivity) — optional ───────────────────────
    tocks_case5:             int   | None = None
    swap_tocks_case5:        int   | None = None
    cz_tocks_case5:          int   | None = None
    tock_ratio_c5_c2:        float | None = None

    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Results exported → {path}")

    def print_table(self) -> None:
        W    = 36
        has3 = self.tocks_case3 is not None

        c3_peak = f"k={self.ancilla_k}" if has3 else "—"
        c3_tocks = str(self.tocks_case3) if has3 else "—"
        c3_phys  = f"{self.total_phys_qubits_case3:,}" if has3 else "—"
        c3_ratio = f"{self.tock_ratio_c3_c1:.2f}×" if has3 else "—"

        # ── Two-column or three-column separator rows ─────────────────────────
        def _sep():
            if has3:
                return ("─" * (W - 2), "─" * 8, "─" * 8, "─" * 10)
            return ("─" * (W - 2), "─" * 8, "─" * 8)

        rows: list[tuple]
        if has3:
            rows = [
                ("Graph topology",             self.graph_type,              "",                     ""),
                ("Nodes  |V|",                 self.graph_nodes,             self.graph_nodes,        self.graph_nodes),
                ("Edges  |E|",                 self.graph_edges,             self.graph_edges,        self.graph_edges),
                ("Max degree  Δ(G)",           self.max_degree,              self.max_degree,         self.max_degree),
                ("Code distance  d",           self.code_distance,           self.code_distance,      self.code_distance),
                ("Phys. qubits / ELU (2d²-1)", self.phys_per_ELU,           self.phys_per_ELU,       self.phys_per_ELU),
                _sep(),
                ("Data ELUs",                  self.data_ELUs,              self.data_ELUs,           self.data_ELUs),
                ("Peak ancilla ELUs",          self.peak_ancilla_ELUs_case1, self.peak_ancilla_ELUs_case2, c3_peak),
                ("MIS size  (Case 1 only)",    self.mis_size,               "—",                     "—"),
                ("Vizing class  (Case 2/3)",   "—",                          self.vizing_class,       self.vizing_class),
                ("Chromatic index  χ'(G)",     "—",                          self.chromatic_index,    self.chromatic_index),
                ("Ancilla pool  k  (Case 3)",   "—",                          "—",                    self.ancilla_k),
                ("Sub-batches  (Case 3)",       "—",                          "—",                    self.n_batches_case3),
                _sep(),
                ("Total Tocks",                self.tocks_case1,             self.tocks_case2,        c3_tocks),
                ("Total physical qubits",      f"{self.total_phys_qubits_case1:,}",
                                               f"{self.total_phys_qubits_case2:,}",                  c3_phys),
                ("Tock ratio  vs Case 1",      "—",                          f"{self.tock_ratio_c2_c1:.2f}×", c3_ratio),
            ]
        else:
            rows = [
                ("Graph topology",             self.graph_type,              ""),
                ("Nodes  |V|",                 self.graph_nodes,             self.graph_nodes),
                ("Edges  |E|",                 self.graph_edges,             self.graph_edges),
                ("Max degree  Δ(G)",           self.max_degree,              self.max_degree),
                ("Code distance  d",           self.code_distance,           self.code_distance),
                ("Phys. qubits / ELU (2d²-1)", self.phys_per_ELU,           self.phys_per_ELU),
                _sep(),
                ("Data ELUs",                  self.data_ELUs,               self.data_ELUs),
                ("Peak ancilla ELUs",          self.peak_ancilla_ELUs_case1, self.peak_ancilla_ELUs_case2),
                ("MIS size  (Case 1 only)",    self.mis_size,                "—"),
                ("Vizing class  (Case 2)",     "—",                          self.vizing_class),
                ("Chromatic index  χ'(G)",     "—",                          self.chromatic_index),
                _sep(),
                ("Total Tocks",                self.tocks_case1,             self.tocks_case2),
                ("Total physical qubits",      f"{self.total_phys_qubits_case1:,}",
                                               f"{self.total_phys_qubits_case2:,}"),
                ("Tock ratio  C2 / C1",        "—",                          f"{self.tock_ratio_c2_c1:.2f}×"),
            ]

        if has3:
            hdr = f"║  {'Metric':<{W - 2}}║{'Case 1':^10}║{'Case 2':^10}║{'Case 3':^12}║"
            print("\n" + "╔" + "═" * W + "╦" + "═" * 10 + "╦" + "═" * 10 + "╦" + "═" * 12 + "╗")
            print(hdr)
            print("╠" + "═" * W + "╬" + "═" * 10 + "╬" + "═" * 10 + "╬" + "═" * 12 + "╣")
            for row in rows:
                if str(row[0]).startswith("─"):
                    print("╟" + "─" * W + "╫" + "─" * 10 + "╫" + "─" * 10 + "╫" + "─" * 12 + "╢")
                else:
                    label, c1, c2, c3 = row
                    print(f"║  {str(label):<{W - 2}}║{str(c1):^10}║{str(c2):^10}║{str(c3):^12}║")
            print("╚" + "═" * W + "╩" + "═" * 10 + "╩" + "═" * 10 + "╩" + "═" * 12 + "╝\n")
        else:
            print("\n" + "╔" + "═" * W + "╦" + "═" * 10 + "╦" + "═" * 10 + "╗")
            print(f"║  {'Metric':<{W - 2}}║{'Case 1':^10}║{'Case 2':^10}║")
            print("╠" + "═" * W + "╬" + "═" * 10 + "╬" + "═" * 10 + "╣")
            for label, c1, c2 in rows:
                if str(label).startswith("─"):
                    print("╟" + "─" * W + "╫" + "─" * 10 + "╫" + "─" * 10 + "╢")
                else:
                    print(f"║  {str(label):<{W - 2}}║{str(c1):^10}║{str(c2):^10}║")
            print("╚" + "═" * W + "╩" + "═" * 10 + "╩" + "═" * 10 + "╝\n")

        # ── Limited connectivity supplement (Cases 4 & 5) ────────────────────
        has4 = self.tocks_case4 is not None
        has5 = self.tocks_case5 is not None
        if has4 or has5:
            c4_tocks = str(self.tocks_case4) if has4 else "—"
            c4_swap  = str(self.swap_tocks_case4) if has4 else "—"
            c4_mpp   = str(self.mpp_tocks_case4) if has4 else "—"
            c4_ratio = f"{self.tock_ratio_c4_c1:.2f}×" if has4 else "—"

            c5_tocks = str(self.tocks_case5) if has5 else "—"
            c5_swap  = str(self.swap_tocks_case5) if has5 else "—"
            c5_cz    = str(self.cz_tocks_case5) if has5 else "—"
            c5_ratio = f"{self.tock_ratio_c5_c2:.2f}×" if has5 else "—"

            topo_str = self.topology_name or "?"
            elus_str = str(self.n_elu_slots) if self.n_elu_slots else "?"

            lim_rows = [
                ("Topology",              topo_str,    topo_str),
                ("ELU slots",             elus_str,    elus_str),
                ("─", "─", "─"),
                ("Total Tocks",           c4_tocks,    c5_tocks),
                ("  SWAP Tocks",          c4_swap,     c5_swap),
                ("  Operation Tocks",     c4_mpp,      c5_cz),
                ("─", "─", "─"),
                ("Tock ratio  vs unlimited", c4_ratio,  c5_ratio),
            ]

            print("╔" + "═" * W + "╦" + "═" * 10 + "╦" + "═" * 10 + "╗")
            print(f"║  {'LIMITED CONNECTIVITY':<{W - 2}}"
                  f"║{'Case 4':^10}║{'Case 5':^10}║")
            print("╠" + "═" * W + "╬" + "═" * 10 + "╬" + "═" * 10 + "╣")
            for label, c4, c5 in lim_rows:
                if str(label).startswith("─"):
                    print("╟" + "─" * W + "╫" + "─" * 10 + "╫" + "─" * 10 + "╢")
                else:
                    print(f"║  {str(label):<{W - 2}}║{str(c4):^10}║{str(c5):^10}║")
            print("╚" + "═" * W + "╩" + "═" * 10 + "╩" + "═" * 10 + "╝\n")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def compute_report(
    scheduler: "SubstrateScheduler",
    cz: "CZScheduler",
    code_distance: int,
    graph_type: str = "custom",
    cz_k: "CZSchedulerFixedAncilla | None" = None,
    mpp_limited: "MPPSchedulerLimited | None" = None,
    cz_limited: "CZSchedulerLimited | None" = None,
) -> ResourceReport:
    """
    Build a ResourceReport from already-run schedulers.
    scheduler.run() and cz.run() must have been called before this function.
    Optionally include Case 3/4/5 data by passing their schedulers.

    Parameters
    ----------
    scheduler      : finished SubstrateScheduler (Case 1)
    cz             : finished CZScheduler (Case 2)
    code_distance  : surface code distance d
    graph_type     : label string for display
    cz_k           : finished CZSchedulerFixedAncilla (Case 3), or None
    mpp_limited    : finished MPPSchedulerLimited (Case 4), or None
    cz_limited     : finished CZSchedulerLimited (Case 5), or None
    """
    G       = scheduler.G
    n       = len(G.nodes)
    e       = len(G.edges)
    max_deg = max((d for _, d in G.degree()), default=0)
    phys    = physical_qubits_per_ELU(code_distance)

    peak_anc_c1 = max((len(tock) for tock in scheduler.schedule), default=0)
    peak_anc_c2 = max((len(r)    for r    in cz.cz_rounds),       default=0)

    tocks_c1  = len(scheduler.schedule)
    tocks_c2  = 2 * cz.chromatic_index
    chi_prime = cz.chromatic_index
    delta     = cz.max_degree
    ratio_c2  = tocks_c2 / tocks_c1 if tocks_c1 > 0 else float("inf")

    # ── Case 3 (optional) ────────────────────────────────────────────────────
    s3_k     = None
    s3_nb    = None
    s3_tocks = None
    s3_peak  = None
    s3_phys  = None
    s3_ratio = None

    if cz_k is not None:
        s3_k     = cz_k.k
        s3_nb    = len(cz_k.batches)
        s3_tocks = 2 * s3_nb
        s3_peak  = cz_k.k                          # ancilla demand = k per batch
        s3_phys  = (n + s3_peak) * phys
        s3_ratio = s3_tocks / tocks_c1 if tocks_c1 > 0 else float("inf")

    # ── Case 4 (optional) ────────────────────────────────────────────────────
    s4_topo = None
    s4_elus = None
    s4_place = None
    s4_tocks = None
    s4_swap = None
    s4_mpp = None
    s4_ratio = None

    if mpp_limited is not None:
        s4_topo  = mpp_limited.topology.name
        s4_elus  = mpp_limited.topology.n_elus
        s4_place = "spectral"  # recorded at call site if needed
        s4_tocks = mpp_limited.n_tocks()
        s4_swap  = mpp_limited.total_swap_tocks
        s4_mpp   = mpp_limited.total_mpp_tocks
        s4_ratio = s4_tocks / tocks_c1 if tocks_c1 > 0 else float("inf")

    # ── Case 5 (optional) ────────────────────────────────────────────────────
    s5_tocks = None
    s5_swap = None
    s5_cz = None
    s5_ratio = None

    if cz_limited is not None:
        if s4_topo is None:
            s4_topo  = cz_limited.topology.name
            s4_elus  = cz_limited.topology.n_elus
        s5_tocks = cz_limited.n_tocks()
        s5_swap  = cz_limited.total_swap_tocks
        s5_cz    = cz_limited.total_cz_tocks
        s5_ratio = s5_tocks / tocks_c2 if tocks_c2 > 0 else float("inf")

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
        tock_ratio_c2_c1         = ratio_c2,
        # Case 3
        ancilla_k                = s3_k,
        n_batches_case3          = s3_nb,
        tocks_case3              = s3_tocks,
        peak_ancilla_ELUs_case3  = s3_peak,
        total_phys_qubits_case3  = s3_phys,
        tock_ratio_c3_c1         = s3_ratio,
        # Case 4
        topology_name            = s4_topo,
        n_elu_slots              = s4_elus,
        tocks_case4              = s4_tocks,
        swap_tocks_case4         = s4_swap,
        mpp_tocks_case4          = s4_mpp,
        tock_ratio_c4_c1         = s4_ratio,
        # Case 5
        tocks_case5              = s5_tocks,
        swap_tocks_case5         = s5_swap,
        cz_tocks_case5           = s5_cz,
        tock_ratio_c5_c2         = s5_ratio,
    )
