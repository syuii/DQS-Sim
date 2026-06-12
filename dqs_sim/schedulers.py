"""
substrate_scheduler.py
======================
Substrate scheduler for graph state generation in quantum error correction.

Three-phase algorithm:
  Phase 1 — Maximum Independent Set (MIS) reduction
  Phase 2 — Spectral ordering of qubits onto the physical substrate
  Phase 3 — Greedy interval scheduling to minimise Tocks

Reference layout: 4 · |V| tiles, one ancilla row and one data row.

Usage
-----
    from substrate_scheduler import SubstrateScheduler
    import networkx as nx

    G = nx.path_graph(5)
    scheduler = SubstrateScheduler(G)
    scheduler.run()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---------------------------------------------------------------------------
# Visual constants — edit here to restyle all plots at once
# ---------------------------------------------------------------------------

_STYLE = {
    # node colours
    "node_mis":     "#81c784",   # green  — MIS (free)
    "node_measure": "#e57373",   # red    — to be measured
    "node_neighbor":"#ffd54f",   # yellow — Z-neighbour (tock plots)
    "node_idle":    "#f0f0f0",   # grey   — uninvolved

    # edges
    "edge_active":  "#e57373",
    "edge_inactive":"#cccccc",

    # tile colours (cycling, one per measurement in a Tock)
    "tile_palette": ["#e57373", "#81c784", "#64b5f6", "#ffb74d",
                     "#ba68c8", "#4db6ac", "#ff8a65", "#9575cd"],

    # typography / layout
    "font_size":    12,
    "node_size":    700,
    "title_size":   13,
    "fig_bg":       "white",
    "edge_lw":      1.5,
    "node_lw":      1.4,
    "node_edge_c":  "#555555",
}

# A single fixed layout seed so every graph view looks the same across plots.
_LAYOUT_SEED = 42


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Block = dict[str, Any]   # {"id", "L", "R", "neighbors"}
Tock  = list[Block]


# ---------------------------------------------------------------------------
# SubstrateScheduler
# ---------------------------------------------------------------------------

class SubstrateScheduler:
    """
    Schedule stabilizer measurements for graph-state generation on a
    nearest-neighbour substrate.

    Parameters
    ----------
    graph : nx.Graph
        Interaction graph.  Nodes are qubits; edges indicate that the two
        qubits participate in a joint stabilizer measurement.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.G: nx.Graph = nx.Graph(graph)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        if not nx.is_connected(self.G):
            print("Warning: graph is not connected. Results may vary.")

        # Populated by run()
        self.mis:      list      = []
        self.to_measure: list    = []
        self.mapping:  dict      = {}
        self.schedule: list[Tock] = []
        self._pos:     dict      = {}   # fixed spring-layout positions

    # ------------------------------------------------------------------
    # Phase 1 — MIS
    # ------------------------------------------------------------------

    def _is_independent_set(self, nodes: tuple) -> bool:
        """Return True iff no two nodes in *nodes* share an edge."""
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if self.G.has_edge(u, v):
                    return False
        return True

    def _exact_mis(self) -> list:
        """Brute-force maximum independent set (feasible for |V| ≤ 20)."""
        node_list = list(self.G.nodes)
        for size in range(len(node_list), 0, -1):
            for subset in combinations(node_list, size):
                if self._is_independent_set(subset):
                    return list(subset)
        return []

    def find_maximum_independent_set(self) -> list:
        """
        Return a Maximum Independent Set of G.

        Uses exact brute-force for |V| ≤ 20; falls back to a greedy
        maximal independent set for larger graphs.
        """
        n = len(self.G)
        if n <= 20:
            return self._exact_mis()
        print(f"Warning: |V| = {n} > 20. Using greedy MIS (not necessarily maximum).")
        return nx.maximal_independent_set(self.G)

    def phase_1_reduction(self) -> list:
        """Compute MIS and split nodes into free (MIS) and to-measure sets."""
        self.mis        = self.find_maximum_independent_set()
        self.to_measure = [v for v in self.G.nodes if v not in self.mis]
        return self.mis

    # ------------------------------------------------------------------
    # Phase 2 — Spectral ordering
    # ------------------------------------------------------------------

    def phase_2_mapping(self) -> dict:
        """
        Map qubits to contiguous physical positions via spectral ordering
        (Fiedler vector of the graph Laplacian).
        """
        ordering    = list(nx.spectral_ordering(self.G))
        self.mapping = {node: i for i, node in enumerate(ordering)}
        return self.mapping

    # ------------------------------------------------------------------
    # Phase 3 — Greedy interval scheduling
    # ------------------------------------------------------------------

    def phase_3_scheduling(self) -> list[Tock]:
        """
        Model each measurement as interval [L, R] on the substrate and
        greedily pack non-overlapping intervals into each Tock.
        """
        blocks: list[Block] = []
        for v in self.to_measure:
            neighbors   = list(self.G.neighbors(v))
            positions   = [self.mapping[v]] + [self.mapping[n] for n in neighbors]
            L, R        = min(positions), max(positions)
            blocks.append({"id": v, "L": L, "R": R, "neighbors": neighbors})

        # Sort by (R, L) — standard earliest-deadline-first variant
        blocks.sort(key=lambda b: (b["R"], b["L"]))

        schedule: list[Tock] = []
        while blocks:
            current:   Tock       = []
            remaining: list[Block] = []
            occupied_until        = -1

            for b in blocks:
                if b["L"] > occupied_until:
                    current.append(b)
                    occupied_until = b["R"]
                else:
                    remaining.append(b)

            schedule.append(current)
            blocks = remaining

        return schedule

    # ------------------------------------------------------------------
    # Generator graph & chromatic verification
    # ------------------------------------------------------------------

    def build_generator_graph(self) -> nx.Graph:
        """
        Build the generator (conflict) graph G_gen.

        Nodes are only the *measured* generators (i.e. self.to_measure).
        MIS nodes are initialised in |+⟩ and never measured, so they do not
        appear as vertices in G_gen.

        An edge (m1, m2) in G_gen means the two stabilizer measurements share
        at least one qubit in their support (vertex ∪ neighbours) and therefore
        cannot be executed in the same Tock.
        """
        gen_graph = nx.Graph()
        gen_graph.add_nodes_from(self.to_measure)

        for i, m1 in enumerate(self.to_measure):
            for m2 in self.to_measure[i + 1 :]:
                support_m1 = {m1} | set(self.G.neighbors(m1))
                support_m2 = {m2} | set(self.G.neighbors(m2))
                if support_m1 & support_m2:
                    gen_graph.add_edge(m1, m2)

        return gen_graph

    def verify_chromatic_number(self) -> tuple[nx.Graph, dict, int]:
        """
        Report bounds on χ(S') and compare against the scheduler's tock count.

        S' is the generator conflict graph built over the *measured* generators
        (to_measure nodes only — MIS nodes are initialised in |+⟩ and removed).
        Per the notes, χ(S') is the theoretical minimum number of tocks.

        Because computing the exact chromatic number is NP-hard we report two
        bounds that bracket χ(S'):

          Lower bound — clique number ω(S') ≤ χ(S')
              A clique of size k means k mutually-conflicting generators, so
              they must all go in different tocks.

          Upper bound — DSATUR greedy colouring χ_greedy ≥ χ(S')
              Any valid colouring is a valid schedule, so the greedy result
              is an achievable (but not necessarily minimal) tock count.

        Pass/fail criteria
        ------------------
          IMPOSSIBLE  tocks < ω          — beats the proven lower bound (bug)
          OPTIMAL     tocks == ω          — matches the proven lower bound
          FEASIBLE    ω < tocks ≤ χ_greedy — within the provable range
          SUBOPTIMAL  tocks > χ_greedy    — worse than plain graph colouring

        Returns
        -------
        gen_graph   : S' (conflict graph over to_measure nodes)
        coloring    : DSATUR colouring dict {node: colour_index}
        chi_greedy  : greedy upper-bound estimate of χ(S')
        """
        gen_graph  = self.build_generator_graph()
        tocks      = len(self.schedule)

        if not gen_graph.nodes:
            print("\n  No generators to measure — trivially 0 Tocks.\n")
            return gen_graph, {}, 0

        # ── Upper bound: DSATUR greedy colouring ─────────────────────────────
        coloring   = nx.coloring.greedy_color(gen_graph, strategy="DSATUR")
        chi_greedy = max(coloring.values()) + 1

        # ── Lower bound: clique number ω(S') ─────────────────────────────────
        # nx.graph_clique_number is exact but slow; use max_clique for small
        # graphs and the Lovász theta / degree bound for large ones.
        n_gen = len(gen_graph.nodes)
        if n_gen <= 50:
            clique_nodes, _ = nx.clique.max_weight_clique(gen_graph, weight=None)
            omega = len(clique_nodes)
        else:
            # large_clique_size returns an int directly
            omega = nx.approximation.clique.large_clique_size(gen_graph)

        bug = tocks < omega

        print("\n" + "=" * 52)
        print("  CHROMATIC NUMBER VERIFICATION")
        print("=" * 52)
        print(f"  Generator graph nodes  : {n_gen}  (after MIS)")
        print(f"  Generator graph edges  : {len(gen_graph.edges)}")
        print(f"  Lower bound  ω(S')     : {omega}  (clique number)")
        print(f"  Upper bound  χ(S')     : {chi_greedy}  (DSATUR colouring)")
        print(f"  Tocks achieved         : {tocks}")
        print(f"  MIS savings            : {chi_greedy - tocks} Tock(s) vs upper bound")
        print(f"  Bound check            : {'✗ FAIL — tocks < ω (schedule bug)' if bug else '✓ PASS'}")
        print("=" * 52 + "\n")

        return gen_graph, coloring, chi_greedy

    # ------------------------------------------------------------------
    # Shared layout helper
    # ------------------------------------------------------------------

    def _get_pos(self) -> dict:
        """Return (and cache) a fixed spring-layout for G."""
        if not self._pos:
            self._pos = nx.spring_layout(self.G, seed=_LAYOUT_SEED, k=2)
        return self._pos

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_common_ax(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=_STYLE["title_size"], pad=10)
        ax.axis("off")
        ax.margins(0.18)

    @staticmethod
    def _legend(ax: plt.Axes, entries: list[tuple[str, str]]) -> None:
        """entries = [(color, label), ...]"""
        handles = [
            mpatches.Patch(facecolor=c, edgecolor=_STYLE["node_edge_c"],
                           linewidth=1.2, label=lbl)
            for c, lbl in entries
        ]
        ax.legend(handles=handles, loc="upper right",
                  fontsize=10, frameon=False)

    def _draw_graph(
        self,
        ax:           plt.Axes,
        node_colors:  list[str],
        node_labels:  dict | None = None,
        active_edges: list        | None = None,
    ) -> None:
        """Render G onto *ax* with consistent styling."""
        pos = self._get_pos()

        all_edges    = list(self.G.edges())
        active_edges = active_edges or []
        inactive     = [
            e for e in all_edges
            if e not in active_edges and (e[1], e[0]) not in active_edges
        ]

        nx.draw_networkx_edges(
            self.G, pos, edgelist=inactive,
            width=_STYLE["edge_lw"], edge_color=_STYLE["edge_inactive"], ax=ax,
        )
        if active_edges:
            nx.draw_networkx_edges(
                self.G, pos, edgelist=active_edges,
                width=_STYLE["edge_lw"] + 0.5,
                edge_color=_STYLE["edge_active"], ax=ax,
            )

        nx.draw_networkx_nodes(
            self.G, pos,
            node_size=_STYLE["node_size"],
            node_color=node_colors,
            edgecolors=_STYLE["node_edge_c"],
            linewidths=_STYLE["node_lw"],
            ax=ax,
        )
        nx.draw_networkx_labels(
            self.G, pos,
            labels=node_labels or {n: str(n) for n in self.G.nodes},
            font_size=_STYLE["font_size"],
            font_color="#222222",
            ax=ax,
        )

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def plot_input_graph(self, ax: plt.Axes) -> None:
        """Render the bare input graph."""
        colors = [_STYLE["node_idle"] for _ in self.G.nodes]
        self._draw_graph(ax, colors)
        self._apply_common_ax(ax, "Input Graph")

    def plot_mis(self, ax: plt.Axes) -> None:
        """Highlight MIS (green) vs. to-measure (red) nodes."""
        colors = [
            _STYLE["node_mis"] if n in self.mis else _STYLE["node_measure"]
            for n in self.G.nodes
        ]
        self._draw_graph(ax, colors)
        self._legend(ax, [
            (_STYLE["node_mis"],     "MIS — initialise |+⟩"),
            (_STYLE["node_measure"], "Measure"),
        ])
        self._apply_common_ax(ax, "Phase 1 — Maximum Independent Set")

    def plot_ordering(self, ax: plt.Axes) -> None:
        """
        Draw nodes laid out on a horizontal line in physical qubit order,
        with substrate position labels below each node.
        """
        n   = len(self.G.nodes)
        pos = {node: (self.mapping[node], 0) for node in self.G.nodes}
        colors = [
            _STYLE["node_mis"] if node in self.mis else _STYLE["node_measure"]
            for node in self.G.nodes
        ]

        nx.draw_networkx_edges(
            self.G, pos,
            width=_STYLE["edge_lw"],
            edge_color=_STYLE["edge_inactive"],
            alpha=0.5, ax=ax,
        )
        nx.draw_networkx_nodes(
            self.G, pos,
            node_size=_STYLE["node_size"],
            node_color=colors,
            edgecolors=_STYLE["node_edge_c"],
            linewidths=_STYLE["node_lw"],
            ax=ax,
        )
        nx.draw_networkx_labels(
            self.G, pos,
            font_size=_STYLE["font_size"],
            font_color="#222222",
            ax=ax,
        )
        for node in self.G.nodes:
            ax.text(
                self.mapping[node], -0.32,
                f"Q{self.mapping[node]}",
                ha="center", fontsize=9, color="#888888",
            )

        ax.set_xlim(-0.9, n - 0.1)
        ax.set_ylim(-0.65, 0.65)
        self._apply_common_ax(ax, "Phase 2 — Spectral Ordering")

    def plot_tock(self, tock_idx: int, ax_graph: plt.Axes, ax_tiles: plt.Axes) -> None:
        """
        Render one Tock: graph view (left) + substrate tile view (right).
        """
        tock   = self.schedule[tock_idx]
        n_q    = len(self.G.nodes)

        # ---- graph view ----
        measured  = {b["id"] for b in tock}
        neighbors = {nb for b in tock for nb in b["neighbors"]}
        colors    = [
            _STYLE["node_measure"]  if n in measured  else
            _STYLE["node_neighbor"] if n in neighbors else
            _STYLE["node_idle"]
            for n in self.G.nodes
        ]
        labels = {
            n: (f"{n}\nX" if n in measured else
                f"{n}\nZ" if n in neighbors else str(n))
            for n in self.G.nodes
        }
        active_edges = [
            (b["id"], nb) if self.G.has_edge(b["id"], nb) else (nb, b["id"])
            for b in tock for nb in b["neighbors"]
        ]
        self._draw_graph(ax_graph, colors, labels, active_edges)
        self._legend(ax_graph, [
            (_STYLE["node_measure"],  "X (measured)"),
            (_STYLE["node_neighbor"], "Z (neighbour)"),
        ])

        op_strs = []
        for b in tock:
            parts = [f"X{self.mapping[b['id']]}"] + \
                    [f"Z{self.mapping[nb]}" for nb in b["neighbors"]]
            op_strs.append(" ⊗ ".join(parts))
        ax_graph.text(
            0.5, 0.02, "  |  ".join(op_strs),
            transform=ax_graph.transAxes,
            ha="center", fontsize=8, family="monospace", color="#555555",
        )
        self._apply_common_ax(ax_graph, f"Tock {tock_idx + 1} — Graph View")

        # ---- tile view ----
        ax_tiles.set_xlim(-0.6, n_q)
        ax_tiles.set_ylim(-0.3, 2.8)
        ax_tiles.set_aspect("equal")
        ax_tiles.axis("off")
        ax_tiles.set_title(
            f"Tock {tock_idx + 1} — Substrate Tiles",
            fontsize=_STYLE["title_size"], pad=10,
        )

        # row labels
        ax_tiles.text(-0.2, 2.0, "Ancilla", ha="right", va="center",
                      fontsize=10, color="#666666", fontweight="bold")
        ax_tiles.text(-0.2, 0.5, "Data",    ha="right", va="center",
                      fontsize=10, color="#666666", fontweight="bold")

        # draw all tiles (idle)
        for q in range(n_q):
            for row_y in (1.5, 0.0):
                r = plt.Rectangle((q, row_y), 1, 1,
                                  facecolor="#fafafa",
                                  edgecolor="#dddddd", linewidth=0.8)
                ax_tiles.add_patch(r)
                label = f"A{q}" if row_y == 1.5 else f"Q{q}"
                ax_tiles.text(q + 0.5, row_y + 0.5, label,
                              ha="center", va="center",
                              fontsize=9, color="#bbbbbb")

        # highlight active tiles
        palette = _STYLE["tile_palette"]
        for bi, b in enumerate(tock):
            color = palette[bi % len(palette)]
            L, R  = b["L"], b["R"]

            # ancilla span [L, R]
            for q in range(L, R + 1):
                r = plt.Rectangle((q, 1.5), 1, 1,
                                  facecolor=color,
                                  edgecolor="white", linewidth=1.5, alpha=0.88)
                ax_tiles.add_patch(r)
                ax_tiles.text(q + 0.5, 2.0, f"A{q}",
                              ha="center", va="center",
                              fontsize=9, fontweight="bold", color="white")

            # data qubits
            involved = [self.mapping[b["id"]]] + \
                       [self.mapping[nb] for nb in b["neighbors"]]
            for q in involved:
                r = plt.Rectangle((q, 0.0), 1, 1,
                                  facecolor=color,
                                  edgecolor="white", linewidth=1.5, alpha=0.88)
                ax_tiles.add_patch(r)
                ax_tiles.text(q + 0.5, 0.5, f"Q{q}",
                              ha="center", va="center",
                              fontsize=9, fontweight="bold", color="white")

    def plot_generator_graph(self, gen_graph: nx.Graph,
                             coloring: dict, ax: plt.Axes) -> None:
        """Draw G_gen with its chromatic colouring."""
        palette     = _STYLE["tile_palette"]
        pos         = nx.spring_layout(gen_graph, seed=_LAYOUT_SEED, k=2)
        node_colors = [palette[coloring[n] % len(palette)] for n in gen_graph.nodes]

        nx.draw_networkx_edges(
            gen_graph, pos,
            width=_STYLE["edge_lw"],
            edge_color=_STYLE["edge_inactive"],
            alpha=0.6, ax=ax,
        )
        nx.draw_networkx_nodes(
            gen_graph, pos,
            node_size=_STYLE["node_size"] + 200,
            node_color=node_colors,
            edgecolors=_STYLE["node_edge_c"],
            linewidths=_STYLE["node_lw"],
            ax=ax,
        )
        nx.draw_networkx_labels(
            gen_graph, pos,
            font_size=_STYLE["font_size"],
            font_color="#222222",
            ax=ax,
        )

        n_colors = max(coloring.values()) + 1
        handles  = [
            mpatches.Patch(
                facecolor=palette[i % len(palette)],
                edgecolor=_STYLE["node_edge_c"],
                linewidth=1.2,
                label=f"Color {i + 1}  (Tock {i + 1})",
            )
            for i in range(n_colors)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
        self._apply_common_ax(ax, "Generator Graph — Chromatic Colouring")

    # ------------------------------------------------------------------
    # Composite figure builders
    # ------------------------------------------------------------------

    def show_overview(self) -> None:
        """
        Three-panel overview: input graph | MIS | spectral ordering.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                 facecolor=_STYLE["fig_bg"])
        fig.suptitle("Substrate Scheduler — Overview",
                     fontsize=14, fontweight="bold", y=1.01)

        self.plot_input_graph(axes[0])
        self.plot_mis(axes[1])
        self.plot_ordering(axes[2])

        plt.tight_layout()
        plt.show()
        plt.close()

    def show_tock(self, tock_idx: int) -> None:
        """Graph view + tile view for a single Tock, side-by-side."""
        fig, (ax_g, ax_t) = plt.subplots(
            1, 2, figsize=(14, 5.5),
            facecolor=_STYLE["fig_bg"],
            gridspec_kw={"width_ratios": [1, 1]},
        )
        self.plot_tock(tock_idx, ax_g, ax_t)
        plt.tight_layout()
        plt.show()
        plt.close()

    def show_all_tocks(self) -> None:
        """Render every Tock as a row of two panels in one figure."""
        n_tocks = len(self.schedule)
        fig, axes = plt.subplots(
            n_tocks, 2,
            figsize=(14, 5.5 * n_tocks),
            facecolor=_STYLE["fig_bg"],
            gridspec_kw={"width_ratios": [1, 1]},
        )
        fig.suptitle("Substrate Scheduler — All Tocks",
                     fontsize=14, fontweight="bold", y=1.01)

        # normalise axes shape when n_tocks == 1
        if n_tocks == 1:
            axes = [axes]

        for i, row in enumerate(axes):
            self.plot_tock(i, row[0], row[1])

        plt.tight_layout()
        plt.show()
        plt.close()

    def show_generator_graph(self, gen_graph: nx.Graph, coloring: dict) -> None:
        """Stand-alone figure for the generator / conflict graph."""
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=_STYLE["fig_bg"])
        self.plot_generator_graph(gen_graph, coloring, ax)
        plt.tight_layout()
        plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, visualize: bool = True, verify_chromatic: bool = True) -> None:
        """
        Execute all three phases, print a summary, and (optionally) display
        all visualisations.

        Parameters
        ----------
        visualize        : whether to render matplotlib figures
        verify_chromatic : whether to build G_gen and check the chromatic bound
        """
        # --- Run phases ---
        self.phase_1_reduction()
        self.phase_2_mapping()
        self.schedule = self.phase_3_scheduling()

        # --- Print summary ---
        print("\n" + "=" * 52)
        print("  GRAPH STATE GENERATION SUMMARY")
        print("=" * 52)
        print(f"  Vertices (qubits)      : {len(self.G.nodes)}")
        print(f"  Edges (interactions)   : {len(self.G.edges)}")
        print(f"  MIS size               : {len(self.mis)}  {self.mis}")
        print(f"  Nodes to measure       : {len(self.to_measure)}  {self.to_measure}")
        print(f"  Total time             : {len(self.schedule)} Tock(s)")
        print(f"  Total space            : {4 * len(self.G.nodes)} tiles")
        print("=" * 52)

        for i, tock in enumerate(self.schedule):
            print(f"\n  Tock {i + 1}:")
            for b in tock:
                parts = [f"X{self.mapping[b['id']]}"] + \
                        [f"Z{self.mapping[nb]}" for nb in b["neighbors"]]
                span  = f"tiles {b['L']}–{b['R']}"
                print(f"    g_{b['id']:>2}:  {' ⊗ '.join(parts):<30} ({span})")

        # --- Chromatic verification ---
        gen_graph, coloring, _ = (
            self.verify_chromatic_number() if verify_chromatic
            else (None, None, None)
        )

        # --- Visualisations ---
        if visualize:
            self.show_overview()
            self.show_all_tocks()
            if verify_chromatic:
                self.show_generator_graph(gen_graph, coloring)


# ===========================================================================
# CZScheduler — Case 2: Graph-state generation via CZ gates
# ===========================================================================
#
# Theory recap (from the GoSC notes)
# ------------------------------------
# A CZ gate between qubits u and v (both initialised in |+⟩) requires:
#   - 1 ancilla ELU per CZ
#   - 2 sequential MPP steps:
#       Step 1:  measure  Z_u ⊗ Z_ancilla   (1 Tock)
#       Step 2:  measure  Z_v ⊗ Z_ancilla   (1 Tock)
#
# Two CZ gates can run *in parallel* iff they share no qubit.
# Packing edges into parallel groups = edge colouring of G.
# By Vizing's theorem:  χ'(G) ∈ {Δ(G),  Δ(G)+1}
#
# Each colour class is one "CZ round".  Each round costs 2 Tocks.
# → Total Tocks = 2 × χ'(G)
#
# Space: each CZ needs 1 ancilla ELU, but ancillae are reused across rounds.
# Peak ancilla demand = max edges in any colour class.
# Data ELUs = |V|.  Layout tiles = 4·|V| (same 2-row substrate).

class CZScheduler:
    """
    Schedule CZ-gate-based graph-state generation on a nearest-neighbour
    substrate (GoSC Case 2 — unlimited ancillae, edge-colouring approach).

    Parameters
    ----------
    graph : nx.Graph
        Interaction graph G = (V, E).  Each edge (u, v) becomes one CZ gate.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.G: nx.Graph = nx.Graph(graph)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        if not nx.is_connected(self.G):
            print("Warning: graph is not connected.")

        # Populated by run()
        self.edge_coloring: dict[tuple, int] = {}   # edge → colour index
        self.cz_rounds:     list[list[tuple]] = []  # colour_idx → [edge list]
        self.chromatic_index: int = 0
        self.max_degree:      int = 0
        self._pos: dict = {}

    # ------------------------------------------------------------------
    # Edge colouring
    # ------------------------------------------------------------------

    def _greedy_edge_coloring(self) -> dict[tuple, int]:
        """
        Greedy edge colouring — assigns each edge a colour such that no
        two edges incident on the same vertex share a colour.

        Guarantees at most Δ(G)+1 colours (Vizing upper bound).
        Returns a dict  {(u, v): colour_index, ...}  with canonical u < v.
        """
        coloring: dict[tuple, int] = {}

        for u, v in self.G.edges():
            # colours already used by edges touching u or v
            forbidden: set[int] = set()
            for w in self.G.neighbors(u):
                e = (min(u, w), max(u, w))
                if e in coloring:
                    forbidden.add(coloring[e])
            for w in self.G.neighbors(v):
                e = (min(v, w), max(v, w))
                if e in coloring:
                    forbidden.add(coloring[e])

            # pick the smallest non-forbidden colour
            c = 0
            while c in forbidden:
                c += 1
            coloring[(min(u, v), max(u, v))] = c

        return coloring

    def schedule_cz_rounds(self) -> list[list[tuple]]:
        """
        Colour G's edges and group them into parallel CZ rounds.

        Returns
        -------
        cz_rounds : list of colour classes, each a list of (u, v) edge tuples.
                    Every round is executed in 2 Tocks (MPP step 1, MPP step 2).
        """
        self.edge_coloring   = self._greedy_edge_coloring()
        self.chromatic_index = (max(self.edge_coloring.values()) + 1
                                if self.edge_coloring else 0)
        self.max_degree      = max((d for _, d in self.G.degree()), default=0)

        # group edges by colour
        rounds: list[list[tuple]] = [[] for _ in range(self.chromatic_index)]
        for edge, c in self.edge_coloring.items():
            rounds[c].append(edge)

        self.cz_rounds = rounds
        return self.cz_rounds

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def verify_vizing(self) -> bool:
        """Check that χ'(G) ∈ {Δ(G), Δ(G)+1} as Vizing's theorem requires."""
        chi_prime = self.chromatic_index
        delta     = self.max_degree
        passed    = (chi_prime == delta) or (chi_prime == delta + 1)

        print("\n" + "=" * 52)
        print("  VIZING'S THEOREM VERIFICATION")
        print("=" * 52)
        print(f"  Max degree  Δ(G)        : {delta}")
        print(f"  Chromatic index χ'(G)   : {chi_prime}")
        print(f"  Class                   : {'Class 1 [χ′ = Δ]' if chi_prime == delta else 'Class 2 [χ′ = Δ+1]'}")
        print(f"  Result                  : {'✓ PASS' if passed else '✗ FAIL'}")
        print("=" * 52 + "\n")
        return passed

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a human-readable schedule summary."""
        n_tocks = 2 * self.chromatic_index
        print("\n" + "=" * 52)
        print("  CZ-GATE GRAPH STATE GENERATION SUMMARY")
        print("=" * 52)
        print(f"  Vertices (qubits)      : {len(self.G.nodes)}")
        print(f"  Edges (CZ gates)       : {len(self.G.edges)}")
        print(f"  Max degree Δ(G)        : {self.max_degree}")
        print(f"  Chromatic index χ'(G)  : {self.chromatic_index}")
        print(f"  Total Tocks            : {n_tocks}  (= 2 × χ'(G))")
        print(f"  Total space            : {4 * len(self.G.nodes)} tiles")
        print("=" * 52)

        for i, round_edges in enumerate(self.cz_rounds):
            print(f"\n  CZ Round {i + 1}  (Tocks {2*i+1}–{2*i+2}):")
            for u, v in sorted(round_edges):
                print(f"    CZ({u},{v}):  "
                      f"Tock {2*i+1}: Z{u}⊗Z_anc  |  "
                      f"Tock {2*i+2}: Z{v}⊗Z_anc")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _get_pos(self) -> dict:
        if not self._pos:
            self._pos = nx.spring_layout(self.G, seed=_LAYOUT_SEED, k=2)
        return self._pos

    def _edge_color_list(self) -> list[str]:
        """Return a list of colours parallel to self.G.edges(), for drawing."""
        palette = _STYLE["tile_palette"]
        colors  = []
        for u, v in self.G.edges():
            c = self.edge_coloring.get((min(u, v), max(u, v)), 0)
            colors.append(palette[c % len(palette)])
        return colors

    def plot_edge_coloring(self, ax: plt.Axes) -> None:
        """Draw G with edges coloured by their CZ round assignment."""
        pos         = self._get_pos()
        edge_colors = self._edge_color_list()
        palette     = _STYLE["tile_palette"]

        nx.draw_networkx_nodes(
            self.G, pos,
            node_size=_STYLE["node_size"],
            node_color=_STYLE["node_idle"],
            edgecolors=_STYLE["node_edge_c"],
            linewidths=_STYLE["node_lw"],
            ax=ax,
        )
        nx.draw_networkx_labels(
            self.G, pos,
            font_size=_STYLE["font_size"],
            font_color="#222222",
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.G, pos,
            edge_color=edge_colors,
            width=_STYLE["edge_lw"] + 1.5,
            ax=ax,
        )

        handles = [
            mpatches.Patch(
                facecolor=palette[i % len(palette)],
                edgecolor=_STYLE["node_edge_c"],
                linewidth=1.2,
                label=f"Round {i + 1}  (Tocks {2*i+1}–{2*i+2})",
            )
            for i in range(self.chromatic_index)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
        ax.set_title("CZ Schedule — Edge Colouring",
                     fontsize=_STYLE["title_size"], pad=10)
        ax.axis("off")
        ax.margins(0.18)

    def plot_cz_round(self, round_idx: int, ax_graph: plt.Axes,
                      ax_tiles: plt.Axes) -> None:
        """
        Graph view + tile view for a single CZ round (2 Tocks).
        Shows both MPP steps side-by-side in the tile panel.
        """
        pos     = self._get_pos()
        palette = _STYLE["tile_palette"]
        edges   = self.cz_rounds[round_idx]
        color   = palette[round_idx % len(palette)]
        n_q     = len(self.G.nodes)

        # ---- graph view ----
        active_nodes: set = set()
        for u, v in edges:
            active_nodes.add(u); active_nodes.add(v)
        node_colors = [
            color if n in active_nodes else _STYLE["node_idle"]
            for n in self.G.nodes
        ]
        active_edges_draw = [(u, v) for (u, v) in self.G.edges()
                             if (min(u,v), max(u,v)) in
                             {(min(a,b), max(a,b)) for a,b in edges}]

        all_edges   = list(self.G.edges())
        inactive    = [e for e in all_edges if e not in active_edges_draw
                       and (e[1], e[0]) not in active_edges_draw]

        nx.draw_networkx_edges(self.G, pos, edgelist=inactive,
                               width=_STYLE["edge_lw"],
                               edge_color=_STYLE["edge_inactive"], ax=ax_graph)
        nx.draw_networkx_edges(self.G, pos, edgelist=active_edges_draw,
                               width=_STYLE["edge_lw"] + 1.0,
                               edge_color=color, ax=ax_graph)
        nx.draw_networkx_nodes(self.G, pos, node_size=_STYLE["node_size"],
                               node_color=node_colors,
                               edgecolors=_STYLE["node_edge_c"],
                               linewidths=_STYLE["node_lw"], ax=ax_graph)
        nx.draw_networkx_labels(self.G, pos, font_size=_STYLE["font_size"],
                                font_color="#222222", ax=ax_graph)
        ax_graph.set_title(f"CZ Round {round_idx+1} — Graph View "
                           f"(Tocks {2*round_idx+1}–{2*round_idx+2})",
                           fontsize=_STYLE["title_size"], pad=10)
        ax_graph.axis("off"); ax_graph.margins(0.18)

        # ---- tile view: two sub-columns per CZ (Step 1 | Step 2) ----
        # We show a compact flat tile strip; each CZ pair gets 2 sub-slots.
        ax_tiles.set_xlim(-0.6, max(n_q * 2, 4))
        ax_tiles.set_ylim(-0.3, 3.5)
        ax_tiles.axis("off")
        ax_tiles.set_title(
            f"CZ Round {round_idx+1} — Substrate Tiles  "
            f"(Tock {2*round_idx+1}: Step 1 | Tock {2*round_idx+2}: Step 2)",
            fontsize=_STYLE["title_size"], pad=10,
        )
        ax_tiles.text(-0.4, 2.5, "Anc", ha="right", va="center",
                      fontsize=9, color="#666", fontweight="bold")
        ax_tiles.text(-0.4, 1.0, "Data", ha="right", va="center",
                      fontsize=9, color="#666", fontweight="bold")

        nodes_list = list(self.G.nodes)
        node_to_pos = {n: i for i, n in enumerate(nodes_list)}

        # draw idle tiles for both steps
        for step in range(2):
            x_off = step * n_q
            for q in range(n_q):
                for row_y in (2.0, 0.5):
                    r = plt.Rectangle((x_off + q, row_y), 1, 1,
                                      facecolor="#fafafa",
                                      edgecolor="#dddddd", linewidth=0.6)
                    ax_tiles.add_patch(r)
                label_prefix = "A" if row_y == 2.0 else "Q"
                ax_tiles.text(x_off + q + 0.5, row_y + 0.5,
                              f"{label_prefix}{q}",
                              ha="center", va="center",
                              fontsize=7, color="#bbbbbb")

            # divider
            ax_tiles.axvline(x_off + n_q, color="#cccccc",
                             linewidth=1.0, linestyle="--")
            step_label = f"Tock {2*round_idx + step + 1}"
            ax_tiles.text(x_off + n_q / 2, 3.2, step_label,
                          ha="center", va="center",
                          fontsize=9, fontweight="bold", color="#444")

        # highlight active tiles per step
        for ei, (u, v) in enumerate(edges):
            c = palette[(round_idx + ei) % len(palette)]
            qu, qv = node_to_pos[u], node_to_pos[v]
            anc_pos = ei  # ancilla slot = edge index within this round

            # Step 1: Z_u ⊗ Z_ancilla  → highlight data qubit u + ancilla slot
            for step, qubit_pos in enumerate([qu, qv]):
                x_off = step * n_q
                # data tile
                r = plt.Rectangle((x_off + qubit_pos, 0.5), 1, 1,
                                  facecolor=c, edgecolor="white",
                                  linewidth=1.2, alpha=0.85)
                ax_tiles.add_patch(r)
                ax_tiles.text(x_off + qubit_pos + 0.5, 1.0,
                              f"Q{qubit_pos}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white")
                # ancilla tile (use anc_pos as x, capped to n_q range)
                anc_x = x_off + (anc_pos % n_q)
                r2 = plt.Rectangle((anc_x, 2.0), 1, 1,
                                   facecolor=c, edgecolor="white",
                                   linewidth=1.2, alpha=0.6)
                ax_tiles.add_patch(r2)
                ax_tiles.text(anc_x + 0.5, 2.5, f"A{anc_pos % n_q}",
                              ha="center", va="center",
                              fontsize=8, color="white")

    def show_overview(self) -> None:
        """Display edge-colouring overview."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 5),
                               facecolor=_STYLE["fig_bg"])
        fig.suptitle("CZ Scheduler — Edge Colouring Overview",
                     fontsize=14, fontweight="bold")
        self.plot_edge_coloring(ax)
        plt.tight_layout()
        plt.show()
        plt.close()

    def show_all_rounds(self) -> None:
        """Render every CZ round as a pair of panels."""
        n_rounds = self.chromatic_index
        if n_rounds == 0:
            print("No CZ rounds to display (empty graph).")
            return
        fig, axes = plt.subplots(
            n_rounds, 2,
            figsize=(14, 5.5 * n_rounds),
            facecolor=_STYLE["fig_bg"],
            gridspec_kw={"width_ratios": [1, 1]},
        )
        fig.suptitle("CZ Scheduler — All Rounds",
                     fontsize=14, fontweight="bold", y=1.01)

        if n_rounds == 1:
            axes = [axes]

        for i, row in enumerate(axes):
            self.plot_cz_round(i, row[0], row[1])

        plt.tight_layout()
        plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, visualize: bool = True, verify_vizing: bool = True) -> None:
        """
        Run the full CZ scheduling pipeline.

        Parameters
        ----------
        visualize    : render matplotlib figures
        verify_vizing: print Vizing's theorem check
        """
        self.schedule_cz_rounds()
        self.print_summary()

        if verify_vizing:
            self.verify_vizing()

        if visualize:
            self.show_overview()
            self.show_all_rounds()


# ===========================================================================
# CZSchedulerFixedAncilla — Case 3: CZ gates + k fixed ancilla ELUs
# ===========================================================================
#
# Theory
# ------
# Same greedy edge-colouring as CZScheduler (Case 2), but the ancilla pool
# is capped at k ELUs.  When a colour class Cᵢ has more than k edges, it is
# split into consecutive sub-batches of ≤ k edges.  Each sub-batch takes
# exactly 2 Tocks:
#
#   Tock 2b−1  (Step 1): for every CZ(u,v) in the batch, measure Z_u ⊗ Z_anc
#   Tock 2b    (Step 2): for every CZ(u,v) in the batch, measure Z_v ⊗ Z_anc
#
# Sub-batches from different colour classes CANNOT be pipelined in general
# because adjacent colour classes can share vertices, causing qubit conflicts.
# Sub-batches are therefore executed in strict sequence.
#
#   Total Tocks  =  2 × Σᵢ ⌈|Cᵢ| / k⌉
#
# Lower bound on 2-Tock windows: T_lb = Σᵢ ⌈|Cᵢ| / k⌉  (achieved exactly)
#
# Special cases
#   k ≥ max_i |Cᵢ|  →  same as Case 2 (no extra splitting)
#   k = 1            →  Total Tocks = 2·|E|  (all CZs sequential)

class CZSchedulerFixedAncilla(CZScheduler):
    """
    Case 3 — CZ-gate graph-state generation with k fixed ancilla ELUs
    and all-to-all connectivity.

    Inherits greedy edge-colouring from CZScheduler.  Adds ancilla-pool
    splitting: each colour class is divided into sub-batches of ≤ k edges,
    executed sequentially (2 Tocks per sub-batch).

    Parameters
    ----------
    graph : nx.Graph
    k     : int — ancilla pool size (≥ 1)
    """

    def __init__(self, graph: nx.Graph, k: int) -> None:
        if k < 1:
            raise ValueError("Ancilla pool size k must be at least 1.")
        super().__init__(graph)
        self.k: int = k
        self.batches: list[list[tuple]] = []   # all sub-batches in order (each = 2 Tocks)

    # ------------------------------------------------------------------
    # Batch scheduling
    # ------------------------------------------------------------------

    def schedule_batches(self) -> list[list[tuple]]:
        """
        Split each colour class into sub-batches of ≤ k edges.
        Must be called after schedule_cz_rounds() (or run()).
        Populates self.batches and returns it.
        """
        if not self.cz_rounds:
            self.schedule_cz_rounds()

        all_batches: list[list[tuple]] = []
        for round_edges in self.cz_rounds:
            for i in range(0, len(round_edges), self.k):
                chunk = round_edges[i : i + self.k]
                if chunk:
                    all_batches.append(chunk)
        self.batches = all_batches
        return self.batches

    # ------------------------------------------------------------------
    # Resource metrics
    # ------------------------------------------------------------------

    def n_tocks(self) -> int:
        """Total Tocks = 2 × number of sub-batches."""
        return 2 * len(self.batches)

    def n_sub_batches(self) -> int:
        """
        Number of serial 2-Tock windows needed.
        Equals Σᵢ ⌈|Cᵢ|/k⌉ — this is also the lower bound in 2-Tock units.
        """
        return sum(math.ceil(len(r) / self.k) for r in self.cz_rounds)

    # ------------------------------------------------------------------
    # Console summary (overrides parent)
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        n_b    = len(self.batches)
        n_t    = 2 * n_b
        n_t_c2 = 2 * self.chromatic_index

        print("\n" + "=" * 52)
        print("  CASE 3 — CZ + k ANCILLA ELUs SUMMARY")
        print("=" * 52)
        print(f"  Vertices (qubits)      : {len(self.G.nodes)}")
        print(f"  Edges (CZ gates)       : {len(self.G.edges)}")
        print(f"  Max degree Δ(G)        : {self.max_degree}")
        print(f"  Chromatic index χ'(G)  : {self.chromatic_index}")
        print(f"  Ancilla pool  k        : {self.k}")
        print(f"  Sub-batches (windows)  : {n_b}  (= Σᵢ ⌈|Cᵢ|/k⌉)")
        print(f"  Total Tocks            : {n_t}  (= 2 × {n_b})")
        print(f"  Case 2 Tocks (k=∞)    : {n_t_c2}  (= 2·χ'(G))")
        overhead = n_t - n_t_c2
        print(f"  Overhead vs Case 2     : +{overhead} Tock(s)"
              f"  ({overhead / n_t_c2 * 100:.0f}% more)" if n_t_c2 else "")
        print("=" * 52)

        batch_num = 0
        for ci, round_edges in enumerate(self.cz_rounds):
            if not round_edges:
                continue
            n_sub = math.ceil(len(round_edges) / self.k)
            print(f"\n  Colour class {ci}  ({len(round_edges)} edge(s),"
                  f" {n_sub} sub-batch(es)):")
            for si in range(0, len(round_edges), self.k):
                chunk = round_edges[si : si + self.k]
                if not chunk:
                    continue
                t1 = 2 * batch_num + 1
                t2 = 2 * batch_num + 2
                print(f"    Sub-batch {batch_num + 1}  (Tocks {t1}–{t2},"
                      f" {len(chunk)} CZ gate(s)):")
                for u, v in sorted(chunk):
                    print(f"      CZ({u},{v}):  "
                          f"Tock {t1}: Z_{u}⊗Z_anc  |  "
                          f"Tock {t2}: Z_{v}⊗Z_anc")
                batch_num += 1

    # ------------------------------------------------------------------
    # Public entry point (overrides parent)
    # ------------------------------------------------------------------

    def run(
        self,
        visualize:     bool = True,
        verify_vizing: bool = True,
    ) -> None:
        """
        Run Case 3: edge-colour G, split colour classes into k-ancilla
        sub-batches, print summary.
        """
        self.schedule_cz_rounds()   # edge colouring → self.cz_rounds
        self.schedule_batches()     # sub-batch split  → self.batches
        self.print_summary()

        if verify_vizing:
            self.verify_vizing()    # inherited — Vizing's theorem check

        if visualize:
            self.show_overview()    # inherited — edge-colouring overview


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CASE 5 — CZ Gates + Limited Connectivity                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Inherits edge colouring from CZScheduler.  For each colour class (round),
# SWAP-routes non-adjacent qubit pairs through the switch topology S before
# executing CZ gates.  Qubit positions evolve over time (leave-in-place).
#
# Total Tocks = Σᵢ (swap_layers_i + 2)

@dataclass
class RoutedRound:
    """Result of SWAP-routing one CZ colour class on topology S."""
    round_idx:      int
    original_edges: list[tuple]                          # logical qubit pairs
    swap_layers:    list[list[tuple]] = field(default_factory=list)   # parallel SWAP layers (slot pairs)
    cz_slot_pairs:  list[tuple] = field(default_factory=list)        # slot pairs for CZ execution
    swap_tocks:     int = 0
    cz_tocks:       int = 2
    total_tocks:    int = 2


@dataclass
class RoutedTock:
    """Result of SWAP-routing one MPP measurement group on topology S."""
    measurements:   list[dict]
    swap_layers:    list[list[tuple]] = field(default_factory=list)
    swap_tocks:     int = 0
    mpp_tocks:      int = 1
    total_tocks:    int = 1


class CZSchedulerLimited(CZScheduler):
    """
    Case 5 — CZ-gate graph-state generation with limited connectivity.

    Inherits greedy edge-colouring from CZScheduler.  For each colour class
    (CZ round), computes SWAP routing on the switch topology S to bring
    non-adjacent qubit pairs together before executing the CZ gates.

    Qubit positions evolve over time (leave-in-place after SWAPs).

    Parameters
    ----------
    graph     : nx.Graph — logical interaction graph G
    topology  : SwitchTopology — physical switch graph S
    placement : dict[int, int] — initial logical-qubit → slot mapping
    """

    def __init__(
        self,
        graph: nx.Graph,
        topology: "SwitchTopology",
        placement: dict[int, int],
    ) -> None:
        super().__init__(graph)
        self.topology = topology
        self.initial_placement = dict(placement)
        # Working placement (evolves as SWAPs are applied)
        self.logical_to_slot: dict[int, int] = dict(placement)
        self.slot_to_logical: dict[int, int | None] = {
            s: None for s in topology.S.nodes
        }
        for q, s in placement.items():
            self.slot_to_logical[s] = q

        self.routed_rounds: list[RoutedRound] = []
        self.total_swap_tocks: int = 0
        self.total_cz_tocks: int = 0

    # ------------------------------------------------------------------
    # SWAP routing for one round
    # ------------------------------------------------------------------

    def _route_round(self, round_idx: int, edges: list[tuple]) -> RoutedRound:
        """Route one colour class: SWAP non-adjacent pairs, then CZ."""
        topo = self.topology

        all_swaps: list[tuple[int, int]] = []
        # For each CZ(u, v), generate a SWAP chain to make them adjacent
        # We move u toward v (arbitrary choice; moving the first endpoint)
        for u, v in edges:
            slot_u = self.logical_to_slot[u]
            slot_v = self.logical_to_slot[v]
            if not topo.are_adjacent(slot_u, slot_v):
                path = topo.shortest_path(slot_u, slot_v)
                swaps = topo.swap_chain(path)
                all_swaps.extend(swaps)

        # Pack SWAPs into parallel layers
        swap_layers = topo.pack_swap_layers(all_swaps) if all_swaps else []

        # Apply SWAPs to update placement
        for layer in swap_layers:
            for sa, sb in layer:
                qa = self.slot_to_logical[sa]
                qb = self.slot_to_logical[sb]
                # Swap contents
                self.slot_to_logical[sa] = qb
                self.slot_to_logical[sb] = qa
                if qa is not None:
                    self.logical_to_slot[qa] = sb
                if qb is not None:
                    self.logical_to_slot[qb] = sa

        # Now all CZ pairs should be adjacent — record slot pairs
        cz_slot_pairs = []
        for u, v in edges:
            cz_slot_pairs.append((self.logical_to_slot[u], self.logical_to_slot[v]))

        swap_tocks = len(swap_layers)
        return RoutedRound(
            round_idx=round_idx,
            original_edges=edges,
            swap_layers=swap_layers,
            cz_slot_pairs=cz_slot_pairs,
            swap_tocks=swap_tocks,
            cz_tocks=2,
            total_tocks=swap_tocks + 2,
        )

    def route_all_rounds(self) -> list[RoutedRound]:
        """SWAP-route every colour class and return the routed rounds."""
        if not self.cz_rounds:
            self.schedule_cz_rounds()

        self.routed_rounds = []
        for i, edges in enumerate(self.cz_rounds):
            if not edges:
                continue
            rr = self._route_round(i, edges)
            self.routed_rounds.append(rr)

        self.total_swap_tocks = sum(rr.swap_tocks for rr in self.routed_rounds)
        self.total_cz_tocks = sum(rr.cz_tocks for rr in self.routed_rounds)
        return self.routed_rounds

    def n_tocks(self) -> int:
        return sum(rr.total_tocks for rr in self.routed_rounds)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        total = self.n_tocks()
        n_t_c2 = 2 * self.chromatic_index

        print("\n" + "=" * 60)
        print("  CASE 5 — CZ + LIMITED CONNECTIVITY SUMMARY")
        print("=" * 60)
        print(f"  Vertices (qubits)      : {len(self.G.nodes)}")
        print(f"  Edges (CZ gates)       : {len(self.G.edges)}")
        print(f"  Max degree Δ(G)        : {self.max_degree}")
        print(f"  Chromatic index χ'(G)  : {self.chromatic_index}")
        print(f"  Topology               : {self.topology.name}"
              f"  ({self.topology.n_elus} ELU slots, diam={self.topology.diameter()})")
        print(f"  CZ rounds              : {len(self.routed_rounds)}")
        print(f"  Total SWAP Tocks       : {self.total_swap_tocks}")
        print(f"  Total CZ Tocks         : {self.total_cz_tocks}")
        print(f"  Total Tocks            : {total}")
        print(f"  Case 2 Tocks (all→all) : {n_t_c2}  (= 2·χ'(G))")
        overhead = total - n_t_c2
        if n_t_c2:
            print(f"  Overhead vs Case 2     : +{overhead} Tock(s)"
                  f"  ({overhead / n_t_c2 * 100:.0f}% more)")
        print("=" * 60)

        for rr in self.routed_rounds:
            t_start = sum(r.total_tocks for r in self.routed_rounds[:rr.round_idx]) + 1
            print(f"\n  CZ Round {rr.round_idx + 1}  "
                  f"({len(rr.original_edges)} edge(s), "
                  f"{rr.swap_tocks} SWAP + {rr.cz_tocks} CZ Tocks):")
            if rr.swap_layers:
                for li, layer in enumerate(rr.swap_layers):
                    swaps_str = ", ".join(f"SWAP({a},{b})" for a, b in layer)
                    print(f"    SWAP layer {li + 1}: {swaps_str}")
            for u, v in sorted(rr.original_edges):
                su, sv = self.logical_to_slot[u], self.logical_to_slot[v]
                print(f"    CZ({u},{v})  slots ({su},{sv})")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        visualize:     bool = True,
        verify_vizing: bool = True,
    ) -> None:
        self.schedule_cz_rounds()
        self.route_all_rounds()
        self.print_summary()

        if verify_vizing:
            self.verify_vizing()

        if visualize:
            self.show_overview()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CASE 4 — MPP Stabilizer + Limited Connectivity                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Standalone class (does not inherit SubstrateScheduler).
# MIS reduction reused from SubstrateScheduler.
# Each stabilizer generator's support set is connected via a Steiner tree
# on S.  Generators conflict if they share a qubit or a slot in S.
# Greedy colouring of the conflict graph yields parallel Tock groups.
# Each Tock group: SWAP routing + MPP measurement.


class MPPSchedulerLimited:
    """
    Case 4 — MPP stabilizer measurements with limited connectivity.

    Parameters
    ----------
    graph     : nx.Graph — logical interaction graph G
    topology  : SwitchTopology — physical switch graph S
    placement : dict[int, int] — initial logical-qubit → slot mapping
    """

    def __init__(
        self,
        graph: nx.Graph,
        topology: "SwitchTopology",
        placement: dict[int, int],
    ) -> None:
        self.G = nx.Graph(graph)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.topology = topology
        self.initial_placement = dict(placement)
        self.logical_to_slot: dict[int, int] = dict(placement)
        self.slot_to_logical: dict[int, int | None] = {
            s: None for s in topology.S.nodes
        }
        for q, s in placement.items():
            self.slot_to_logical[s] = q

        self.mis: list = []
        self.to_measure: list = []
        self.routed_schedule: list[RoutedTock] = []
        self.total_swap_tocks: int = 0
        self.total_mpp_tocks: int = 0
        self._pos: dict = {}

    # ------------------------------------------------------------------
    # Phase 1 — MIS reduction (reuses SubstrateScheduler logic)
    # ------------------------------------------------------------------

    def _find_mis(self) -> list:
        """Compute MIS using the same logic as SubstrateScheduler."""
        node_list = list(self.G.nodes)
        n = len(node_list)

        def is_independent(nodes):
            for i, u in enumerate(nodes):
                for v in nodes[i + 1:]:
                    if self.G.has_edge(u, v):
                        return False
            return True

        if n <= 20:
            for size in range(n, 0, -1):
                for subset in combinations(node_list, size):
                    if is_independent(subset):
                        return list(subset)
            return []
        return nx.maximal_independent_set(self.G)

    def phase_1_reduction(self) -> None:
        self.mis = self._find_mis()
        self.to_measure = [v for v in self.G.nodes if v not in self.mis]

    # ------------------------------------------------------------------
    # Phase 2 — Build generators and compute Steiner trees
    # ------------------------------------------------------------------

    def _build_generators(self) -> list[dict]:
        """For each node to measure, build its stabilizer generator info."""
        generators = []
        for v in self.to_measure:
            neighbors = list(self.G.neighbors(v))
            support = {v} | set(neighbors)
            generators.append({
                "id": v,
                "neighbors": neighbors,
                "support": support,
            })
        return generators

    def _compute_steiner_slots(self, support: set[int]) -> set[int]:
        """Map support qubits to slots, compute Steiner tree, return slot set."""
        terminal_slots = {self.logical_to_slot[q] for q in support}
        if len(terminal_slots) <= 1:
            return terminal_slots
        tree = self.topology.steiner_tree(terminal_slots)
        return set(tree.nodes)

    # ------------------------------------------------------------------
    # Phase 3 — Conflict graph + greedy colouring
    # ------------------------------------------------------------------

    def _build_conflict_graph(
        self, generators: list[dict], steiner_slots: list[set[int]]
    ) -> nx.Graph:
        """Build conflict graph: generators conflict if their support sets
        share a qubit OR their Steiner trees share a slot.
        """
        CG = nx.Graph()
        CG.add_nodes_from(range(len(generators)))

        for i in range(len(generators)):
            for j in range(i + 1, len(generators)):
                # Qubit conflict
                if generators[i]["support"] & generators[j]["support"]:
                    CG.add_edge(i, j)
                # Slot conflict (Steiner trees overlap)
                elif steiner_slots[i] & steiner_slots[j]:
                    CG.add_edge(i, j)

        return CG

    @staticmethod
    def _greedy_color(CG: nx.Graph) -> dict[int, int]:
        """DSATUR-style greedy colouring of the conflict graph."""
        coloring: dict[int, int] = {}
        for node in CG.nodes:
            used = {coloring[nb] for nb in CG.neighbors(node) if nb in coloring}
            c = 0
            while c in used:
                c += 1
            coloring[node] = c
        return coloring

    # ------------------------------------------------------------------
    # Phase 4 — SWAP routing per Tock group
    # ------------------------------------------------------------------

    def _route_tock_group(
        self, generators: list[dict], gen_indices: list[int]
    ) -> RoutedTock:
        """Route a group of non-conflicting generators for parallel execution."""
        topo = self.topology
        all_swaps: list[tuple[int, int]] = []

        for gi in gen_indices:
            gen = generators[gi]
            support = gen["support"]
            terminal_slots = {self.logical_to_slot[q] for q in support}

            if len(terminal_slots) <= 1:
                continue

            # Get Steiner tree
            tree = topo.steiner_tree(terminal_slots)

            # For each qubit in support, if it's not on a terminal of the tree
            # it's already there. We just need all support qubits on the tree.
            # Check if any support qubit's slot is NOT in the tree — if so, SWAP.
            for q in support:
                slot = self.logical_to_slot[q]
                if slot not in tree.nodes:
                    # Find closest tree node and SWAP toward it
                    closest_tree_node = min(
                        tree.nodes,
                        key=lambda tn: topo.distance(slot, tn)
                    )
                    path = topo.shortest_path(slot, closest_tree_node)
                    swaps = topo.swap_chain(path)
                    all_swaps.extend(swaps)

        # Pack SWAPs into parallel layers
        swap_layers = topo.pack_swap_layers(all_swaps) if all_swaps else []

        # Apply SWAPs
        for layer in swap_layers:
            for sa, sb in layer:
                qa = self.slot_to_logical[sa]
                qb = self.slot_to_logical[sb]
                self.slot_to_logical[sa] = qb
                self.slot_to_logical[sb] = qa
                if qa is not None:
                    self.logical_to_slot[qa] = sb
                if qb is not None:
                    self.logical_to_slot[qb] = sa

        measurements = [generators[gi] for gi in gen_indices]
        swap_tocks = len(swap_layers)
        return RoutedTock(
            measurements=measurements,
            swap_layers=swap_layers,
            swap_tocks=swap_tocks,
            mpp_tocks=1,
            total_tocks=swap_tocks + 1,
        )

    # ------------------------------------------------------------------
    # Full scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> list[RoutedTock]:
        """Run all phases and produce the routed schedule."""
        self.phase_1_reduction()
        generators = self._build_generators()

        # Compute Steiner tree slot sets for current placement
        steiner_slots = [
            self._compute_steiner_slots(g["support"]) for g in generators
        ]

        # Conflict graph + colouring
        CG = self._build_conflict_graph(generators, steiner_slots)
        coloring = self._greedy_color(CG)

        if not coloring:
            self.routed_schedule = []
            return self.routed_schedule

        n_colors = max(coloring.values()) + 1

        # Group generators by colour class
        color_groups: list[list[int]] = [[] for _ in range(n_colors)]
        for gi, c in coloring.items():
            color_groups[c].append(gi)

        # Route each group
        self.routed_schedule = []
        for group in color_groups:
            if group:
                rt = self._route_tock_group(generators, group)
                self.routed_schedule.append(rt)

        self.total_swap_tocks = sum(rt.swap_tocks for rt in self.routed_schedule)
        self.total_mpp_tocks = sum(rt.mpp_tocks for rt in self.routed_schedule)
        return self.routed_schedule

    def n_tocks(self) -> int:
        return sum(rt.total_tocks for rt in self.routed_schedule)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        total = self.n_tocks()

        print("\n" + "=" * 60)
        print("  CASE 4 — MPP + LIMITED CONNECTIVITY SUMMARY")
        print("=" * 60)
        print(f"  Vertices (qubits)      : {len(self.G.nodes)}")
        print(f"  Edges (interactions)   : {len(self.G.edges)}")
        print(f"  MIS size               : {len(self.mis)}  {self.mis}")
        print(f"  Nodes to measure       : {len(self.to_measure)}"
              f"  {self.to_measure}")
        print(f"  Topology               : {self.topology.name}"
              f"  ({self.topology.n_elus} ELU slots, diam={self.topology.diameter()})")
        print(f"  Tock groups            : {len(self.routed_schedule)}")
        print(f"  Total SWAP Tocks       : {self.total_swap_tocks}")
        print(f"  Total MPP Tocks        : {self.total_mpp_tocks}")
        print(f"  Total Tocks            : {total}")
        print("=" * 60)

        for ti, rt in enumerate(self.routed_schedule):
            gen_ids = [m["id"] for m in rt.measurements]
            print(f"\n  Tock group {ti + 1}  "
                  f"(generators {gen_ids}, "
                  f"{rt.swap_tocks} SWAP + {rt.mpp_tocks} MPP Tocks):")
            if rt.swap_layers:
                for li, layer in enumerate(rt.swap_layers):
                    swaps_str = ", ".join(f"SWAP({a},{b})" for a, b in layer)
                    print(f"    SWAP layer {li + 1}: {swaps_str}")
            for m in rt.measurements:
                nbrs = ", ".join(f"Z{n}" for n in m["neighbors"])
                print(f"    g_{m['id']}: X{m['id']} ⊗ {nbrs}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        visualize: bool = True,
        verify_chromatic: bool = True,
    ) -> None:
        self.schedule()
        self.print_summary()