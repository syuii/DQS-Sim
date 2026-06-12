"""
topology.py
===========
Switch topology model for limited-connectivity scheduling (Cases 4 & 5).

A SwitchTopology wraps a physical graph S whose nodes are ELU slots and
whose edges define which pairs can communicate in a single Tock.  Provides
shortest-path routing, SWAP-layer packing, Steiner tree approximation,
and initial qubit placement strategies.

Supported topologies
--------------------
  linear   — path graph P_n
  star     — star graph S_n  (hub = slot 0)
  grid     — ceil(√n) × ceil(√n) grid  (some slots may be empty padding)
  complete — K_n  (all-to-all; equivalent to Cases 1–3)
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

_TOPOLOGY_BUILDERS: dict[str, Any] = {}


def _register(name: str):
    def decorator(fn):
        _TOPOLOGY_BUILDERS[name] = fn
        return fn
    return decorator


@_register("linear")
def _build_linear(n: int) -> nx.Graph:
    return nx.path_graph(n)


@_register("star")
def _build_star(n: int) -> nx.Graph:
    return nx.star_graph(n - 1)          # star_graph(k) has k+1 nodes


@_register("grid")
def _build_grid(n: int) -> nx.Graph:
    side = math.ceil(math.sqrt(n))
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    # Remove extra nodes beyond n (grid may be slightly oversized)
    for node in list(G.nodes):
        if node >= n:
            G.remove_node(node)
    return G


@_register("complete")
def _build_complete(n: int) -> nx.Graph:
    return nx.complete_graph(n)


TOPOLOGY_NAMES = list(_TOPOLOGY_BUILDERS.keys())


# ---------------------------------------------------------------------------
# SwitchTopology
# ---------------------------------------------------------------------------

class SwitchTopology:
    """Physical switch graph S connecting ELU slots.

    Parameters
    ----------
    name    : topology type  ("linear", "star", "grid", "complete")
    n_elus  : number of ELU slots (nodes in S, must be >= |V| of logical graph)
    """

    def __init__(self, name: str, n_elus: int) -> None:
        if name not in _TOPOLOGY_BUILDERS:
            raise ValueError(
                f"Unknown topology '{name}'. Options: {', '.join(TOPOLOGY_NAMES)}"
            )
        if n_elus < 1:
            raise ValueError("n_elus must be >= 1.")

        self.name: str = name
        self.n_elus: int = n_elus
        self.S: nx.Graph = _TOPOLOGY_BUILDERS[name](n_elus)

        if not nx.is_connected(self.S):
            raise ValueError(
                f"Topology '{name}' with n_elus={n_elus} is disconnected."
            )

        # Precompute all-pairs shortest paths (fast for N <= ~200)
        self._sp_length: dict[int, dict[int, int]] = dict(
            nx.all_pairs_shortest_path_length(self.S)
        )
        self._sp_path: dict[int, dict[int, list[int]]] = dict(
            nx.all_pairs_shortest_path(self.S)
        )

    # ── Queries ───────────────────────────────────────────────────────────

    def are_adjacent(self, u: int, v: int) -> bool:
        return self.S.has_edge(u, v)

    def distance(self, u: int, v: int) -> int:
        return self._sp_length[u][v]

    def shortest_path(self, u: int, v: int) -> list[int]:
        return self._sp_path[u][v]

    def diameter(self) -> int:
        return max(max(d.values()) for d in self._sp_length.values())

    # ── Steiner tree (for Case 4 MPP routing) ────────────────────────────

    def steiner_tree(self, terminals: set[int]) -> nx.Graph:
        """Approximate Steiner tree spanning *terminals* in S.

        Uses NetworkX's 2-approximation.  Falls back to the full
        subgraph induced by terminals if the approximation fails
        (e.g., single-node terminal set).
        """
        if len(terminals) <= 1:
            T = nx.Graph()
            T.add_nodes_from(terminals)
            return T
        return nx.approximation.steiner_tree(
            self.S, list(terminals), method="mehlhorn"
        )

    # ── SWAP routing ─────────────────────────────────────────────────────

    @staticmethod
    def swap_chain(path: list[int]) -> list[tuple[int, int]]:
        """Given a shortest path [s0, s1, …, sN], return the SWAP pairs
        that move the qubit at s0 to s(N-1) (adjacent to sN).

        Each SWAP exchanges the contents of two adjacent slots.
        After all SWAPs, the qubit originally at s0 is now at s(N-1).
        """
        # We need len(path)-2 SWAPs: (s0,s1), (s1,s2), …, (s_{N-2},s_{N-1})
        return [(path[i], path[i + 1]) for i in range(len(path) - 2)]

    @staticmethod
    def pack_swap_layers(
        swaps: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Pack a flat list of SWAPs into parallel layers.

        Each layer is a set of SWAPs that can execute simultaneously
        (no slot appears in more than one SWAP per layer).
        Returns the minimum-greedy layering.
        """
        remaining = list(swaps)
        layers: list[list[tuple[int, int]]] = []
        while remaining:
            layer: list[tuple[int, int]] = []
            used: set[int] = set()
            still_remaining: list[tuple[int, int]] = []
            for a, b in remaining:
                if a not in used and b not in used:
                    layer.append((a, b))
                    used.add(a)
                    used.add(b)
                else:
                    still_remaining.append((a, b))
            layers.append(layer)
            remaining = still_remaining
        return layers

    # ── Pretty print ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"SwitchTopology('{self.name}', n_elus={self.n_elus}, "
                f"diam={self.diameter()})")


# ---------------------------------------------------------------------------
# Qubit placement strategies
# ---------------------------------------------------------------------------

def place_identity(G: nx.Graph, topo: SwitchTopology) -> dict[int, int]:
    """Map logical qubit i → ELU slot i (1:1, simplest baseline)."""
    nodes = sorted(G.nodes)
    return {q: q for q in nodes}


def place_spectral(G: nx.Graph, topo: SwitchTopology) -> dict[int, int]:
    """Spectral ordering of G mapped onto a BFS ordering of S.

    Fiedler-vector ordering of G produces a 1D sequence that keeps
    heavily-connected qubits close.  We lay that sequence along a BFS
    traversal of S so that adjacent qubits in the spectral order land
    on nearby slots in S.
    """
    if len(G.nodes) <= 2:
        return place_identity(G, topo)

    spectral_order = nx.spectral_ordering(G)

    # BFS ordering of S from the "most central" node (min eccentricity)
    center = nx.center(topo.S)[0]
    bfs_order = list(nx.bfs_tree(topo.S, center).nodes)

    mapping: dict[int, int] = {}
    for i, logical_q in enumerate(spectral_order):
        mapping[logical_q] = bfs_order[i]
    return mapping


def place_greedy(G: nx.Graph, topo: SwitchTopology) -> dict[int, int]:
    """Greedy placement: place highest-degree logical qubit on the
    highest-degree slot in S, then BFS through G, always picking the
    closest available slot adjacent to already-placed neighbors.
    """
    if len(G.nodes) == 0:
        return {}

    # Sort logical qubits by degree (descending)
    logical_by_deg = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)

    # Sort slots by degree (descending)
    slots_by_deg = sorted(topo.S.nodes, key=lambda n: topo.S.degree(n), reverse=True)

    mapping: dict[int, int] = {}
    used_slots: set[int] = set()

    # Place the first (highest-degree) qubit on the best slot
    first_q = logical_by_deg[0]
    first_slot = slots_by_deg[0]
    mapping[first_q] = first_slot
    used_slots.add(first_slot)

    # BFS through G starting from the first qubit
    visited = {first_q}
    queue = [first_q]
    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

                if neighbor not in mapping:
                    # Find the closest available slot to already-placed neighbors
                    placed_neighbor_slots = [
                        mapping[n] for n in G.neighbors(neighbor)
                        if n in mapping
                    ]
                    if placed_neighbor_slots:
                        best_slot = None
                        best_dist = float("inf")
                        for slot in topo.S.nodes:
                            if slot in used_slots:
                                continue
                            d = min(topo.distance(slot, ps) for ps in placed_neighbor_slots)
                            if d < best_dist:
                                best_dist = d
                                best_slot = slot
                        if best_slot is not None:
                            mapping[neighbor] = best_slot
                            used_slots.add(best_slot)

    # Handle any disconnected components (fallback: assign remaining slots)
    for q in G.nodes:
        if q not in mapping:
            for slot in topo.S.nodes:
                if slot not in used_slots:
                    mapping[q] = slot
                    used_slots.add(slot)
                    break

    return mapping


PLACEMENT_STRATEGIES = {
    "identity": place_identity,
    "spectral": place_spectral,
    "greedy":   place_greedy,
}
