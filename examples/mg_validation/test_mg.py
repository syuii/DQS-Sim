"""Validate Misra-Gries colouring: proper, complete, and <= Delta+1 everywhere."""
import sys

sys.path.insert(0, r".")
import networkx as nx

from dqs_sim.schedulers import CZScheduler, misra_gries_edge_coloring


def check(G, name):
    delta = max((d for _, d in G.degree()), default=0)
    col = misra_gries_edge_coloring(G)
    # complete: every edge coloured exactly once
    keyed = {(min(u, v), max(u, v)) for u, v in G.edges()}
    assert set(col) == keyed, f"{name}: edges missing/extra"
    # proper: no two edges at a vertex share a colour
    for x in G.nodes():
        cs = [col[(min(x, w), max(x, w))] for w in G.neighbors(x)]
        assert len(cs) == len(set(cs)), f"{name}: clash at vertex {x}"
    n_col = max(col.values()) + 1 if col else 0
    assert n_col <= delta + 1, f"{name}: {n_col} colours > Delta+1 = {delta+1}"
    # greedy comparison
    g = CZScheduler(G)
    g.schedule_cz_rounds()
    print(f"{name:22s} Delta={delta:2d}  greedy={g.chromatic_index:2d}  "
          f"MG={n_col:2d}  (Vizing bound {delta+1})")
    return g.chromatic_index, n_col


families = [
    ("path P8", nx.path_graph(8)),
    ("cycle C6", nx.cycle_graph(6)),
    ("cycle C7 (odd)", nx.cycle_graph(7)),
    ("star K1,7", nx.star_graph(7)),
    ("complete K5 (odd)", nx.complete_graph(5)),
    ("complete K8 (even)", nx.complete_graph(8)),
    ("complete K9 (odd)", nx.complete_graph(9)),
    ("grid 3x3", nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))),
    ("grid 5x5", nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5))),
    ("random G(8,.35)", nx.erdos_renyi_graph(8, 0.35, seed=42)),
    ("random G(12,.5)", nx.erdos_renyi_graph(12, 0.5, seed=7)),
    ("random G(20,.3)", nx.erdos_renyi_graph(20, 0.3, seed=3)),
    ("petersen", nx.petersen_graph()),
    ("hypercube Q4", nx.convert_node_labels_to_integers(nx.hypercube_graph(4))),
]

# incidence graphs from the thesis crossover study
sys.path.insert(0, r".\examples")
from thesis_sweeps import build_incidence
for q in (2, 3, 4, 5):
    families.append((f"PG(2,{q})", build_incidence(q)))

for name, G in families:
    check(G, name)

# K5 headline: vizing mode must reach the true optimum (class 2 -> chi' = 5)
s = CZScheduler(nx.complete_graph(5), coloring_mode="vizing")
s.schedule_cz_rounds()
assert s.chromatic_index == 5, f"K5 vizing mode gave {s.chromatic_index}"
print(f"\nK5 vizing mode: {s.chromatic_index} rounds -> T2 = {2*s.chromatic_index} Tocks (optimum)")
print("ALL CHECKS PASSED")
