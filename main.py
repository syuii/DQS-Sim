"""
main.py
=======
Demonstrates both scheduling strategies.
"""

import networkx as nx
from substrate_scheduler import SubstrateScheduler, CZScheduler

# Pick your graph here
#G = nx.path_graph(10)
#G = nx.cycle_graph(100)
G = nx.star_graph(100)

# --- Case 1 ---
print("\n" + "#"*52)
print("#  CASE 1 — Stabilizer MPP Scheduler")
print("#"*52)
scheduler = SubstrateScheduler(G)
scheduler.run(visualize=False, verify_chromatic=True)

# --- Case 2 ---
print("\n" + "#"*52)
print("#  CASE 2 — CZ Gate Scheduler")
print("#"*52)
cz = CZScheduler(G)
cz.run(visualize=False, verify_vizing=True)

# --- Comparison ---
tocks_c1 = len(scheduler.schedule)
tocks_c2 = 2 * cz.chromatic_index
print("\n" + "="*52)
print("  COMPARISON")
print("="*52)
print(f"  Graph        : |V|={len(G.nodes)}, |E|={len(G.edges)}")
print(f"  Case 1 Tocks : {tocks_c1}  (MPP/stabilizer)")
print(f"  Case 2 Tocks : {tocks_c2}  (CZ gates, χ'={cz.chromatic_index})")
if tocks_c1 > 0:
    print(f"  Ratio C2/C1  : {tocks_c2 / tocks_c1:.2f}×")
print("="*52 + "\n")
