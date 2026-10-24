"""Three checks: complete-graph scaling with MG, K5 colouring visual, timings."""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from dqs_sim.schedulers import (CZScheduler, SubstrateScheduler,
                                misra_gries_edge_coloring)

mpl.rcParams.update({"font.family": "sans-serif", "savefig.dpi": 200,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
NAVY, ORANGE, GREEN, GREY = "#285A96", "#C1853C", "#4E8C6A", "#C9CDD4"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

import contextlib
import io


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


# 1. complete-graph scaling
ns, t1s, t2g, t2v = [], [], [], []
for n in range(4, 17):
    G = nx.complete_graph(n)
    s = SubstrateScheduler(G)
    quiet(s.phase_1_reduction)
    quiet(s.phase_2_mapping)
    sched = quiet(s.phase_3_scheduling)
    t1 = len(sched)
    g = CZScheduler(G); quiet(g.schedule_cz_rounds)
    v = CZScheduler(G, coloring_mode="vizing"); quiet(v.schedule_cz_rounds)
    ns.append(n); t1s.append(t1)
    t2g.append(2 * g.chromatic_index); t2v.append(2 * v.chromatic_index)

print("n    T1(MPP)  T2 greedy  T2 vizing")
for row in zip(ns, t1s, t2g, t2v):
    print("{:<5}{:<9}{:<11}{}".format(*row))

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(ns, t1s, "o-", color=NAVY, lw=2.4, ms=5, label="Case 1 — MPP")
ax.plot(ns, t2g, "s-", color=ORANGE, lw=2.4, ms=5, label="Case 2 — CZ (greedy)")
ax.plot(ns, t2v, "^-", color=GREEN, lw=2.4, ms=5,
        label="Case 2 — CZ (Misra-Gries)")
ax.set_xlabel("complete graph size n", fontsize=12)
ax.set_ylabel("Tocks", fontsize=12)
ax.set_title("Complete graphs: MPP vs CZ under both colourings",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, frameon=False)
ax.grid(linestyle="--", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "mg_complete_scaling.png", bbox_inches="tight")
plt.close(fig)

# 2. K5 colouring, greedy vs MG, drawn on the graph
G5 = nx.complete_graph(5)
greedy = CZScheduler(G5); quiet(greedy.schedule_cz_rounds)
mg = misra_gries_edge_coloring(G5)
pos = nx.circular_layout(G5)
palette = ["#285A96", "#C1853C", "#4E8C6A", "#9C5A8C", "#7FA6CE",
           "#B4534B", "#8C8C5A"]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, col, title in [
        (axes[0], greedy.edge_coloring, f"greedy: {greedy.chromatic_index} rounds"),
        (axes[1], mg, f"Misra-Gries: {max(mg.values())+1} rounds (optimum)")]:
    for (u, v), c in col.items():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=palette[c], lw=4.0, zorder=1)
    for v, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.12, facecolor="white",
                                edgecolor="#1a1a1a", lw=2, zorder=2))
        ax.text(x, y, str(v), ha="center", va="center", fontsize=12, zorder=3)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("K5 edge colouring — each colour is one 2-Tock round",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "mg_k5_colourings.png", bbox_inches="tight")
plt.close(fig)

# 3. timing: greedy vs Misra-Gries
cases = [
    ("K5", nx.complete_graph(5)),
    ("K20", nx.complete_graph(20)),
    ("K40", nx.complete_graph(40)),
    ("grid 10x10", nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10))),
    ("grid 20x20", nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20))),
    ("random n=100 p=0.1", nx.erdos_renyi_graph(100, 0.1, seed=1)),
    ("random n=200 p=0.05", nx.erdos_renyi_graph(200, 0.05, seed=1)),
]
print("\ngraph                 edges   greedy(ms)   misra-gries(ms)   ratio")
for name, G in cases:
    m = G.number_of_edges()
    sch = CZScheduler(G)
    t0 = time.perf_counter()
    for _ in range(5):
        sch._greedy_edge_coloring()
    tg = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(5):
        misra_gries_edge_coloring(G)
    tm = (time.perf_counter() - t0) / 5 * 1000
    print(f"{name:22s}{m:5d}   {tg:9.2f}   {tm:14.2f}   {tm/tg:6.1f}x")
