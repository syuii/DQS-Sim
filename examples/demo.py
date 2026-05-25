"""
examples/demo.py
================
Preset demo runs for DQS-Sim.

Runs Cases 1, 2, and 3 across several graph topologies and saves all
figures + JSON reports to Results/.

Usage
-----
    python examples/demo.py                   # all presets
    python examples/demo.py --graph star      # one topology only
    python examples/demo.py --show            # display figures interactively too
"""

import argparse
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from dqs_sim import (
    CZScheduler,
    CZSchedulerFixedAncilla,
    SubstrateScheduler,
    compute_report,
)
from dqs_sim.visualizer import (
    show_comparison_dashboard,
    show_gantt,
    show_heatmaps,
    show_2d_layout,
    show_qubit_occupancy,
    show_qubit_occupancy_case3,
)

# ---------------------------------------------------------------------------
# Presets: (graph_label, graph_object, ancilla_k_for_case3)
# ---------------------------------------------------------------------------

PRESETS = [
    ("path",     nx.path_graph(8),                           2),
    ("cycle",    nx.cycle_graph(8),                          2),
    ("star",     nx.star_graph(8),                           2),
    ("grid",     nx.convert_node_labels_to_integers(
                     nx.grid_2d_graph(3, 3)),                2),
    ("complete", nx.complete_graph(5),                       2),
]


def run_preset(graph_label: str, G: nx.Graph, k: int,
               out_root: str, display: bool) -> None:
    G.remove_edges_from(nx.selfloop_edges(G))
    n = len(G.nodes)
    folder = os.path.join(out_root, f"{graph_label}_n{n}")
    os.makedirs(folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {graph_label.upper()}  |V|={n}  |E|={len(G.edges)}  k={k}")
    print(f"{'='*60}")

    # ── Case 1 ──────────────────────────────────────────────────────────────
    scheduler = SubstrateScheduler(G)
    scheduler.run(visualize=False, verify_chromatic=False)

    # ── Case 2 ──────────────────────────────────────────────────────────────
    cz = CZScheduler(G)
    cz.run(visualize=False, verify_vizing=False)

    # ── Case 3 ──────────────────────────────────────────────────────────────
    cz_k = CZSchedulerFixedAncilla(G, k=k)
    cz_k.run(visualize=False, verify_vizing=False)

    # ── Report ───────────────────────────────────────────────────────────────
    report = compute_report(scheduler, cz, code_distance=7,
                            graph_type=graph_label, cz_k=cz_k)
    report.print_table()
    report.export_json(os.path.join(folder, "report.json"))

    # ── Figures ───────────────────────────────────────────────────────────────
    show_gantt(scheduler, cz, cz_k=cz_k,
               save_path=os.path.join(folder, "gantt.png"), display=display)
    show_heatmaps(scheduler, cz, cz_k=cz_k,
                  save_path=os.path.join(folder, "heatmap.png"), display=display)
    show_2d_layout(scheduler, cz, cz_k=cz_k,
                   save_path=os.path.join(folder, "layout.png"), display=display)
    show_qubit_occupancy(scheduler,
                         save_path=os.path.join(folder, "occupancy.png"),
                         display=display)
    show_qubit_occupancy_case3(cz_k,
                               save_path=os.path.join(folder, "occupancy3.png"),
                               display=display)
    show_comparison_dashboard(report,
                              save_path=os.path.join(folder, "dashboard.png"),
                              display=display)

    print(f"  Saved → {folder}/")


def main() -> None:
    p = argparse.ArgumentParser(description="DQS-Sim preset demo runs")
    p.add_argument("--graph", default=None,
                   choices=[name for name, _, _ in PRESETS],
                   help="Run only this graph topology (default: all)")
    p.add_argument("--out", default="Results",
                   help="Root output directory (default: Results/)")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively in addition to saving")
    args = p.parse_args()

    presets = [(n, G, k) for n, G, k in PRESETS
               if args.graph is None or n == args.graph]

    for graph_label, G, k in presets:
        run_preset(graph_label, G, k, out_root=args.out, display=args.show)

    print(f"\nAll done. Results in {args.out}/")


if __name__ == "__main__":
    main()
