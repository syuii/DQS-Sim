"""
main.py
=======
CLI entry point for the graph state generation simulator.

Usage examples
--------------
    python main.py                                  # star(10), d=7, both cases
    python main.py --graph path --n 20              # path graph, 20 nodes
    python main.py --graph star --n 50 --gantt      # show Gantt chart
    python main.py --graph grid --n 5 --dashboard   # 5×5 grid, comparison figure
    python main.py --graph random --n 30 --p 0.4    # random graph, 30 nodes
    python main.py --graph star --n 10 --visualize  # per-tock figures
    python main.py --graph star --n 10 --export results.json
    python main.py --case 1 --graph cycle --n 15    # Case 1 only
"""

import argparse
import sys

import networkx as nx

from cost_estimator import compute_report
from substrate_scheduler import CZScheduler, SubstrateScheduler
from visualizer import show_comparison_dashboard, show_gantt, show_heatmaps, show_2d_layout

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

GRAPH_BUILDERS = {
    "path":     lambda n, _: nx.path_graph(n),
    "cycle":    lambda n, _: nx.cycle_graph(n),
    "star":     lambda n, _: nx.star_graph(n),
    "complete": lambda n, _: nx.complete_graph(n),
    "grid":     lambda n, _: nx.convert_node_labels_to_integers(
                                 nx.grid_2d_graph(n, n)),
    "random":   lambda n, p: nx.erdos_renyi_graph(n, p, seed=42),
    "petersen": lambda *_: nx.petersen_graph(),
    "hypercube":lambda n, _: nx.hypercube_graph(n),  # n = dimension
}

GRAPH_HELP = ", ".join(GRAPH_BUILDERS)


def build_graph(graph_type: str, n: int, p: float) -> nx.Graph:
    if graph_type not in GRAPH_BUILDERS:
        print(f"Unknown graph '{graph_type}'. Options: {GRAPH_HELP}")
        sys.exit(1)
    G = GRAPH_BUILDERS[graph_type](n, p)
    # Remove self-loops (safety)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Graph State Generation Simulator — GoSC / Lattice Surgery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Graph types: " + GRAPH_HELP + "\n\n"
            "Notes:\n"
            "  'grid' creates an n×n grid (total n² nodes).\n"
            "  'hypercube' creates a 2^n-node hypercube (n = dimension).\n"
            "  'petersen' ignores --n.\n"
        ),
    )

    # ── Graph ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Graph")
    g.add_argument("--graph", choices=list(GRAPH_BUILDERS), default="star",
                   metavar="TYPE", help=f"Graph topology ({GRAPH_HELP})")
    g.add_argument("--n", type=int, default=10,
                   help="Node count (grid side / hypercube dimension where applicable)")
    g.add_argument("--p", type=float, default=0.3,
                   help="Edge probability (random graph only)")

    # ── Simulation ────────────────────────────────────────────────────────────
    s = p.add_argument_group("Simulation")
    s.add_argument("--case", choices=["1", "2", "both"], default="both",
                   help="Which scheduling case to run")
    s.add_argument("--no-verify", action="store_true",
                   help="Skip chromatic / Vizing verification (faster for large graphs)")

    # ── Cost estimation ───────────────────────────────────────────────────────
    c = p.add_argument_group("Cost estimation")
    c.add_argument("--code-distance", type=int, default=7, metavar="D",
                   help="Surface code distance d  →  2d²-1 physical qubits per ELU")

    # ── Output ────────────────────────────────────────────────────────────────
    o = p.add_argument_group("Output")
    o.add_argument("--visualize", action="store_true",
                   help="Show per-tock / per-round matplotlib figures (best for small graphs)")
    o.add_argument("--gantt", action="store_true",
                   help="Gantt chart: qubit activity over time (scales to ~100 nodes)")
    o.add_argument("--heatmap", action="store_true",
                   help="Schedule heatmap: compact matrix view (works at any scale)")
    o.add_argument("--layout", action="store_true",
                   help="2D patch layout: ELUs as physical surface-code patches")
    o.add_argument("--dashboard", action="store_true",
                   help="Bar-chart resource comparison (both cases)")
    o.add_argument("--save-gantt",     metavar="FILE.png",
                   help="Save Gantt chart to file")
    o.add_argument("--save-heatmap",   metavar="FILE.png",
                   help="Save heatmap to file")
    o.add_argument("--save-layout",    metavar="FILE.png",
                   help="Save 2D patch layout to file")
    o.add_argument("--save-dashboard", metavar="FILE.png",
                   help="Save dashboard to file")
    o.add_argument("--export", metavar="FILE.json",
                   help="Export resource report as JSON")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = make_parser().parse_args()
    verify = not args.no_verify

    G = build_graph(args.graph, args.n, args.p)
    print(f"\n  Graph: {args.graph}(n={args.n})  "
          f"|V|={len(G.nodes)}, |E|={len(G.edges)}, "
          f"Δ={max(d for _, d in G.degree()) if G.nodes else 0}")

    scheduler: SubstrateScheduler | None = None
    cz:        CZScheduler        | None = None

    # ── Case 1 ───────────────────────────────────────────────────────────────
    if args.case in ("1", "both"):
        print("\n" + "#" * 52)
        print("#  CASE 1 — Stabilizer MPP Scheduler")
        print("#" * 52)
        scheduler = SubstrateScheduler(G)
        scheduler.run(visualize=args.visualize, verify_chromatic=verify)

    # ── Case 2 ───────────────────────────────────────────────────────────────
    if args.case in ("2", "both"):
        print("\n" + "#" * 52)
        print("#  CASE 2 — CZ Gate Scheduler")
        print("#" * 52)
        cz = CZScheduler(G)
        cz.run(visualize=args.visualize, verify_vizing=verify)

    # ── Resource report (requires both cases) ─────────────────────────────────
    if scheduler is not None and cz is not None:
        report = compute_report(
            scheduler,
            cz,
            code_distance=args.code_distance,
            graph_type=args.graph,
        )

        print("\n" + "=" * 52)
        print("  RESOURCE REPORT")
        print("=" * 52)
        report.print_table()

        if args.export:
            report.export_json(args.export)

        if args.gantt or args.save_gantt:
            show_gantt(scheduler, cz, save_path=args.save_gantt)

        if args.heatmap or args.save_heatmap:
            show_heatmaps(scheduler, cz, save_path=args.save_heatmap)

        if args.layout or args.save_layout:
            show_2d_layout(scheduler, cz, save_path=args.save_layout)

        if args.dashboard or args.save_dashboard:
            show_comparison_dashboard(report, save_path=args.save_dashboard)

    elif args.gantt or args.heatmap or args.layout or args.dashboard:
        print(
            "\n  Note: visualisation flags require both cases "
            "(use --case both, which is the default)."
        )


if __name__ == "__main__":
    main()
