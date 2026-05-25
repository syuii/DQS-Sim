"""
main.py — Graph State Generation Simulator (GoSC / Lattice Surgery)

Usage
-----
    python main.py                                       # defaults: star(10), d=7
    python main.py --graph cycle --n 20 --case 1        # Case 1 only, cycle graph
    python main.py --graph star  --n 10 --save-all            # saves to ./star_n10/
    python main.py --graph star  --n 10 --save-all results/   # saves to results/star_n10/
    python main.py --graph star  --n 10 --layout --gantt --heatmap --dashboard --occupancy  # display
    python main.py --graph random --n 100 --no-verify   # skip verification (faster)

    --save-all DIR creates DIR/{graph}_n{n}/ and writes all figures + report.json inside.
    Omit DIR to use the current directory. Individual --save-* flags override specific paths.

Graph types: path, cycle, star, complete, grid, random, petersen, hypercube
    grid → n×n  |  hypercube → 2^n nodes  |  random → use --p for edge probability
"""

import argparse
import os
import sys

import networkx as nx

from cost_estimator import compute_report
from substrate_scheduler import CZScheduler, CZSchedulerFixedAncilla, SubstrateScheduler
from visualizer import (show_comparison_dashboard, show_gantt, show_heatmaps,
                        show_2d_layout, show_qubit_occupancy, show_qubit_occupancy_case3)


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
    s.add_argument("--case", choices=["1", "2", "3", "both", "all"],
                   default="both",
                   help="Scheduling case(s): 1=MPP, 2=CZ unlimited, "
                        "3=CZ k-ancilla, both=1+2, all=1+2+3")
    s.add_argument("--ancillas", type=int, default=None, metavar="K",
                   help="Ancilla pool size k for Case 3 (--case 3 or all). "
                        "If omitted with --case all, auto-selects k = ⌊peak_colour_class/2⌋. "
                        "If omitted with --case 3, defaults to k=1.")
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
    o.add_argument("--show", action="store_true",
                   help="Display all summary visualizations interactively "
                        "(Gantt, heatmap, 2D layout, occupancy, dashboard). "
                        "Compatible with --save-* flags: use both to display AND save.")
    o.add_argument("--gantt", action="store_true",
                   help="Gantt chart: qubit activity over time (scales to ~100 nodes)")
    o.add_argument("--heatmap", action="store_true",
                   help="Schedule heatmap: compact matrix view (works at any scale)")
    o.add_argument("--layout", action="store_true",
                   help="2D patch layout: ELUs as physical surface-code patches")
    o.add_argument("--occupancy", action="store_true",
                   help="Per-tock qubit occupancy tile strip (all running cases)")
    o.add_argument("--dashboard", action="store_true",
                   help="Bar-chart resource comparison (both cases)")
    o.add_argument("--save-all", metavar="DIR", nargs="?", const=".",
                   help="Save all figures + JSON to {graph}_n{n}/ inside DIR (default: current dir)")
    o.add_argument("--save-gantt",     metavar="FILE.png",
                   help="Save Gantt chart to file")
    o.add_argument("--save-heatmap",   metavar="FILE.png",
                   help="Save heatmap to file")
    o.add_argument("--save-layout",    metavar="FILE.png",
                   help="Save 2D patch layout to file")
    o.add_argument("--save-occupancy", metavar="FILE.png",
                   help="Save Case 1 qubit occupancy figure to file")
    o.add_argument("--save-occupancy3", metavar="FILE.png",
                   help="Save Case 3 qubit occupancy figure to file")
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

    # --save-all: auto-create {graph}_n{n}/ and fill in any unset save paths
    if args.save_all:
        folder = os.path.join(args.save_all, f"{args.graph}_n{args.n}")
        os.makedirs(folder, exist_ok=True)
        args.save_gantt     = args.save_gantt     or os.path.join(folder, "gantt.png")
        args.save_heatmap   = args.save_heatmap   or os.path.join(folder, "heatmap.png")
        args.save_layout    = args.save_layout    or os.path.join(folder, "layout.png")
        args.save_occupancy  = args.save_occupancy  or os.path.join(folder, "occupancy.png")
        args.save_occupancy3 = args.save_occupancy3 or os.path.join(folder, "occupancy3.png")
        args.save_dashboard  = args.save_dashboard  or os.path.join(folder, "dashboard.png")
        args.export         = args.export         or os.path.join(folder, "report.json")
        print(f"  Output folder: {folder}/")

    G = build_graph(args.graph, args.n, args.p)
    print(f"\n  Graph: {args.graph}(n={args.n})  "
          f"|V|={len(G.nodes)}, |E|={len(G.edges)}, "
          f"Δ={max(d for _, d in G.degree()) if G.nodes else 0}")

    scheduler: SubstrateScheduler        | None = None
    cz:        CZScheduler               | None = None
    cz_k:      CZSchedulerFixedAncilla   | None = None

    # ── Case 1 ───────────────────────────────────────────────────────────────
    if args.case in ("1", "both", "all"):
        print("\n" + "#" * 52)
        print("#  CASE 1 — Stabilizer MPP Scheduler")
        print("#" * 52)
        scheduler = SubstrateScheduler(G)
        scheduler.run(visualize=args.visualize, verify_chromatic=verify)

    # ── Case 2 ───────────────────────────────────────────────────────────────
    if args.case in ("2", "both", "all"):
        print("\n" + "#" * 52)
        print("#  CASE 2 — CZ Gate Scheduler  (unlimited ancillas)")
        print("#" * 52)
        cz = CZScheduler(G)
        cz.run(visualize=args.visualize, verify_vizing=verify)

    # ── Case 3 ───────────────────────────────────────────────────────────
    if args.case in ("3", "all"):
        k = args.ancillas
        if k is None:
            if cz is not None and cz.cz_rounds:
                peak = max(len(r) for r in cz.cz_rounds)
                k = max(1, peak // 2)
                print(f"\n  [Case 3] --ancillas not set. "
                      f"Auto-selecting k={k}  (⌊{peak}/2⌋ from peak colour class).")
            else:
                k = 1
                print(f"\n  [Case 3] --ancillas not set. Defaulting to k=1.")
        print("\n" + "#" * 52)
        print(f"#  CASE 3 — CZ Gates, k={k} Ancilla ELU(s)")
        print("#" * 52)
        cz_k = CZSchedulerFixedAncilla(G, k=k)
        cz_k.run(visualize=args.visualize, verify_vizing=verify)

    # ── Resource report ──────────────────────────────────────────────────────
    if scheduler is not None and cz is not None:
        report = compute_report(
            scheduler,
            cz,
            code_distance=args.code_distance,
            graph_type=args.graph,
            cz_k=cz_k,
        )

        print("\n" + "=" * 52)
        print("  RESOURCE REPORT")
        print("=" * 52)
        report.print_table()

        if args.export:
            report.export_json(args.export)

        show = args.show   # True → display interactively even when saving

        if args.show or args.gantt or args.save_gantt:
            show_gantt(scheduler, cz, cz_k=cz_k,
                       save_path=args.save_gantt, display=show)

        if args.show or args.heatmap or args.save_heatmap:
            show_heatmaps(scheduler, cz, cz_k=cz_k,
                          save_path=args.save_heatmap, display=show)

        if args.show or args.layout or args.save_layout:
            show_2d_layout(scheduler, cz, cz_k=cz_k,
                           save_path=args.save_layout, display=show)

        if args.show or args.occupancy or args.save_occupancy:
            show_qubit_occupancy(scheduler,
                                 save_path=args.save_occupancy, display=show)

        if (args.show or args.occupancy or args.save_occupancy3) and cz_k is not None:
            show_qubit_occupancy_case3(cz_k,
                                       save_path=args.save_occupancy3, display=show)

        if args.show or args.dashboard or args.save_dashboard:
            show_comparison_dashboard(report,
                                      save_path=args.save_dashboard, display=show)

    elif args.show or args.gantt or args.heatmap or args.layout or args.occupancy or args.dashboard:
        print(
            "\n  Note: visualisation flags require Cases 1 and 2 "
            "(use --case both or --case all)."
        )


if __name__ == "__main__":
    main()
