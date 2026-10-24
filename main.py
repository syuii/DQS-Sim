"""
main.py - Graph State Generation Simulator (GoSC / Lattice Surgery)

Usage
-----
    python main.py                                       # defaults: star(10), d=7
    python main.py --graph cycle --n 20 --case 1        # Case 1 only, cycle graph
    python main.py --graph star  --n 10 --save-all            # saves to ./star_n10/
    python main.py --graph star  --n 10 --save-all results/   # saves to results/star_n10/
    python main.py --graph star  --n 10 --layout --gantt --heatmap --dashboard --occupancy  # display
    python main.py --graph random --n 100 --no-verify   # skip verification (faster)

    # k-port switch (Cases 4 & 5)
    python main.py --graph cycle --n 8 --case all --switch-k 4 --no-verify
    python main.py --graph star  --n 6 --case limited --switch-k 6

    --save-all DIR creates DIR/{graph}_n{n}/ and writes all figures + report.json inside.
    Omit DIR to use the current directory. Individual --save-* flags override specific paths.

Graph types: path, cycle, star, complete, grid, random, petersen, hypercube
    grid -> n×n  |  hypercube -> 2^n nodes  |  random -> use --p for edge probability
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import networkx as nx

from dqs_sim.cost import compute_report
from dqs_sim.schedulers import (
    CZScheduler, CZSchedulerFixedAncilla, CZSchedulerLimited,
    MPPConflictScheduler, MPPSchedulerLimited,
)
from dqs_sim.topology import KPortSwitch
from dqs_sim.visualizer import (show_comparison_dashboard, show_gantt, show_heatmaps,
                                show_2d_layout, show_qubit_occupancy,
                                show_qubit_occupancy_case2, show_qubit_occupancy_case3,
                                show_qubit_occupancy_case4, show_qubit_occupancy_case5,
                                show_port_utilization, show_heatmaps_limited)


# Graph builders

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


def load_graph_file(path: str) -> nx.Graph:
    """Read an arbitrary target graph from a file.

    Accepts an edge list (one "u v" pair per line, '#' comments allowed) or
    a NetworkX node-link JSON file, so any graph can be scheduled without
    being one of the built-in families.
    """
    if not os.path.exists(path):
        print(f"Graph file not found: {path}")
        sys.exit(1)

    if path.lower().endswith(".json"):
        with open(path) as f:
            G = nx.node_link_graph(json.load(f))
    else:
        G = nx.read_edgelist(path, nodetype=int, comments="#")

    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    if len(G) == 0:
        print(f"Graph file '{path}' contains no edges.")
        sys.exit(1)
    return G


def build_graph(graph_type: str, n: int, p: float,
                graph_file: str | None = None) -> nx.Graph:
    if graph_file:
        G = load_graph_file(graph_file)
    else:
        if graph_type not in GRAPH_BUILDERS:
            print(f"Unknown graph '{graph_type}'. Options: {GRAPH_HELP}")
            sys.exit(1)
        G = GRAPH_BUILDERS[graph_type](n, p)
    # Remove self-loops (safety)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# CLI

CASE_CHOICES = ["1", "2", "3", "4", "5", "both", "all", "limited"]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Graph State Generation Simulator - GoSC / Lattice Surgery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Graph types: " + GRAPH_HELP + "\n\n"
            "Notes:\n"
            "  'grid' creates an n×n grid (total n² nodes).\n"
            "  'hypercube' creates a 2^n-node hypercube (n = dimension).\n"
            "  'petersen' ignores --n.\n"
        ),
    )

    # Graph
    g = p.add_argument_group("Graph")
    g.add_argument("--graph", choices=list(GRAPH_BUILDERS), default="star",
                   metavar="TYPE", help=f"Graph topology ({GRAPH_HELP})")
    g.add_argument("--n", type=int, default=10,
                   help="Node count (grid side / hypercube dimension where applicable)")
    g.add_argument("--p", type=float, default=0.3,
                   help="Edge probability (random graph only)")
    g.add_argument("--graph-file", metavar="FILE", default=None,
                   help="Schedule an arbitrary graph read from FILE "
                        "(edge list, one 'u v' per line, or node-link JSON). "
                        "Overrides --graph.")

    # Simulation
    s = p.add_argument_group("Simulation")
    s.add_argument("--case", choices=CASE_CHOICES,
                   default="both",
                   help="Scheduling case(s): 1=MPP, 2=CZ unlimited, "
                        "3=CZ k-ancilla, 4=MPP k-port, 5=CZ k-port, "
                        "both=1+2, all=1+2+3+4+5, limited=4+5")
    s.add_argument("--ancillas", type=int, default=None, metavar="K",
                   help="Ancilla pool size k for Case 3 (--case 3 or all). "
                        "If omitted with --case all, auto-selects k = ⌊peak_colour_class/2⌋. "
                        "If omitted with --case 3, defaults to k=1.")
    s.add_argument("--colouring", choices=["greedy", "vizing"], default="greedy",
                   help="CZ edge-colouring mode: greedy (fast, up to 2*Delta-1 "
                        "colours) or vizing (Misra-Gries, guaranteed Delta+1)")
    s.add_argument("--mis", choices=["auto", "exact", "greedy"], default="auto",
                   help="MIS mode for Cases 1/4: auto (exact for n<=20, greedy "
                        "beyond), exact (branch-and-bound at any size), or greedy")
    s.add_argument("--no-verify", action="store_true",
                   help="Skip chromatic / Vizing verification (faster for large graphs)")

    # k-port switch (Cases 4 & 5)
    t = p.add_argument_group("k-port switch (Cases 4 & 5)")
    t.add_argument("--switch-k", type=int, default=None, metavar="K",
                   help="Ports per switch (simultaneously connected ELUs). "
                        "Required for --case 4, 5, limited, or all. "
                        "Each CZ step uses 2 ports; each MPP uses |support| ports.")
    t.add_argument("--switches", type=int, default=1, metavar="S",
                   help="Number of parallel switches (default: 1). "
                        "Each switch independently connects k ELUs per Tock. "
                        "ELUs on the same switch can interact; different switches cannot.")

    # Cost estimation
    c = p.add_argument_group("Cost estimation")
    c.add_argument("--code-distance", type=int, default=7, metavar="D",
                   help="Surface code distance d  ->  2d²-1 physical qubits per ELU")

    # Output
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
    o.add_argument("--save-occupancy2", metavar="FILE.png",
                   help="Save Case 2 qubit occupancy figure to file")
    o.add_argument("--save-occupancy3", metavar="FILE.png",
                   help="Save Case 3 qubit occupancy figure to file")
    o.add_argument("--save-dashboard", metavar="FILE.png",
                   help="Save dashboard to file")
    o.add_argument("--save-occupancy4", metavar="FILE.png",
                   help="Save Case 4 per-tock occupancy figure to file")
    o.add_argument("--save-occupancy5", metavar="FILE.png",
                   help="Save Case 5 per-tock occupancy figure to file")
    o.add_argument("--save-port-util", metavar="FILE.png",
                   help="Save port utilization chart (Cases 4/5) to file")
    o.add_argument("--save-heatmap-limited", metavar="FILE.png",
                   help="Save k-port heatmap (Cases 4/5) to file")
    o.add_argument("--export", metavar="FILE.json",
                   help="Export resource report as JSON")

    return p


# Helpers

def _needs_limited(case: str) -> bool:
    return case in ("4", "5", "limited", "all")


def _build_switch(args, G: nx.Graph) -> KPortSwitch:
    """Build a KPortSwitch, auto-selecting k if not provided."""
    n = len(G.nodes)
    k = args.switch_k
    if k is None:
        # Default: must fit largest stabilizer (|support| data + 1 ancilla)
        max_support = 1 + max((d for _, d in G.degree()), default=0)
        min_k = max_support + 1  # data + 1 ancilla ELU
        k = max(min_k, n)
        print(f"  [Switch] --switch-k not set. Auto-selecting k={k} "
              f"(min ports={max_support}+1anc={min_k}).")
    n_sw = args.switches
    # n_elus = data ELUs + generous ancilla pool
    n_elus = n * 2  # data + up to |V| ancillas
    return KPortSwitch(k=k, n_elus=n_elus, n_switches=n_sw)


# Entry point

def main() -> None:
    # Output uses symbols like: and χ; avoid UnicodeEncodeError on non-UTF-8
    # consoles (e.g. Windows with a Japanese locale, cp932).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
        args.save_occupancy2 = args.save_occupancy2 or os.path.join(folder, "occupancy2.png")
        args.save_occupancy3 = args.save_occupancy3 or os.path.join(folder, "occupancy3.png")
        args.save_dashboard  = args.save_dashboard  or os.path.join(folder, "dashboard.png")
        args.save_occupancy4      = args.save_occupancy4      or os.path.join(folder, "occupancy4.png")
        args.save_occupancy5      = args.save_occupancy5      or os.path.join(folder, "occupancy5.png")
        args.save_port_util       = args.save_port_util       or os.path.join(folder, "port_util.png")
        args.save_heatmap_limited = args.save_heatmap_limited or os.path.join(folder, "heatmap_limited.png")
        args.export         = args.export         or os.path.join(folder, "report.json")
        print(f"  Output folder: {folder}/")

    G = build_graph(args.graph, args.n, args.p, args.graph_file)
    label = (f"file:{os.path.basename(args.graph_file)}" if args.graph_file
             else f"{args.graph}(n={args.n})")
    print(f"\n  Graph: {label}  "
          f"|V|={len(G.nodes)}, |E|={len(G.edges)}, "
          f"Δ={max(d for _, d in G.degree()) if G.nodes else 0}")

    scheduler: MPPConflictScheduler      | None = None
    cz:        CZScheduler               | None = None
    cz_k:      CZSchedulerFixedAncilla   | None = None
    mpp_lim:   MPPSchedulerLimited       | None = None
    cz_lim:    CZSchedulerLimited        | None = None

    # Case 1
    if args.case in ("1", "both", "all"):
        print("\n" + "#" * 52)
        print("#  CASE 1 - Stabilizer MPP Scheduler")
        print("#" * 52)
        scheduler = MPPConflictScheduler(G, mis_mode=args.mis)
        scheduler.run(visualize=args.visualize, verify_chromatic=verify)

    # Case 2
    if args.case in ("2", "both", "all"):
        print("\n" + "#" * 52)
        print("#  CASE 2 - CZ Gate Scheduler  (unlimited ancillas)")
        print("#" * 52)
        cz = CZScheduler(G, coloring_mode=args.colouring)
        cz.run(visualize=args.visualize, verify_vizing=verify)

    # Case 3
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
        print(f"#  CASE 3 - CZ Gates, k={k} Ancilla ELU(s)")
        print("#" * 52)
        cz_k = CZSchedulerFixedAncilla(G, k=k, coloring_mode=args.colouring)
        cz_k.run(visualize=args.visualize, verify_vizing=verify)

    # Cases 4 & 5: build k-port switch
    switch = None
    if _needs_limited(args.case):
        switch = _build_switch(args, G)
        print(f"\n  Switch: {switch}")

    # Case 4
    if args.case in ("4", "limited", "all"):
        mpp_lim = MPPSchedulerLimited(G, switch, mis_mode=args.mis)
        # Only measured generators need ports; MIS vertices start in |+> for free.
        mpp_lim.phase_1_reduction()
        max_support = 1 + max((G.degree(v) for v in mpp_lim.to_measure), default=0)
        min_k = max_support + 1  # data + 1 ancilla
        if switch is not None and min_k > switch.k:
            mpp_lim = None
            print(f"\n  WARNING: Case 4 skipped: largest generator needs "
                  f"{max_support} data + 1 ancilla = {min_k} ports "
                  f"> k={switch.k}. Use --switch-k {min_k} or higher.")
        else:
            print("\n" + "#" * 60)
            print("#  CASE 4 - MPP Stabilizer + k-Port Switch")
            print("#" * 60)
            mpp_lim.run(visualize=args.visualize)

    # Case 5
    if args.case in ("5", "limited", "all"):
        print("\n" + "#" * 60)
        print("#  CASE 5 - CZ Gates + k-Port Switch")
        print("#" * 60)
        cz_lim = CZSchedulerLimited(G, switch, coloring_mode=args.colouring)
        cz_lim.run(visualize=args.visualize, verify_vizing=verify)

    # Resource report
    if scheduler is not None and cz is not None:
        report = compute_report(
            scheduler,
            cz,
            code_distance=args.code_distance,
            graph_type=args.graph,
            cz_k=cz_k,
            mpp_limited=mpp_lim,
            cz_limited=cz_lim,
        )

        print("\n" + "=" * 52)
        print("  RESOURCE REPORT")
        print("=" * 52)
        report.print_table()

        if args.export:
            report.export_json(args.export)

        show = args.show  # True -> display interactively even when saving

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

        if (args.show or args.occupancy or args.save_occupancy2) and cz is not None:
            show_qubit_occupancy_case2(cz,
                                       save_path=args.save_occupancy2, display=show)

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

    # k-port switch visualizations (Cases 4 & 5)
    has_limited = mpp_lim is not None or cz_lim is not None
    if has_limited:
        show = getattr(args, 'show', False)

        if (show or args.occupancy or args.save_occupancy4) and mpp_lim is not None:
            show_qubit_occupancy_case4(mpp_lim,
                                       save_path=args.save_occupancy4, display=show)

        if (show or args.occupancy or args.save_occupancy5) and cz_lim is not None:
            show_qubit_occupancy_case5(cz_lim,
                                       save_path=args.save_occupancy5, display=show)

        if show or args.save_port_util:
            show_port_utilization(mpp_lim=mpp_lim, cz_lim=cz_lim,
                                  save_path=args.save_port_util, display=show)

        if show or args.save_heatmap_limited:
            show_heatmaps_limited(mpp_lim=mpp_lim, cz_lim=cz_lim,
                                  save_path=args.save_heatmap_limited, display=show)


if __name__ == "__main__":
    main()
