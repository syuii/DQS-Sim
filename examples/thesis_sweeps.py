"""
thesis_sweeps.py
================
Sweep experiments for the thesis Results chapter.

Three experiments, all run through the same scheduler classes as the CLI:

  E1  Family survey  — all five cases on six graph families of comparable size.
  E2  Scaling sweep  — T1 (MPP) and T2 (CZ) versus graph size n per family.
  E3  Hardware sweep — T3 versus ancilla-pool size k, and T5 versus switch
                       port count k, with the Case-4 feasibility threshold.

Outputs
-------
  sweeps.json                     raw numbers for every experiment
  fig_survey.pdf                  grouped bars, Tocks by family and case
  fig_scaling.pdf                 2x2 small multiples, Tocks vs n
  fig_tradeoffs.pdf               T3(k) and T5(k) trade-off curves

Usage
-----
  python examples/thesis_sweeps.py OUTPUT_DIR
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dqs_sim.schedulers import (CZScheduler, CZSchedulerFixedAncilla,
                                CZSchedulerLimited, MPPSchedulerLimited,
                                SubstrateScheduler)
from dqs_sim.topology import KPortSwitch

# ── Thesis figure style ──────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "CMU Serif", "Times New Roman"],
    "mathtext.fontset":  "cm",
    "axes.edgecolor":    "#5A6473",
    "axes.labelcolor":   "#1A1A1A",
    "axes.titlecolor":   "#1A1A1A",
    "text.color":        "#1A1A1A",
    "xtick.color":       "#5A6473",
    "ytick.color":       "#5A6473",
    "axes.linewidth":    0.8,
    "savefig.dpi":       300,
})

_W = 6.3   # thesis \textwidth in inches

# Fixed case colours (validated categorical palette; identical across figures)
CASE_COL = {
    1: "#285A96",   # navy  — MPP unlimited
    2: "#C1853C",   # ochre — CZ unlimited
    3: "#4E8C6A",   # green — CZ, k ancillas
    4: "#9C5A8C",   # plum  — MPP, switch
    5: "#7FA6CE",   # light navy — CZ, switch
}
RED_HL = "#B4534B"

CASE_LBL = {
    1: "Case 1 (MPP)",
    2: "Case 2 (CZ)",
    3: "Case 3 (CZ, $k_a{=}2$)",
    4: "Case 4 (MPP, switch)",
    5: "Case 5 (CZ, switch)",
}


# ── Quiet scheduler wrappers ─────────────────────────────────────────────────

def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def t1_mpp(G: nx.Graph) -> int:
    # Layout-free distributed MPP (Case 1, DQC): MIS + conflict-graph
    # colouring; no substrate mapping, no interval blocking.
    from dqs_sim.schedulers import mpp_conflict_schedule
    return len(_quiet(mpp_conflict_schedule, G))


def t2_cz(G: nx.Graph) -> tuple[int, int]:
    c = CZScheduler(G)
    _quiet(c.schedule_cz_rounds)
    return 2 * c.chromatic_index, c.chromatic_index


def t3_cz_k(G: nx.Graph, k: int) -> int:
    c = CZSchedulerFixedAncilla(G, k=k)
    _quiet(c.schedule_batches)
    return c.n_tocks()


def t4_mpp_sw(G: nx.Graph, k: int, s: int = 1) -> int | None:
    sw = KPortSwitch(k=k, n_elus=2 * len(G), n_switches=s)
    m = MPPSchedulerLimited(G, sw)
    try:
        _quiet(m.schedule)
        return m.n_tocks()
    except ValueError:
        return None          # generator support exceeds switch width


def t5_cz_sw(G: nx.Graph, k: int, s: int = 1) -> int | None:
    sw = KPortSwitch(k=k, n_elus=2 * len(G), n_switches=s)
    c = CZSchedulerLimited(G, sw)
    try:
        _quiet(c.schedule_batches)
        return c.n_tocks()
    except ValueError:
        return None          # k < 2: cannot host a single CZ


# ── Graph builders ───────────────────────────────────────────────────────────

def build(family: str, n: int) -> nx.Graph:
    if family == "path":
        return nx.path_graph(n)
    if family == "cycle":
        return nx.cycle_graph(n)
    if family == "star":
        return nx.star_graph(n - 1)          # n nodes total
    if family == "complete":
        return nx.complete_graph(n)
    if family == "grid":
        side = int(round(n ** 0.5))
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    if family == "random":
        G = nx.erdos_renyi_graph(n, 0.35, seed=42)
        if not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, 0.35, seed=1)
        return G
    raise ValueError(family)


# ── E1: family survey ────────────────────────────────────────────────────────

def run_survey(k_anc: int = 2, k_sw: int = 4) -> list[dict]:
    rows = []
    for family, n in [("path", 8), ("cycle", 8), ("star", 8),
                      ("complete", 8), ("grid", 9), ("random", 8)]:
        G = build(family, n)
        T2, chi = t2_cz(G)
        rows.append({
            "family":  family,
            "n":       len(G),
            "m":       len(G.edges),
            "delta":   max(d for _, d in G.degree()),
            "chi":     chi,
            "T1":      t1_mpp(G),
            "T2":      T2,
            "T3":      t3_cz_k(G, k_anc),
            "T4":      t4_mpp_sw(G, k_sw),
            "T5":      t5_cz_sw(G, k_sw),
        })
    return rows


# ── E2: scaling sweep ────────────────────────────────────────────────────────

def run_scaling() -> dict:
    ns = list(range(4, 21, 2))
    out: dict = {"n": ns}
    for family in ["path", "cycle", "star", "complete"]:
        t1s, t2s = [], []
        for n in ns:
            G = build(family, n)
            t1s.append(t1_mpp(G))
            t2s.append(t2_cz(G)[0])
        out[family] = {"T1": t1s, "T2": t2s}
    return out


# ── E3: hardware sweeps ──────────────────────────────────────────────────────

def run_hardware() -> dict:
    graphs = {"cycle12": build("cycle", 12), "complete8": build("complete", 8)}
    out: dict = {}
    for name, G in graphs.items():
        T2, chi = t2_cz(G)
        delta = max(d for _, d in G.degree())
        anc_k = list(range(1, 9))
        sw_k  = list(range(2, 13))
        out[name] = {
            "delta":   delta,
            "T2":      T2,
            "anc_k":   anc_k,
            "T3":      [t3_cz_k(G, k) for k in anc_k],
            "sw_k":    sw_k,
            "T5":      [t5_cz_sw(G, k) for k in sw_k],
            "T4":      [t4_mpp_sw(G, k) for k in sw_k],
            "k4_min":  delta + 2,       # smallest k with |support|+1 <= k
        }
    return out


# ── E4: MPP/CZ crossover on projective-plane incidence graphs ────────────────
#
# T1 is governed by the clique number of the generator conflict graph (cover
# vertices within distance 2 of each other), while T2 = 2 chi'(G) is governed
# by the degree.  Incidence graphs of projective planes PG(2,q) drive the two
# apart: every pair of points shares a line, so all point generators mutually
# conflict (conflict clique ~ n/2) while the degree stays q+1 ~ sqrt(n).

# Perfect difference sets for PG(2,q): points {0..m-1}, lines {D+i mod m}.
_PLANE_DS = {
    2: ([0, 1, 3], 7),
    3: ([0, 1, 3, 9], 13),
    4: ([0, 1, 4, 14, 16], 21),
    5: ([0, 1, 3, 8, 12, 18], 31),
}


def build_incidence(q: int) -> nx.Graph:
    """Point-line incidence graph of the projective plane PG(2,q).

    Nodes 0..m-1 are points, m..2m-1 are lines (m = q^2+q+1)."""
    ds, m = _PLANE_DS[q]
    G = nx.Graph()
    G.add_nodes_from(range(2 * m))
    for i in range(m):
        for d in ds:
            G.add_edge((d + i) % m, m + i)      # point (d+i) lies on line i
    return G


def conflict_clique(G: nx.Graph) -> int:
    """Clique number of the generator conflict graph after MIS reduction."""
    s = SubstrateScheduler(G)
    _quiet(s.phase_1_reduction)
    cover = list(s.to_measure)
    C = nx.Graph()
    C.add_nodes_from(cover)
    closed = {v: set(G[v]) | {v} for v in cover}
    for i, u in enumerate(cover):
        for v in cover[i + 1:]:
            if closed[u] & closed[v]:
                C.add_edge(u, v)
    # exact branch-and-bound; find_cliques would enumerate exponentially many
    # maximal cliques on these dense conflict graphs
    clique, _ = nx.max_weight_clique(C, weight=None)
    return max(len(clique), 1)


def run_crossover() -> list[dict]:
    rows = []
    graphs = [("path", build("path", 8)), ("cycle", build("cycle", 8)),
              ("star", build("star", 8)), ("complete", build("complete", 8)),
              ("grid", build("grid", 9)), ("random", build("random", 8))]
    graphs += [(f"PG(2,{q})", build_incidence(q)) for q in (2, 3, 4, 5)]
    for name, G in graphs:
        T2, chi = t2_cz(G)
        rows.append({
            "graph":  name,
            "n":      len(G),
            "delta":  max(d for _, d in G.degree()),
            "chi":    chi,
            "omega":  conflict_clique(G),
            "T1":     t1_mpp(G),
            "T2":     T2,
        })
    return rows


def run_gridflip() -> list[dict]:
    """Square grids: the congestion flip.  omega stays flat while the 1D
    substrate's interval congestion (cutwidth ~ side) drives T1 past T2."""
    rows = []
    for side in range(2, 8):
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        T2, chi = t2_cz(G)
        rows.append({
            "graph": f"{side}x{side}",
            "side":  side,
            "n":     len(G),
            "chi":   chi,
            "omega": conflict_clique(G),
            "T1":    t1_mpp(G),
            "T2":    T2,
        })
    return rows


# ── Figures ──────────────────────────────────────────────────────────────────

FAMILY_LBL = {
    "path": "Path $P_8$", "cycle": "Cycle $C_8$", "star": "Star $K_{1,7}$",
    "complete": "Complete $K_8$", "grid": "Grid $3{\\times}3$",
    "random": "Random $G(8,0.35)$",
}


def fig_survey(rows: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(_W, 2.9), facecolor="white")
    nfam, ncase = len(rows), 5
    width = 0.16
    for ci, case in enumerate([1, 2, 3, 4, 5]):
        xs, ys = [], []
        for fi, row in enumerate(rows):
            t = row[f"T{case}"]
            x = fi + (ci - 2) * width
            if t is None:
                ax.text(x, 0.6, "$\\times$", ha="center", va="bottom",
                        fontsize=9, color=CASE_COL[case], fontweight="bold")
            else:
                xs.append(x); ys.append(t)
        bars = ax.bar(xs, ys, width=width * 0.92, color=CASE_COL[case],
                      edgecolor="white", linewidth=0.6, zorder=2,
                      label=CASE_LBL[case])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(nfam))
    ax.set_xticklabels([FAMILY_LBL[r["family"]] for r in rows], fontsize=8)
    ax.set_ylabel("Tocks", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_ylim(0, max(r["T3"] for r in rows) * 1.30)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handlelength=1.0, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_scaling(data: dict, path: str) -> None:
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(_W, 4.4), facecolor="white",
                             sharex=True)
    panels = [("path", "(a) Path $P_n$"), ("cycle", "(b) Cycle $C_n$"),
              ("star", "(c) Star $K_{1,n-1}$"), ("complete", "(d) Complete $K_n$")]
    ns = data["n"]
    for ax, (family, title) in zip(axes.flat, panels):
        d = data[family]
        ax.plot(ns, d["T1"], "-o", color=CASE_COL[1], markersize=4,
                linewidth=1.4, label="Case 1 (MPP)", zorder=3)
        ax.plot(ns, d["T2"], "-s", color=CASE_COL[2], markersize=4,
                linewidth=1.4, label="Case 2 (CZ)", zorder=3)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.25, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[1]:
        ax.set_xlabel("Graph size $n$", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Tocks", fontsize=8.5)
    axes[0, 0].set_ylim(1.4, 4.9)   # room so the legend clears both flat lines
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoffs(hw: dict, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_W, 2.7), facecolor="white")
    series = [("cycle12", "$C_{12}$", CASE_COL[1], "o"),
              ("complete8", "$K_8$", CASE_COL[2], "s")]

    # (a) T3 vs ancilla pool size
    for name, lbl, col, mk in series:
        d = hw[name]
        ax1.plot(d["anc_k"], d["T3"], "-" + mk, color=col, markersize=4,
                 linewidth=1.4, label=lbl, zorder=3)
        ax1.axhline(d["T2"], color=col, linestyle=":", linewidth=1.0,
                    alpha=0.65, zorder=1)
    ax1.set_xlabel("Ancilla pool size $k_a$", fontsize=8.5)
    ax1.set_ylabel("Tocks", fontsize=8.5)
    ax1.set_title("(a) Case 3 — ancilla-limited", fontsize=9,
                  fontweight="bold", pad=5)

    # (b) T5 vs switch ports, with Case-4 feasibility thresholds
    for name, lbl, col, mk in series:
        d = hw[name]
        ks = [k for k, t in zip(d["sw_k"], d["T5"]) if t is not None]
        ts = [t for t in d["T5"] if t is not None]
        ax2.plot(ks, ts, "-" + mk, color=col, markersize=4,
                 linewidth=1.4, label=lbl, zorder=3)
        ax2.axhline(d["T2"], color=col, linestyle=":", linewidth=1.0,
                    alpha=0.65, zorder=1)
        ax2.axvline(d["k4_min"], color=col, linestyle="--", linewidth=1.0,
                    alpha=0.65, zorder=1)
    ax2.set_xlabel("Switch ports $k$", fontsize=8.5)
    ax2.set_title("(b) Case 5 — switch-limited", fontsize=9,
                  fontweight="bold", pad=5)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.25, zorder=0)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_crossover(rows: list[dict], grids: list[dict], path: str) -> None:
    """(a) conflict flip on PG(2,q); (b) congestion flip on square grids;
    (c) predictor scatter with the congestion exceptions marked."""
    fig = plt.figure(figsize=(_W, 5.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                          hspace=0.45, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # (a) the incidence family: T1 grows ~ n/2, T2 grows ~ 2 sqrt(n)
    pg = [r for r in rows if r["graph"].startswith("PG")]
    ns  = [r["n"] for r in pg]
    ax1.plot(ns, [r["T1"] for r in pg], "o-", color=CASE_COL[1],
             linewidth=1.4, markersize=4.5, label="MPP, $T_1$")
    ax1.plot(ns, [r["T2"] for r in pg], "s-", color=CASE_COL[2],
             linewidth=1.4, markersize=4.5, label="CZ, $T_2$")
    for r in pg:
        if r["graph"] == "PG(2,2)":
            ax1.annotate(r["graph"], (r["n"], r["T1"]), fontsize=7,
                         xytext=(7, -3), textcoords="offset points", ha="left")
        else:
            ax1.annotate(r["graph"], (r["n"], r["T1"]), fontsize=7,
                         xytext=(0, 6), textcoords="offset points",
                         ha="center")
    ax1.margins(x=0.1, y=0.14)
    ax1.set_xlabel("Graph size $n$", fontsize=9)
    ax1.set_ylabel("Tocks", fontsize=9)
    ax1.set_title("(a) Conflict: incidence graphs", fontsize=9.5,
                  fontweight="bold", pad=5)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    # (b) square grids: omega flat, T1 flat, MPP stays ahead at every size
    sides = [r["side"] for r in grids]
    ax2.plot(sides, [r["T1"] for r in grids], "o-", color=CASE_COL[1],
             linewidth=1.4, markersize=4.5, label="MPP, $T_1$")
    ax2.plot(sides, [r["T2"] for r in grids], "s-", color=CASE_COL[2],
             linewidth=1.4, markersize=4.5, label="CZ, $T_2$")
    ax2.plot(sides, [r["omega"] for r in grids], ":", color=CASE_COL[1],
             linewidth=1.2, label=r"$\omega(\mathcal{C})$ (flat)")
    ax2.margins(y=0.14)
    ax2.set_xlabel(r"Grid side $\sqrt{n}$", fontsize=9)
    ax2.set_title("(b) Grid graphs: MPP stays ahead", fontsize=9.5,
                  fontweight="bold", pad=5)
    ax2.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor="none",
               facecolor="white", loc="lower right")

    # (c) predictor scatter: shaded halves name the winner and a legend
    # names the graph families, so no per-point labels are needed.
    surv = [r for r in rows if not r["graph"].startswith("PG")]
    pgs  = [r for r in rows if r["graph"].startswith("PG")]
    rat = lambda r: r["omega"] / (2 * r["chi"])
    out = lambda r: r["T1"] / r["T2"]
    xs = [rat(r) for r in rows]
    ax3.plot([min(xs), max(xs)], [min(xs), max(xs)], color="#5A6473",
             linewidth=0.7, alpha=0.35, zorder=1)
    ax3.scatter([rat(r) for r in surv], [out(r) for r in surv], s=36,
                color=CASE_COL[1], zorder=3,
                label="common families (path, cycle, star, ...)")
    ax3.scatter([rat(r) for r in pgs], [out(r) for r in pgs], s=36,
                color=CASE_COL[2], zorder=3,
                label="incidence graphs $\\mathrm{PG}(2,q)$")
    ax3.scatter([rat(r) for r in grids], [out(r) for r in grids], s=36,
                marker="^", color=CASE_COL[1], zorder=3,
                label=r"square grids ($2{\times}2$ to $7{\times}7$)")
    ax3.axhline(1, color="#5A6473", linewidth=0.9, linestyle="--",
                alpha=0.6)
    ax3.axvline(1, color="#5A6473", linewidth=0.9, linestyle="--",
                alpha=0.6)
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.margins(x=0.08, y=0.14)
    ylo, yhi = ax3.get_ylim()
    ax3.axhspan(1.0, yhi, color=CASE_COL[2], alpha=0.08, zorder=0)
    ax3.axhspan(ylo, 1.0, color=CASE_COL[1], alpha=0.06, zorder=0)
    ax3.set_ylim(ylo, yhi)
    ax3.set_xlabel(r"conflict ratio  $\omega(\mathcal{C})\,/\,2\chi'$"
                   "  (rule predicts CZ past 1)", fontsize=9)
    ax3.set_ylabel(r"measured  $T_1/T_2$", fontsize=9)
    ax3.set_title("(c) Which operation wins", fontsize=9.5,
                  fontweight="bold", pad=5)
    ax3.text(0.02, 0.95, "CZ wins\n(upper half)", transform=ax3.transAxes,
             fontsize=9, ha="left", va="top", color=CASE_COL[2],
             fontweight="bold")
    ax3.text(0.98, 0.05, "MPP wins\n(lower half)", transform=ax3.transAxes,
             fontsize=9, ha="right", va="bottom", color=CASE_COL[1],
             fontweight="bold")
    ax3.legend(fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=3,
               columnspacing=1.4, handletextpad=0.4)

    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.25, zorder=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "Results/sweeps"
    os.makedirs(out_dir, exist_ok=True)

    survey    = run_survey()
    scaling   = run_scaling()
    hw        = run_hardware()
    crossover = run_crossover()
    gridflip  = run_gridflip()

    with open(os.path.join(out_dir, "sweeps.json"), "w") as f:
        json.dump({"survey": survey, "scaling": scaling, "hardware": hw,
                   "crossover": crossover, "gridflip": gridflip},
                  f, indent=2)

    fig_survey(survey, os.path.join(out_dir, "fig_survey.pdf"))
    fig_scaling(scaling, os.path.join(out_dir, "fig_scaling.pdf"))
    fig_tradeoffs(hw, os.path.join(out_dir, "fig_tradeoffs.pdf"))
    fig_crossover(crossover, gridflip,
                  os.path.join(out_dir, "fig_crossover.pdf"))

    print(f"Sweeps written to {out_dir}/")
    for r in survey:
        print(r)
    print("-- crossover --")
    for r in crossover:
        print(r)
    print("-- gridflip --")
    for r in gridflip:
        print(r)


if __name__ == "__main__":
    main()
