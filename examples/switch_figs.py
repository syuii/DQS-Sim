"""Port-wiring worked-example figures for the k-port switch cases.

Serif fonts, no side or bottom annotations, panels sized for a
single-column page.
"""

import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

mpl.rcParams.update({
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "CMU Serif", "Times New Roman"],
    "mathtext.fontset":  "cm",
    "text.color":        "#1A1A1A",
    "savefig.dpi":       300,
})

NAVY, LNAVY  = "#285A96", "#B8CDE4"
ORANGE, LORG = "#C1853C", "#E6C79A"
IDLE, IDLE_E = "#F2F3F5", "#C9CDD4"
SLATE, RED   = "#5A6473", "#B4534B"
GREY_A       = "#8C8C8C"


def _port_panel(ax, n, title, engaged, wires, dashed=(), k=4,
                tcol="#1a1a1a", show_anc=True, title_fs=9.5, node_fs=7.5):
    """One time-step: QPU row on top, k-port switch below, wires between."""
    names = list(range(n)) + (["anc"] if show_anc else [])
    xs = {nd: i * 1.0 for i, nd in enumerate(names)}
    if show_anc:
        xs["anc"] += 0.35
    W = max(xs.values()) + 1.0
    ax.set_xlim(-0.7, W)
    ax.set_ylim(-0.15, 3.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=title_fs, pad=4, color=tcol)

    for nd in names:
        fill, ec, tc = engaged.get(nd, (IDLE, IDLE_E, "#AEB4BE"))
        ax.add_patch(FancyBboxPatch((xs[nd] - 0.36, 2.45), 0.72, 0.62,
                                    boxstyle="round,pad=0.02",
                                    facecolor=fill, edgecolor=ec,
                                    linewidth=1.2, zorder=3))
        lb = "anc" if nd == "anc" else str(nd)
        ax.text(xs[nd], 2.76, lb, ha="center", va="center", fontsize=node_fs,
                color=tc, zorder=4)

    cx = W / 2 - 0.35
    ax.add_patch(FancyBboxPatch((cx - 1.7, 0.0), 3.4, 0.62,
                                boxstyle="round,pad=0.02", facecolor="white",
                                edgecolor=NAVY, linewidth=1.3, zorder=3))
    ax.text(cx, 0.31, "switch", ha="center", va="center", fontsize=node_fs,
            color=NAVY, zorder=4)
    wires = sorted(wires, key=lambda t: xs[t[0]])
    ports = [cx - 1.28 + i * 0.85 for i in range(k)]
    for i, px in enumerate(ports):
        used = i < len(wires)
        ax.add_patch(Circle((px, 0.72), 0.10,
                            facecolor=NAVY if used else "white",
                            edgecolor=NAVY, linewidth=1.0, zorder=5))
    for i, (nd, col) in enumerate(wires):
        ax.plot([xs[nd], ports[i]], [2.42, 0.83], color=col,
                linewidth=1.5, zorder=2, solid_capstyle="round")
    for nd, col in dashed:
        ex, ey = xs[nd] * 0.25 + (cx + 2.35) * 0.75, 1.15
        ax.plot([xs[nd], ex], [2.42, ey], color=RED, linewidth=1.4,
                linestyle=(0, (4, 3)), zorder=2)
        ax.plot([ex - 0.11, ex + 0.11], [ey - 0.11, ey + 0.11], color=RED,
                linewidth=1.7, zorder=6)
        ax.plot([ex - 0.11, ex + 0.11], [ey + 0.11, ey - 0.11], color=RED,
                linewidth=1.7, zorder=6)


def c6_switch_fig(out):
    fig = plt.figure(figsize=(6.0, 2.95), facecolor="white")
    gs = fig.add_gridspec(2, 12, hspace=0.3)
    groups = [(1, [0, 2]), (3, [2, 4]), (5, [0, 4])]
    for i, (v, nbrs) in enumerate(groups):
        ax = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
        engaged = {v: (NAVY, NAVY, "white")}
        for nb in nbrs:
            engaged[nb] = (LNAVY, NAVY, "#1a1a1a")
        engaged["anc"] = ("white", SLATE, "#1a1a1a")
        wires = [(v, NAVY)] + [(nb, NAVY) for nb in nbrs] + [("anc", GREY_A)]
        wires = wires[:4]
        _port_panel(ax, 6, f"Tock {i+1}:  measure $g_{v}$", engaged, wires,
                    tcol=NAVY)

    batches = [[(0, 1), (2, 3)], [(4, 5)], [(0, 5), (1, 2)], [(3, 4)]]
    for i, edges in enumerate(batches):
        ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        engaged, wires = {}, []
        for j, (u, v) in enumerate(edges):
            col, lf = (ORANGE, LORG) if j == 0 else (SLATE, "#DDE1E6")
            engaged[u] = (lf, col, "#1a1a1a")
            engaged[v] = (lf, col, "#1a1a1a")
            wires += [(u, col), (v, col)]
        _port_panel(ax, 6, f"Round {i+1} ({len(edges)} CZ)",
                    engaged, wires, tcol=ORANGE, show_anc=False,
                    title_fs=8, node_fs=6.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def k5_switch_fig(out):
    fig = plt.figure(figsize=(3.8, 1.15), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.18)

    ax = fig.add_subplot(gs[0])
    engaged = {0: (NAVY, NAVY, "white")}
    for nb in (1, 2):
        engaged[nb] = (LNAVY, NAVY, "#1a1a1a")
    for nb in (3, 4):
        engaged[nb] = ("#F6DEDB", RED, "#1a1a1a")
    engaged["anc"] = ("white", SLATE, "#1a1a1a")
    wires = [(0, NAVY), (1, NAVY), (2, NAVY), ("anc", GREY_A)]
    _port_panel(ax, 5, "Case 4", engaged,
                wires, dashed=[(3, RED), (4, RED)], tcol=RED)

    ax = fig.add_subplot(gs[1])
    engaged, wires = {}, []
    for j, (u, v) in enumerate([(0, 1), (2, 3)]):
        col, lf = (ORANGE, LORG) if j == 0 else (SLATE, "#DDE1E6")
        engaged[u] = (lf, col, "#1a1a1a")
        engaged[v] = (lf, col, "#1a1a1a")
        wires += [(u, col), (v, col)]
    _port_panel(ax, 5, "Case 5", engaged, wires,
                tcol=ORANGE)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _two_switch_panel(ax, n, engaged, wires, node_fs=7.5, names=None,
                      centers=None):
    """One time-step with TWO 4-port switches side by side.

    names: node row, ints or 'a1'/'a2' ancilla entries (extra gap, math label).
    wires: [(node, colour, switch_idx)] -> next free port of that switch.
    """
    if names is None:
        names = list(range(n))
    xs, x = {}, 0.0
    for nd in names:
        if isinstance(nd, str):
            x += 0.35
        xs[nd] = x
        x += 1.0
    W = max(xs.values()) + 1.0
    ax.set_xlim(-0.7, W)
    ax.set_ylim(-0.15, 3.35)
    ax.set_aspect("equal")
    ax.axis("off")

    for nd in names:
        fill, ec, tc = engaged.get(nd, (IDLE, IDLE_E, "#AEB4BE"))
        ax.add_patch(FancyBboxPatch((xs[nd] - 0.36, 2.45), 0.72, 0.62,
                                    boxstyle="round,pad=0.02",
                                    facecolor=fill, edgecolor=ec,
                                    linewidth=1.2, zorder=3))
        lb = f"$a_{nd[1:]}$" if isinstance(nd, str) else str(nd)
        ax.text(xs[nd], 2.76, lb, ha="center", va="center",
                fontsize=node_fs, color=tc, zorder=4)

    if centers is None:
        centers = [W * 0.25 - 0.35, W * 0.75 - 0.15]
    ports = {}
    for s, cx in enumerate(centers):
        ax.add_patch(FancyBboxPatch((cx - 1.5, 0.0), 3.0, 0.62,
                                    boxstyle="round,pad=0.02",
                                    facecolor="white", edgecolor=NAVY,
                                    linewidth=1.3, zorder=3))
        ax.text(cx, 0.31, f"switch {s + 1}", ha="center", va="center",
                fontsize=node_fs, color=NAVY, zorder=4)
        ports[s] = [cx - 1.125 + i * 0.75 for i in range(4)]

    used = {0: 0, 1: 0}
    for nd, col, s in sorted(wires, key=lambda t: xs[t[0]]):
        px = ports[s][used[s]]
        used[s] += 1
        ax.plot([xs[nd], px], [2.42, 0.83], color=col, linewidth=1.5,
                zorder=2, solid_capstyle="round")
    for s in (0, 1):
        for i, px in enumerate(ports[s]):
            ax.add_patch(Circle((px, 0.72), 0.10,
                                facecolor=NAVY if i < used[s] else "white",
                                edgecolor=NAVY, linewidth=1.0, zorder=5))


def c8_mpp_two_switch_fig(out):
    fig, ax = plt.subplots(figsize=(3.0, 1.0), facecolor="white")
    names = [0, 1, 2, 3, "a1", 4, 5, 6, 7, "a2"]
    engaged = {1: (NAVY, NAVY, "white"), 5: (NAVY, NAVY, "white"),
               "a1": ("white", SLATE, "#1a1a1a"),
               "a2": ("white", SLATE, "#1a1a1a")}
    for nb in (0, 2, 4, 6):
        engaged[nb] = (LNAVY, NAVY, "#1a1a1a")
    wires = [(0, NAVY, 0), (1, NAVY, 0), (2, NAVY, 0), ("a1", GREY_A, 0),
             (4, NAVY, 1), (5, NAVY, 1), (6, NAVY, 1), ("a2", GREY_A, 1)]
    _two_switch_panel(ax, 10, engaged, wires, names=names,
                      centers=(2.0, 7.55))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def c8_cz_two_switch_fig(out):
    fig, ax = plt.subplots(figsize=(2.4, 1.0), facecolor="white")
    gates = [((0, 1), 0), ((2, 3), 0), ((4, 5), 1), ((6, 7), 1)]
    engaged, wires = {}, []
    for j, ((u, v), s) in enumerate(gates):
        col, lf = (ORANGE, LORG) if j % 2 == 0 else (SLATE, "#DDE1E6")
        engaged[u] = (lf, col, "#1a1a1a")
        engaged[v] = (lf, col, "#1a1a1a")
        wires += [(u, col, s), (v, col, s)]
    _two_switch_panel(ax, 8, engaged, wires)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    c6_switch_fig(os.path.join(out_dir, "c6_switch_ports.pdf"))
    k5_switch_fig(os.path.join(out_dir, "k5_switch_ports.pdf"))
    c8_cz_two_switch_fig(os.path.join(out_dir, "c8_cz_switches.pdf"))
    c8_mpp_two_switch_fig(os.path.join(out_dir, "c8_mpp_switches.pdf"))
    print("written to", out_dir)
