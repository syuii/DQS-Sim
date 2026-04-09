"""
visualizer.py
=============
Clean, scalable visualisations for the graph state generation simulator.

Four main outputs
-----------------
  show_heatmaps(scheduler, cz)
      Compact schedule matrix — works at any vertex count.
      Rows = qubits, columns = tocks, colour = role.

  show_2d_layout(scheduler, cz)
      Each ELU drawn as a physical patch (rounded square).
      Edges coloured by tock (Case 1) or CZ round (Case 2).

  show_gantt(scheduler, cz)
      Horizontal bar chart of qubit activity over time.
      Smart tick density so axes stay readable at scale.

  show_comparison_dashboard(report)
      Three-panel bar chart: tocks / peak ancilla / physical qubits.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

if TYPE_CHECKING:
    from cost_estimator import ResourceReport
    from substrate_scheduler import CZScheduler, SubstrateScheduler


# ── Colour palette (perceptually distinct, print-safe) ──────────────────────
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F",
]

# Semantic colours
_COL_IDLE     = "#F4F4F4"   # idle patch fill
_COL_MIS      = "#A8D8A8"   # MIS (|+⟩ init) fill
_COL_MEASURE  = "#F4A7A7"   # to-measure fill
_COL_BORDER   = "#555555"   # default patch border
_COL_EDGE_OFF = "#DDDDDD"   # idle edge colour
_FIG_BG       = "white"


# ============================================================================
# Internal geometry helpers
# ============================================================================

def _normalize_pos(pos: dict, margin: float = 0.10) -> dict:
    """Scale spring-layout positions to [margin, 1-margin]²."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_rng = x_max - x_min or 1.0
    y_rng = y_max - y_min or 1.0
    s = 1.0 - 2 * margin
    return {
        n: (margin + s * (p[0] - x_min) / x_rng,
            margin + s * (p[1] - y_min) / y_rng)
        for n, p in pos.items()
    }


def _adaptive_hw(pos: dict) -> float:
    """
    Compute patch half-width = 30 % of the minimum inter-node distance,
    clamped to a sensible range.
    """
    pts = list(pos.values())
    if len(pts) < 2:
        return 0.05
    min_d = min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(pts)
        for b in pts[i + 1:]
    )
    return max(0.012, min(0.055, min_d * 0.30))


def _border_exit(cx: float, cy: float,
                 tx: float, ty: float, hw: float) -> tuple[float, float]:
    """
    Return the point where the ray from (cx, cy) toward (tx, ty)
    first exits the square patch of half-width *hw* centred at (cx, cy).
    """
    dx, dy = tx - cx, ty - cy
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return cx + hw, cy
    t_x = hw / abs(dx) if abs(dx) > 1e-10 else float("inf")
    t_y = hw / abs(dy) if abs(dy) > 1e-10 else float("inf")
    t   = min(t_x, t_y)
    return cx + dx * t, cy + dy * t


# ── Tick helpers ─────────────────────────────────────────────────────────────

def _smart_ticks(ax: plt.Axes, n: int, axis: str = "x",
                 labels: list[str] | None = None, fontsize: int = 8) -> None:
    """Label every k-th tick so the axis stays readable at any scale."""
    step   = max(1, n // 20)
    ticks  = list(range(0, n, step))
    lbl    = [labels[i] for i in ticks] if labels else [str(i) for i in ticks]
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    labset = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    setter(ticks)
    labset(lbl, fontsize=fontsize)


def _fig_height(n: int) -> float:
    return max(4.0, min(n * 0.30 + 2.5, 22.0))


# ============================================================================
# 2D Patch Layout
# ============================================================================

def _draw_patch_layout(
    pos:        dict,           # normalised {node: (x, y)}
    node_fill:  dict,           # {node: hex}
    node_lw:    dict,           # {node: float}
    node_ec:    dict,           # {node: hex}
    edge_col:   dict,           # {(u,v): hex}  canonical u < v
    edge_lw:    dict,           # {(u,v): float}
    edges:      list,           # all edges to draw (from G)
    ax:         plt.Axes,
    title:      str,
    hw:         float,
    show_labels: bool = True,
) -> None:
    """
    Core 2D patch renderer.  Draws edges first (lines from border to border),
    then ELU patches (rounded squares) on top.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # ── edges ────────────────────────────────────────────────────────────────
    for u, v in edges:
        cx_u, cy_u = pos[u]
        cx_v, cy_v = pos[v]
        ex_u, ey_u = _border_exit(cx_u, cy_u, cx_v, cy_v, hw)
        ex_v, ey_v = _border_exit(cx_v, cy_v, cx_u, cy_u, hw)
        key = (min(u, v), max(u, v))
        col = edge_col.get(key, _COL_EDGE_OFF)
        lw  = edge_lw.get(key, 0.8)
        ax.plot([ex_u, ex_v], [ey_u, ey_v],
                color=col, linewidth=lw,
                solid_capstyle="round", zorder=1)

    # ── patches ──────────────────────────────────────────────────────────────
    for node, (cx, cy) in pos.items():
        fc  = node_fill.get(node, _COL_IDLE)
        ec  = node_ec.get(node, _COL_BORDER)
        lw  = node_lw.get(node, 1.2)
        box = FancyBboxPatch(
            (cx - hw, cy - hw), 2 * hw, 2 * hw,
            boxstyle="round,pad=0.015",
            facecolor=fc, edgecolor=ec,
            linewidth=lw, zorder=2,
        )
        ax.add_patch(box)

        # Port indicator dots (N / S / E / W communication qubits)
        dot_r = hw * 0.18
        dot_c = "#AAAAAA"
        for dx, dy in [(0, hw), (0, -hw), (hw, 0), (-hw, 0)]:
            ax.add_patch(plt.Circle(
                (cx + dx * 0.92, cy + dy * 0.92), dot_r,
                color=dot_c, zorder=3,
            ))

        # Label
        if show_labels:
            ax.text(cx, cy, str(node),
                    ha="center", va="center",
                    fontsize=max(6, min(10, int(hw * 160))),
                    fontweight="bold", color="#2C3E50", zorder=4)


def show_2d_layout(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    save_path: str | None = None,
) -> None:
    """
    Two-panel figure showing ELUs as physical surface-code patches.

    Left  — Case 1: MIS nodes (green) vs to-measure (pink).
             Edges coloured by the Tock in which they are activated.
    Right — Case 2: Edges coloured by CZ round.

    Small dots on each patch edge represent the four communication-qubit
    ports used for lattice-surgery Bell pairs.
    """
    G   = scheduler.G
    n   = len(G.nodes)
    raw = scheduler._get_pos()
    pos = _normalize_pos(raw)
    hw  = _adaptive_hw(pos)
    show_labels = n <= 30

    # Build edge → tock map for Case 1
    edge_tock: dict[tuple, int] = {}
    for t, tock in enumerate(scheduler.schedule):
        for b in tock:
            u = b["id"]
            for v in b["neighbors"]:
                key = (min(u, v), max(u, v))
                if key not in edge_tock:
                    edge_tock[key] = t

    # Build edge → round map for Case 2
    edge_round: dict[tuple, int] = {
        (min(u, v), max(u, v)): r
        for r, edges in enumerate(cz.cz_rounds)
        for u, v in edges
    }

    all_edges = list(G.edges())
    n_tocks   = len(scheduler.schedule)
    n_rounds  = cz.chromatic_index

    # ── Case 1 colours ───────────────────────────────────────────────────────
    c1_node_fill = {
        nd: (_COL_MIS if nd in scheduler.mis else _COL_MEASURE)
        for nd in G.nodes
    }
    c1_node_ec = {
        nd: ("#4CAF50" if nd in scheduler.mis else "#E53935")
        for nd in G.nodes
    }
    c1_node_lw  = {nd: 1.8 for nd in G.nodes}
    c1_edge_col = {
        (min(u, v), max(u, v)): _PALETTE[edge_tock[(min(u,v),max(u,v))] % len(_PALETTE)]
        if (min(u,v), max(u,v)) in edge_tock else _COL_EDGE_OFF
        for u, v in all_edges
    }
    c1_edge_lw = {
        (min(u, v), max(u, v)): (
            1.8 if (min(u,v),max(u,v)) in edge_tock else 0.6
        )
        for u, v in all_edges
    }

    # ── Case 2 colours ───────────────────────────────────────────────────────
    c2_node_fill = {nd: _COL_IDLE for nd in G.nodes}
    c2_node_ec   = {nd: _COL_BORDER for nd in G.nodes}
    c2_node_lw   = {nd: 1.2 for nd in G.nodes}
    c2_edge_col  = {
        (min(u, v), max(u, v)): _PALETTE[edge_round[(min(u,v),max(u,v))] % len(_PALETTE)]
        if (min(u,v), max(u,v)) in edge_round else _COL_EDGE_OFF
        for u, v in all_edges
    }
    c2_edge_lw = {
        (min(u, v), max(u, v)): 2.0
        for u, v in all_edges
    }

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8), facecolor=_FIG_BG,
    )
    fig.suptitle("2D ELU Patch Layout", fontsize=15, fontweight="bold", y=1.01)

    _draw_patch_layout(
        pos, c1_node_fill, c1_node_lw, c1_node_ec,
        c1_edge_col, c1_edge_lw, all_edges,
        ax1,
        f"Case 1 — MPP  (edges coloured by Tock, {n_tocks} total)",
        hw, show_labels,
    )
    _draw_patch_layout(
        pos, c2_node_fill, c2_node_lw, c2_node_ec,
        c2_edge_col, c2_edge_lw, all_edges,
        ax2,
        f"Case 2 — CZ Gates  (edges coloured by Round, {n_rounds} total)",
        hw, show_labels,
    )

    # ── Legends ──────────────────────────────────────────────────────────────
    # Case 1 node legend
    ax1.legend(
        handles=[
            mpatches.Patch(facecolor=_COL_MIS,     edgecolor="#4CAF50", lw=1.5, label="MIS — init |+⟩"),
            mpatches.Patch(facecolor=_COL_MEASURE,  edgecolor="#E53935", lw=1.5, label="Measured"),
        ],
        fontsize=9, frameon=True, framealpha=0.9,
        loc="lower left", borderpad=0.8,
    )

    # Case 1 tock legend (compact, only first 8 colours)
    n_show = min(n_tocks, 8)
    ax1_leg2_handles = [
        mpatches.Patch(facecolor=_PALETTE[t % len(_PALETTE)],
                       label=f"Tock {t + 1}")
        for t in range(n_show)
    ]
    if n_tocks > 8:
        ax1_leg2_handles.append(
            mpatches.Patch(facecolor="#ccc", label=f"… +{n_tocks - 8} more")
        )
    ax1.legend(
        handles=ax1_leg2_handles,
        fontsize=8, frameon=True, framealpha=0.9,
        loc="lower right", ncol=2, borderpad=0.8,
        title="Tock", title_fontsize=8,
    )

    # Case 2 round legend
    n_show2 = min(n_rounds, 8)
    ax2_handles = [
        mpatches.Patch(facecolor=_PALETTE[r % len(_PALETTE)],
                       label=f"Round {r + 1}")
        for r in range(n_show2)
    ]
    if n_rounds > 8:
        ax2_handles.append(
            mpatches.Patch(facecolor="#ccc", label=f"… +{n_rounds - 8} more")
        )
    ax2.legend(
        handles=ax2_handles,
        fontsize=8, frameon=True, framealpha=0.9,
        loc="lower right", ncol=2, borderpad=0.8,
        title="CZ Round", title_fontsize=8,
    )

    plt.tight_layout()
    _save_or_show(fig, save_path, "2D layout")


# ============================================================================
# Schedule Heatmap
# ============================================================================

def _build_heatmap_case1(scheduler: "SubstrateScheduler") -> np.ndarray:
    """Matrix[qubit_row, tock] = 0 (idle) | 1 (Z-neighbour) | 2 (X-measured)."""
    G   = scheduler.G
    nds = list(G.nodes)
    ntr = {n: i for i, n in enumerate(nds)}
    m   = np.zeros((len(nds), len(scheduler.schedule)), dtype=np.int8)
    for t, tock in enumerate(scheduler.schedule):
        for b in tock:
            m[ntr[b["id"]], t] = 2
            for nb in b["neighbors"]:
                if m[ntr[nb], t] == 0:
                    m[ntr[nb], t] = 1
    return m


def _build_heatmap_case2(cz: "CZScheduler") -> np.ndarray:
    """Matrix[qubit_row, tock] = 0 (idle) | 1 (step-1 qubit) | 2 (step-2 qubit)."""
    G   = cz.G
    nds = list(G.nodes)
    ntr = {n: i for i, n in enumerate(nds)}
    n_t = 2 * cz.chromatic_index
    m   = np.zeros((len(nds), n_t), dtype=np.int8)
    for r, edges in enumerate(cz.cz_rounds):
        t1, t2 = 2 * r, 2 * r + 1
        for u, v in edges:
            m[ntr[u], t1] = 1
            m[ntr[v], t2] = 2
    return m


def plot_heatmap(
    matrix:   np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title:    str,
    cmap:     ListedColormap,
    legend_handles: list,
    ax:       plt.Axes,
    vline_every: int = 0,
) -> None:
    """
    Generic schedule heatmap renderer using imshow.
    Works cleanly for any number of qubits and tocks.
    """
    n_q, n_t = matrix.shape

    ax.imshow(
        matrix, aspect="auto",
        cmap=cmap, vmin=0, vmax=cmap.N - 1,
        interpolation="nearest", origin="upper",
    )

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_q, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, zorder=3)
    ax.tick_params(which="minor", length=0)

    # Optional vertical dividers (e.g. every CZ round)
    if vline_every > 0:
        for x in range(vline_every - 1, n_t, vline_every):
            ax.axvline(x + 0.5, color="#444444", linewidth=0.8, zorder=4)

    # Smart axis labels
    x_step = max(1, n_t // 25)
    y_step = max(1, n_q // 25)
    ax.set_xticks(range(0, n_t, x_step))
    ax.set_xticklabels(x_labels[::x_step], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(0, n_q, y_step))
    ax.set_yticklabels(y_labels[::y_step], fontsize=7)

    ax.set_xlabel("Tock  (logical time step)", fontsize=10)
    ax.set_ylabel("Qubit / ELU", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    ax.legend(
        handles=legend_handles,
        fontsize=8, frameon=True, framealpha=0.9,
        loc="upper right", bbox_to_anchor=(1.0, -0.14),
        ncol=len(legend_handles), borderpad=0.6,
    )


def show_heatmaps(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    save_path: str | None = None,
) -> None:
    """
    Side-by-side schedule heatmaps for Case 1 and Case 2.

    Every cell is one (qubit, tock) pair.  Works cleanly for any graph size —
    at high vertex counts the cell shading communicates the schedule pattern
    even without readable labels.
    """
    G   = scheduler.G
    nds = list(G.nodes)
    n_q = len(nds)

    mat1 = _build_heatmap_case1(scheduler)
    mat2 = _build_heatmap_case2(cz)
    n_t1 = mat1.shape[1]
    n_t2 = mat2.shape[1]

    # Case 1: white | light blue | deep blue
    cmap1 = ListedColormap(["#F4F4F4", "#90CAF9", "#1565C0"])
    # Case 2: white | amber | deep orange
    cmap2 = ListedColormap(["#F4F4F4", "#FFCC80", "#E65100"])

    x_lbl1 = [f"T{t+1}" for t in range(n_t1)]
    x_lbl2 = [f"T{t+1}" for t in range(n_t2)]
    y_lbl  = [f"Q{nd}"  for nd in nds]

    legend1 = [
        mpatches.Patch(facecolor="#F4F4F4", edgecolor="#ccc", label="Idle"),
        mpatches.Patch(facecolor="#90CAF9",                   label="Z-neighbour"),
        mpatches.Patch(facecolor="#1565C0",                   label="X-measured"),
    ]
    legend2 = [
        mpatches.Patch(facecolor="#F4F4F4", edgecolor="#ccc", label="Idle"),
        mpatches.Patch(facecolor="#FFCC80",                   label="Step 1  (Z_u⊗Z_anc)"),
        mpatches.Patch(facecolor="#E65100",                   label="Step 2  (Z_v⊗Z_anc)"),
    ]

    height = max(5, min(n_q * 0.22 + 3, 20))
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(max(14, n_t1 * 0.4 + 8), height),
        facecolor=_FIG_BG,
    )
    fig.suptitle("Schedule Heatmaps", fontsize=14, fontweight="bold")

    plot_heatmap(mat1, x_lbl1, y_lbl,
                 f"Case 1 — MPP  ({n_t1} Tock{'s' if n_t1!=1 else ''})",
                 cmap1, legend1, ax1)
    plot_heatmap(mat2, x_lbl2, y_lbl,
                 f"Case 2 — CZ  ({n_t2} Tock{'s' if n_t2!=1 else ''},"
                 f"  {cz.chromatic_index} rounds)",
                 cmap2, legend2, ax2,
                 vline_every=2)   # divider every CZ round

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save_or_show(fig, save_path, "heatmap")


# ============================================================================
# Gantt charts  (improved: smart y-ticks, cleaner grid)
# ============================================================================

def plot_gantt_case1(scheduler: "SubstrateScheduler", ax: plt.Axes) -> None:
    """Gantt chart for Case 1 — rows sorted by spectral substrate position."""
    G            = scheduler.G
    nodes_sorted = sorted(G.nodes, key=lambda nd: scheduler.mapping[nd])
    node_to_row  = {nd: i for i, nd in enumerate(nodes_sorted)}
    n_t          = len(scheduler.schedule)
    n_q          = len(nodes_sorted)

    ax.set_xlim(0, n_t)
    ax.set_ylim(-0.5, n_q - 0.5)
    ax.set_facecolor(_FIG_BG)
    ax.set_xlabel("Tock  (logical time step)", fontsize=10)
    ax.set_ylabel("Qubit / ELU  (substrate order)", fontsize=10)
    ax.set_title("Case 1 — MPP Schedule", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    x_lbl = [f"T{t+1}" for t in range(n_t)]
    y_lbl = [f"Q{scheduler.mapping[nd]}" for nd in nodes_sorted]
    _smart_ticks(ax, n_t,  "x", x_lbl, fontsize=8)
    _smart_ticks(ax, n_q,  "y", y_lbl, fontsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
    ax.grid(axis="y", linestyle=":", alpha=0.18, zorder=0)

    for t, tock in enumerate(scheduler.schedule):
        col = _PALETTE[t % len(_PALETTE)]
        for b in tock:
            row = node_to_row[b["id"]]
            ax.broken_barh([(t + 0.07, 0.86)], (row - 0.38, 0.76),
                           facecolors=col, edgecolors="white",
                           linewidth=0.6, alpha=0.92, zorder=2)
            if n_q <= 40:
                ax.text(t + 0.5, row, "X", ha="center", va="center",
                        fontsize=6, fontweight="bold", color="white", zorder=3)
            for nb in b["neighbors"]:
                r_nb = node_to_row[nb]
                ax.broken_barh([(t + 0.07, 0.86)], (r_nb - 0.38, 0.76),
                               facecolors=col, edgecolors="white",
                               linewidth=0.6, alpha=0.32, zorder=2)
                if n_q <= 40:
                    ax.text(t + 0.5, r_nb, "Z", ha="center", va="center",
                            fontsize=6, color="white", zorder=3)

    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#444", alpha=0.92, label="X-measured"),
            mpatches.Patch(facecolor="#444", alpha=0.32, label="Z-neighbour"),
        ],
        fontsize=8, frameon=False, loc="upper right",
    )


def plot_gantt_case2(cz: "CZScheduler", ax: plt.Axes) -> None:
    """Gantt chart for Case 2 — each CZ round = 2 consecutive Tocks."""
    G            = cz.G
    nodes_list   = list(G.nodes)
    node_to_row  = {nd: i for i, nd in enumerate(nodes_list)}
    n_q          = len(nodes_list)
    n_t          = 2 * cz.chromatic_index

    ax.set_xlim(0, n_t)
    ax.set_ylim(-0.5, n_q - 0.5)
    ax.set_facecolor(_FIG_BG)
    ax.set_xlabel("Tock  (logical time step)", fontsize=10)
    ax.set_ylabel("Qubit / ELU", fontsize=10)
    ax.set_title("Case 2 — CZ-Gate Schedule", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    x_lbl = [f"T{t+1}" for t in range(n_t)]
    y_lbl = [f"Q{nd}"  for nd in nodes_list]
    _smart_ticks(ax, n_t, "x", x_lbl, fontsize=8)
    _smart_ticks(ax, n_q, "y", y_lbl, fontsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
    ax.grid(axis="y", linestyle=":", alpha=0.18, zorder=0)

    for r, edges in enumerate(cz.cz_rounds):
        col   = _PALETTE[r % len(_PALETTE)]
        t1, t2 = 2 * r, 2 * r + 1
        # Light band per round
        ax.axvspan(t1, t2 + 1, alpha=0.07, color=col, zorder=0)
        if n_t <= 80:
            ax.text(t1 + 1.0, n_q - 0.3,
                    f"R{r+1}", ha="center", va="center",
                    fontsize=6, color=col, fontweight="bold", zorder=3)
        for u, v in edges:
            for tock, qubit in [(t1, u), (t2, v)]:
                row = node_to_row[qubit]
                ax.broken_barh([(tock + 0.07, 0.86)], (row - 0.38, 0.76),
                               facecolors=col, edgecolors="white",
                               linewidth=0.6, alpha=0.85, zorder=2)
                if n_q <= 40:
                    ax.text(tock + 0.5, row, "Z", ha="center", va="center",
                            fontsize=6, fontweight="bold", color="white", zorder=3)


def show_gantt(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    save_path: str | None = None,
) -> None:
    """Side-by-side Gantt charts for both cases."""
    n_q    = len(scheduler.G.nodes)
    height = _fig_height(n_q)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, height),
                                   facecolor=_FIG_BG)
    fig.suptitle("Schedule Gantt Charts", fontsize=14, fontweight="bold")
    plot_gantt_case1(scheduler, ax1)
    plot_gantt_case2(cz, ax2)
    plt.tight_layout()
    _save_or_show(fig, save_path, "Gantt chart")


# ============================================================================
# Comparison dashboard  (bar charts — unchanged logic, cleaner styling)
# ============================================================================

def show_comparison_dashboard(
    report:    "ResourceReport",
    save_path: str | None = None,
) -> None:
    """Three-panel bar chart: tocks / peak ancilla ELUs / physical qubits."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    fig.suptitle(
        f"Resource Comparison — {report.graph_type}(n={report.graph_nodes}), "
        f"d={report.code_distance}",
        fontsize=13, fontweight="bold",
    )

    cases  = ["Case 1\n(MPP)", "Case 2\n(CZ)"]
    colors = ["#4C72B0", "#DD8452"]

    def _panel(ax: plt.Axes, vals: list, title: str,
               ylabel: str, fmt: str = "{}") -> None:
        bars = ax.bar(cases, vals, color=colors, edgecolor="white",
                      width=0.48, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.set_ylim(0, max(vals) * 1.28)
        for bar, val in zip(bars, vals):
            lbl = fmt.format(val) if isinstance(val, int) else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    lbl, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    _panel(axes[0],
           [report.tocks_case1, report.tocks_case2],
           "Total Tocks\n(logical time steps)", "Tocks")

    _panel(axes[1],
           [report.peak_ancilla_ELUs_case1, report.peak_ancilla_ELUs_case2],
           "Peak Ancilla ELUs\n(simultaneous demand)", "ELUs")

    _panel(axes[2],
           [report.total_phys_qubits_case1, report.total_phys_qubits_case2],
           f"Total Physical Qubits\n(d={report.code_distance}, "
           f"{report.phys_per_ELU}/ELU)",
           "Physical qubits", fmt="{:,}")

    # Tock-ratio annotation
    if report.tock_ratio_c2_c1 < float("inf"):
        axes[0].annotate(
            f"×{report.tock_ratio_c2_c1:.1f}",
            xy=(1, report.tocks_case2),
            xytext=(0.62, report.tocks_case2 * 1.16),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
            fontsize=10, fontweight="bold", ha="center", color="#444",
        )

    plt.tight_layout()
    _save_or_show(fig, save_path, "dashboard")


# ============================================================================
# Shared save/show helper
# ============================================================================

def _save_or_show(fig: plt.Figure, save_path: str | None, label: str) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  {label.capitalize()} saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
