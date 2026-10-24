"""
visualizer.py
=============
Clean, scalable visualisations for the graph state generation simulator.

Four main outputs
-----------------
  show_heatmaps(scheduler, cz)
      Compact schedule matrix - works at any vertex count.
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
    from substrate_scheduler import CZScheduler, CZSchedulerFixedAncilla, SubstrateScheduler


# Colour palette (perceptually distinct, print-safe)
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F",
]

# Semantic colours
_COL_IDLE     = "#F4F4F4"  # idle patch fill
_COL_MIS      = "#A8D8A8"  # MIS (|+⟩ init) fill
_COL_MEASURE  = "#F4A7A7"  # to-measure fill
_COL_BORDER   = "#555555"  # default patch border
_COL_EDGE_OFF = "#DDDDDD"  # idle edge colour
_FIG_BG       = "white"


# Internal geometry helpers

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


# Tick helpers

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


# 2D Patch Layout

def _draw_patch_layout(
    pos:        dict,  # normalised {node: (x, y)}
    node_fill:  dict,  # {node: hex}
    node_lw:    dict,  # {node: float}
    node_ec:    dict,  # {node: hex}
    edge_col:   dict,  # {(u,v): hex}  canonical u < v
    edge_lw:    dict,  # {(u,v): float}
    edges:      list,  # all edges to draw (from G)
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

    # edges
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

    # patches
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
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Two-panel figure showing ELUs as physical surface-code patches.

    Left  - Case 1: MIS nodes (green) vs to-measure (pink).
             Edges coloured by the Tock in which they are activated.
    Right - Case 2: Edges coloured by CZ round.

    When cz_k is provided, a third panel is added for Case 3 with edges
    coloured by sub-batch index.

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

    # Case 1 colours
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

    # Case 2 colours
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

    # Figure
    has_case3 = cz_k is not None
    fig_w = 24 if has_case3 else 16
    if has_case3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_w, 8), facecolor=_FIG_BG)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 8), facecolor=_FIG_BG)
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

    # Legends
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

    # Case 3 panel (optional)
    if has_case3:
        edge_batch: dict[tuple, int] = {
            (min(u, v), max(u, v)): b
            for b, batch in enumerate(cz_k.batches)
            for u, v in batch
        }
        c3_node_fill = {nd: _COL_IDLE   for nd in G.nodes}
        c3_node_ec   = {nd: _COL_BORDER for nd in G.nodes}
        c3_node_lw   = {nd: 1.2         for nd in G.nodes}
        c3_edge_col  = {
            (min(u, v), max(u, v)): _PALETTE[edge_batch[(min(u,v),max(u,v))] % len(_PALETTE)]
            if (min(u,v), max(u,v)) in edge_batch else _COL_EDGE_OFF
            for u, v in all_edges
        }
        c3_edge_lw   = {(min(u, v), max(u, v)): 2.0 for u, v in all_edges}

        n_batches = len(cz_k.batches)
        _draw_patch_layout(
            pos, c3_node_fill, c3_node_lw, c3_node_ec,
            c3_edge_col, c3_edge_lw, all_edges,
            ax3,
            f"Case 3 — CZ Gates  (edges coloured by Sub-batch, k={cz_k.k}, {n_batches} total)",
            hw, show_labels,
        )
        n_show3 = min(n_batches, 8)
        ax3_handles = [
            mpatches.Patch(facecolor=_PALETTE[b % len(_PALETTE)],
                           label=f"Batch {b + 1}")
            for b in range(n_show3)
        ]
        if n_batches > 8:
            ax3_handles.append(
                mpatches.Patch(facecolor="#ccc", label=f"… +{n_batches - 8} more")
            )
        ax3.legend(
            handles=ax3_handles,
            fontsize=8, frameon=True, framealpha=0.9,
            loc="lower right", ncol=2, borderpad=0.8,
            title="Sub-batch", title_fontsize=8,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path, "2D layout", display)


# Schedule Heatmap

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


def _build_heatmap_case3(cz_k: "CZSchedulerFixedAncilla") -> np.ndarray:
    """Matrix[qubit_row, tock] = 0 (idle) | 1 (step-1 qubit) | 2 (step-2 qubit)."""
    G   = cz_k.G
    nds = list(G.nodes)
    ntr = {n: i for i, n in enumerate(nds)}
    n_t = cz_k.n_tocks()
    m   = np.zeros((len(nds), n_t), dtype=np.int8)
    for b, batch in enumerate(cz_k.batches):
        t1, t2 = 2 * b, 2 * b + 1
        for u, v in batch:
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
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Side-by-side schedule heatmaps for Case 1 and Case 2.

    Every cell is one (qubit, tock) pair.  Works cleanly for any graph size -
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

    has_case3 = cz_k is not None
    if has_case3:
        mat3 = _build_heatmap_case3(cz_k)
        n_t3 = mat3.shape[1]
    else:
        n_t3 = 0

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
    fig_w  = max(14, (n_t1 + n_t2 + n_t3) * 0.3 + 8)

    if has_case3:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(fig_w, height), facecolor=_FIG_BG,
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(fig_w, height), facecolor=_FIG_BG,
        )
    fig.suptitle("Schedule Heatmaps", fontsize=14, fontweight="bold")

    plot_heatmap(mat1, x_lbl1, y_lbl,
                 f"Case 1 — MPP  ({n_t1} Tock{'s' if n_t1!=1 else ''})",
                 cmap1, legend1, ax1)
    plot_heatmap(mat2, x_lbl2, y_lbl,
                 f"Case 2 — CZ  ({n_t2} Tock{'s' if n_t2!=1 else ''},"
                 f"  {cz.chromatic_index} rounds)",
                 cmap2, legend2, ax2,
                 vline_every=2)  # divider every CZ round

    if has_case3:
        x_lbl3  = [f"T{t+1}" for t in range(n_t3)]
        legend3 = [
            mpatches.Patch(facecolor="#F4F4F4", edgecolor="#ccc", label="Idle"),
            mpatches.Patch(facecolor="#FFCC80",                   label="Step 1  (Z_u⊗Z_anc)"),
            mpatches.Patch(facecolor="#E65100",                   label="Step 2  (Z_v⊗Z_anc)"),
        ]
        plot_heatmap(mat3, x_lbl3, y_lbl,
                     f"Case 3 — CZ  (k={cz_k.k},  {n_t3} Tock{'s' if n_t3!=1 else ''},  {len(cz_k.batches)} batches)",
                     cmap2, legend3, ax3,
                     vline_every=2)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save_or_show(fig, save_path, "heatmap", display)


# Gantt charts  (improved: smart y-ticks, cleaner grid)

def plot_gantt_case1(scheduler: "SubstrateScheduler", ax: plt.Axes) -> None:
    """Gantt chart for Case 1 - rows sorted by spectral substrate position."""
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
    """Gantt chart for Case 2 - each CZ round = 2 consecutive Tocks."""
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


def plot_gantt_case3(cz_k: "CZSchedulerFixedAncilla", ax: plt.Axes) -> None:
    """Gantt chart for Case 3 - sub-batches of k CZ gates, 2 Tocks each."""
    G            = cz_k.G
    nodes_list   = list(G.nodes)
    node_to_row  = {nd: i for i, nd in enumerate(nodes_list)}
    n_q          = len(nodes_list)
    n_t          = cz_k.n_tocks()

    ax.set_xlim(0, n_t)
    ax.set_ylim(-0.5, n_q - 0.5)
    ax.set_facecolor(_FIG_BG)
    ax.set_xlabel("Tock  (logical time step)", fontsize=10)
    ax.set_ylabel("Qubit / ELU", fontsize=10)
    ax.set_title(f"Case 3 — CZ  (k={cz_k.k} ancilla(s),  {len(cz_k.batches)} sub-batches)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    x_lbl = [f"T{t+1}" for t in range(n_t)]
    y_lbl = [f"Q{nd}"  for nd in nodes_list]
    _smart_ticks(ax, n_t, "x", x_lbl, fontsize=8)
    _smart_ticks(ax, n_q, "y", y_lbl, fontsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
    ax.grid(axis="y", linestyle=":", alpha=0.18, zorder=0)

    for b, batch in enumerate(cz_k.batches):
        col   = _PALETTE[b % len(_PALETTE)]
        t1, t2 = 2 * b, 2 * b + 1
        ax.axvspan(t1, t2 + 1, alpha=0.07, color=col, zorder=0)
        if n_t <= 80:
            ax.text(t1 + 1.0, n_q - 0.3,
                    f"B{b+1}", ha="center", va="center",
                    fontsize=6, color=col, fontweight="bold", zorder=3)
        for u, v in batch:
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
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Side-by-side Gantt charts for both (or all three) cases."""
    n_q    = len(scheduler.G.nodes)
    height = _fig_height(n_q)
    if cz_k is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, height),
                                             facecolor=_FIG_BG)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, height),
                                       facecolor=_FIG_BG)
    fig.suptitle("Schedule Gantt Charts", fontsize=14, fontweight="bold")
    plot_gantt_case1(scheduler, ax1)
    plot_gantt_case2(cz, ax2)
    if cz_k is not None:
        plot_gantt_case3(cz_k, ax3)
    plt.tight_layout()
    _save_or_show(fig, save_path, "Gantt chart", display)


# Comparison dashboard  (bar charts: unchanged logic, cleaner styling)

def show_comparison_dashboard(
    report:    "ResourceReport",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Three-panel bar chart: tocks / peak ancilla ELUs / physical qubits.
    When the report includes Case 3 data, a third bar is added per panel.
    """
    has3 = report.tocks_case3 is not None

    if has3:
        cases  = ["Case 1\n(MPP)", "Case 2\n(CZ, ∞ anc)", f"Case 3\n(CZ, k={report.ancilla_k})"]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
    else:
        cases  = ["Case 1\n(MPP)", "Case 2\n(CZ)"]
        colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 3, figsize=(16 if has3 else 14, 5),
                             facecolor=_FIG_BG)
    fig.suptitle(
        f"Resource Comparison — {report.graph_type}(n={report.graph_nodes}), "
        f"d={report.code_distance}",
        fontsize=13, fontweight="bold",
    )

    def _panel(ax: plt.Axes, vals: list, title: str,
               ylabel: str, fmt: str = "{}") -> None:
        bar_w = 0.36 if has3 else 0.48
        bars = ax.bar(cases, vals, color=colors, edgecolor="white",
                      width=bar_w, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.set_ylim(0, max(vals) * 1.30)
        for bar, val in zip(bars, vals):
            lbl = fmt.format(val) if isinstance(val, int) else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    lbl, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    tock_vals = [report.tocks_case1, report.tocks_case2]
    anc_vals  = [report.peak_ancilla_ELUs_case1, report.peak_ancilla_ELUs_case2]
    phys_vals = [report.total_phys_qubits_case1, report.total_phys_qubits_case2]
    if has3:
        tock_vals.append(report.tocks_case3)
        anc_vals.append(report.peak_ancilla_ELUs_case3)
        phys_vals.append(report.total_phys_qubits_case3)

    _panel(axes[0], tock_vals,
           "Total Tocks\n(logical time steps)", "Tocks")
    _panel(axes[1], anc_vals,
           "Peak Ancilla ELUs\n(simultaneous demand)", "ELUs")
    _panel(axes[2], phys_vals,
           f"Total Physical Qubits\n(d={report.code_distance}, "
           f"{report.phys_per_ELU}/ELU)",
           "Physical qubits", fmt="{:,}")

    # Tock-ratio annotation on Case 2 bar
    if report.tock_ratio_c2_c1 < float("inf"):
        axes[0].annotate(
            f"×{report.tock_ratio_c2_c1:.1f}",
            xy=(1, report.tocks_case2),
            xytext=(0.55 if has3 else 0.62, report.tocks_case2 * 1.16),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
            fontsize=9, fontweight="bold", ha="center", color="#444",
        )
    # Tock-ratio annotation on Case 3 bar
    if has3 and report.tock_ratio_c3_c1 is not None and report.tock_ratio_c3_c1 < float("inf"):
        axes[0].annotate(
            f"×{report.tock_ratio_c3_c1:.1f}",
            xy=(2, report.tocks_case3),
            xytext=(1.65, report.tocks_case3 * 1.16),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
            fontsize=9, fontweight="bold", ha="center", color="#444",
        )

    plt.tight_layout()
    _save_or_show(fig, save_path, "dashboard", display)


# Shared save/show helper
# Qubit Occupancy  (per-tock tile strip, restored from original notebook)

def _draw_occupancy_tiles(
    ax:       plt.Axes,
    tock:     list,
    mapping:  dict,
    n_q:      int,
    tock_idx: int,
    palette:  list[str],
) -> None:
    """
    Draw the ancilla (top) and data (bottom) tile strips for one tock.
    Idle tiles are light grey; active tiles are coloured per measurement.
    """
    ax.set_xlim(-0.6, n_q)
    ax.set_ylim(-0.3, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Tock {tock_idx + 1} — Qubit Occupancy",
                 fontsize=11, fontweight="bold", pad=8)

    # Row labels
    ax.text(-0.3, 2.0, "Ancilla", ha="right", va="center",
            fontsize=10, color="#555", fontweight="bold")
    ax.text(-0.3, 0.5, "Data",    ha="right", va="center",
            fontsize=10, color="#555", fontweight="bold")

    # Idle tiles
    for q in range(n_q):
        for row_y, prefix in [(1.5, "A"), (0.0, "Q")]:
            ax.add_patch(plt.Rectangle(
                (q, row_y), 1, 1,
                facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8,
            ))
            ax.text(q + 0.5, row_y + 0.5, f"{prefix}{q}",
                    ha="center", va="center", fontsize=8, color="#bbbbbb")

    # Active tiles: one colour per measurement in this tock
    for bi, b in enumerate(tock):
        col      = palette[bi % len(palette)]
        L, R     = b["L"], b["R"]
        involved = [mapping[b["id"]]] + [mapping[nb] for nb in b["neighbors"]]

        for q in range(L, R + 1):  # ancilla span
            ax.add_patch(plt.Rectangle(
                (q, 1.5), 1, 1,
                facecolor=col, edgecolor="white", linewidth=1.5, alpha=0.88,
            ))
            ax.text(q + 0.5, 2.0, f"A{q}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

        for q in involved:  # data qubits
            ax.add_patch(plt.Rectangle(
                (q, 0.0), 1, 1,
                facecolor=col, edgecolor="white", linewidth=1.5, alpha=0.88,
            ))
            ax.text(q + 0.5, 0.5, f"Q{q}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")


def show_qubit_occupancy(
    scheduler: "SubstrateScheduler",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Per-tock qubit occupancy figure (Case 1 - MPP).

    Each row in the figure is one Tock.  Left panel = graph view with
    X/Z roles highlighted.  Right panel = the tile strip (ancilla row on top,
    data row on bottom) showing exactly which physical substrate positions are
    occupied.  Restored from the original notebook visualisation.
    """
    n_tocks = len(scheduler.schedule)
    if n_tocks == 0:
        print("  No tocks to display.")
        return

    n_q   = len(scheduler.G.nodes)
    # Scale: tile strip needs width ∝ n_q; cap the figure width sensibly
    strip_w = max(8, min(n_q * 0.55 + 2, 28))
    fig, axes = plt.subplots(
        n_tocks, 2,
        figsize=(strip_w + 7, 5.5 * n_tocks),
        facecolor=_FIG_BG,
        gridspec_kw={"width_ratios": [1, max(1, n_q // 6)]},
    )
    fig.suptitle("Qubit Occupancy per Tock  (Case 1 — MPP)",
                 fontsize=14, fontweight="bold", y=1.01)

    if n_tocks == 1:
        axes = [axes]  # normalise to list of rows

    pos = scheduler._get_pos()

    for t, (row, tock) in enumerate(zip(axes, scheduler.schedule)):
        ax_graph, ax_tiles = row

        # Graph view
        measured  = {b["id"] for b in tock}
        neighbors = {nb for b in tock for nb in b["neighbors"]}

        node_colors = [
            _COL_MEASURE if nd in measured  else
            "#FFD54F"    if nd in neighbors else
            _COL_IDLE
            for nd in scheduler.G.nodes
        ]
        active_edges = [
            (b["id"], nb) if scheduler.G.has_edge(b["id"], nb) else (nb, b["id"])
            for b in tock for nb in b["neighbors"]
        ]
        all_edges  = list(scheduler.G.edges())
        inactive   = [e for e in all_edges
                      if e not in active_edges
                      and (e[1], e[0]) not in active_edges]

        import networkx as nx
        nx.draw_networkx_edges(scheduler.G, pos, edgelist=inactive,
                               width=1.2, edge_color=_COL_EDGE_OFF, ax=ax_graph)
        nx.draw_networkx_edges(scheduler.G, pos, edgelist=active_edges,
                               width=2.0, edge_color=_COL_MEASURE, ax=ax_graph)
        nx.draw_networkx_nodes(scheduler.G, pos,
                               node_size=700, node_color=node_colors,
                               edgecolors=_COL_BORDER, linewidths=1.2,
                               ax=ax_graph)

        labels = {
            nd: (f"{nd}\nX" if nd in measured  else
                 f"{nd}\nZ" if nd in neighbors else str(nd))
            for nd in scheduler.G.nodes
        }
        nx.draw_networkx_labels(scheduler.G, pos, labels,
                                font_size=9, font_color="#222", ax=ax_graph)

        # operator string below the graph
        op_strs = []
        for b in tock:
            parts = [f"X{scheduler.mapping[b['id']]}"] + \
                    [f"Z{scheduler.mapping[nb]}" for nb in b["neighbors"]]
            op_strs.append(" ⊗ ".join(parts))
        ax_graph.text(0.5, 0.02, "  |  ".join(op_strs),
                      transform=ax_graph.transAxes,
                      ha="center", fontsize=8, family="monospace",
                      color="#666", zorder=3)

        ax_graph.legend(
            handles=[
                mpatches.Patch(facecolor=_COL_MEASURE, edgecolor=_COL_BORDER,
                               lw=1.2, label="X (measured)"),
                mpatches.Patch(facecolor="#FFD54F",    edgecolor=_COL_BORDER,
                               lw=1.2, label="Z (neighbour)"),
            ],
            fontsize=8, frameon=False, loc="upper right",
        )
        ax_graph.set_title(f"Tock {t + 1} — Graph View",
                           fontsize=11, fontweight="bold", pad=8)
        ax_graph.axis("off")
        ax_graph.margins(0.18)

        # Tile strip
        _draw_occupancy_tiles(ax_tiles, tock, scheduler.mapping,
                              n_q, t, _PALETTE)

    plt.tight_layout()
    _save_or_show(fig, save_path, "qubit occupancy", display)


def show_qubit_occupancy_case3(
    cz_k:      "CZSchedulerFixedAncilla",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Per-tock qubit occupancy figure for Case 3 (CZ + k ancillas).

    Each row is one Tock.  Left panel = graph view.  Right panel = tile strip.

    Tile strip layout (n_q columns, one per data qubit):
      Top row    - ancilla ELU: drawn ONLY above the data qubit it is docked
                   with this tock.  Idle slots show a faint dashed outline.
      Bottom row - data qubits: always present; highlighted when active.
      Connector  - a short vertical line links each active (ancilla, qubit) pair
                   so it is visually clear which ancilla is touching which qubit.

    This makes the ancilla "repositioning" between Step 1 and Step 2 directly
    visible: in Step 1 ancilla Aᵢ sits above Qᵤ; in Step 2 it moves to Qᵥ.
    """
    G        = cz_k.G
    n_q      = len(G.nodes)
    n_tocks  = cz_k.n_tocks()
    if n_tocks == 0:
        print("  No tocks to display.")
        return

    nodes_list  = list(G.nodes)
    node_to_col = {nd: i for i, nd in enumerate(nodes_list)}

    palette = _PALETTE
    k       = cz_k.k

    strip_w = max(8, min(n_q * 0.60 + 3, 30))
    fig, axes = plt.subplots(
        n_tocks, 2,
        figsize=(strip_w + 7, 5.5 * n_tocks),
        facecolor=_FIG_BG,
        gridspec_kw={"width_ratios": [1, max(1, n_q // 6)]},
    )
    fig.suptitle(
        f"Qubit Occupancy per Tock  (Case 3 — CZ, k={k} ancilla(s))",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if n_tocks == 1:
        axes = [axes]

    pos = cz_k._get_pos()

    import networkx as nx_local

    for t in range(n_tocks):
        ax_graph, ax_tiles = axes[t]
        batch_idx = t // 2
        step      = t % 2  # 0 = Step 1  (measure Z_u⊗Z_anc)
                              # 1 = Step 2  (measure Z_v⊗Z_anc)
        batch     = cz_k.batches[batch_idx]
        col       = palette[batch_idx % len(palette)]

        active_step1 = {u for u, v in batch}
        active_step2 = {v for u, v in batch}
        active_now   = active_step1 if step == 0 else active_step2
        active_other = active_step2 if step == 0 else active_step1

        # Graph view
        node_colors = [
            "#FFCC80" if nd in active_now   else
            "#FFECB3" if nd in active_other else
            _COL_IDLE
            for nd in G.nodes
        ]
        batch_edge_set = {(min(u, v), max(u, v)) for u, v in batch}
        active_edges   = [(u, v) for u, v in G.edges()
                          if (min(u,v), max(u,v)) in batch_edge_set]
        inactive_edges = [(u, v) for u, v in G.edges()
                          if (min(u,v), max(u,v)) not in batch_edge_set]

        nx_local.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                                     width=1.2, edge_color=_COL_EDGE_OFF, ax=ax_graph)
        nx_local.draw_networkx_edges(G, pos, edgelist=active_edges,
                                     width=2.5, edge_color=col, ax=ax_graph)
        nx_local.draw_networkx_nodes(G, pos,
                                     node_size=700, node_color=node_colors,
                                     edgecolors=_COL_BORDER, linewidths=1.2,
                                     ax=ax_graph)
        step_lbl = "Z_u⊗Z_anc" if step == 0 else "Z_v⊗Z_anc"
        labels = {
            nd: (f"{nd}\n{step_lbl}" if nd in active_now else str(nd))
            for nd in G.nodes
        }
        nx_local.draw_networkx_labels(G, pos, labels,
                                      font_size=8, font_color="#222", ax=ax_graph)

        op_parts = [
            f"Z_{u if step==0 else v}⊗Z_anc{i}"
            for i, (u, v) in enumerate(batch)
        ]
        ax_graph.text(0.5, 0.02, "  |  ".join(op_parts),
                      transform=ax_graph.transAxes,
                      ha="center", fontsize=8, family="monospace",
                      color="#666", zorder=3)

        ax_graph.legend(
            handles=[
                mpatches.Patch(facecolor="#FFCC80", edgecolor=_COL_BORDER,
                               lw=1.2, label="Active this tock"),
                mpatches.Patch(facecolor="#FFECB3", edgecolor=_COL_BORDER,
                               lw=1.2, label="Other CZ endpoint"),
            ],
            fontsize=8, frameon=False, loc="upper right",
        )
        step_name = "Step 1 (Z_u⊗Z_anc)" if step == 0 else "Step 2 (Z_v⊗Z_anc)"
        ax_graph.set_title(f"Tock {t+1} — Sub-batch {batch_idx+1}, {step_name}",
                           fontsize=11, fontweight="bold", pad=8)
        ax_graph.axis("off")
        ax_graph.margins(0.18)

        # Tile strip
        # n_q columns: one per data qubit.
        # Top row  (y=1.5..2.5): ancilla ELU docked HERE if active this tock.
        # Bottom row (y=0..1)  : data qubit tile (always present).
        # Active pair shares a color; a connector line links them.
        # Idle ancilla slots above inactive qubits show a faint dashed border.

        # Build: node → (ancilla_index, tile_color)
        active_map: dict = {}
        for i, (u, v) in enumerate(batch):
            qubit = u if step == 0 else v
            active_map[qubit] = (i, palette[i % len(palette)])

        n_active = len(active_map)

        ax_tiles.set_xlim(-0.6, n_q + 0.2)
        ax_tiles.set_ylim(-0.3, 2.8)
        ax_tiles.set_aspect("equal")
        ax_tiles.axis("off")
        ax_tiles.set_title(
            f"Tock {t+1} — Tile Occupancy  "
            f"({n_active} / {k} ancilla ELU(s) docked)",
            fontsize=10, fontweight="bold", pad=8,
        )

        ax_tiles.text(-0.3, 2.0, "Ancilla", ha="right", va="center",
                      fontsize=10, color="#555", fontweight="bold")
        ax_tiles.text(-0.3, 0.5, "Data",    ha="right", va="center",
                      fontsize=10, color="#555", fontweight="bold")

        for qi, nd in enumerate(nodes_list):
            if nd in active_map:
                ai, tile_col = active_map[nd]

                # Data qubit tile: active (colored)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor=tile_col, edgecolor="white",
                    linewidth=1.5, alpha=0.88, zorder=2,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white", zorder=3)

                # Ancilla tile: docked directly above (same column, same color)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 1.5), 1, 1,
                    facecolor=tile_col, edgecolor="white",
                    linewidth=1.5, alpha=0.88, zorder=2,
                ))
                ax_tiles.text(qi + 0.5, 2.0, f"A{ai}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white", zorder=3)

                # Connector: short vertical line linking the two tiles
                ax_tiles.plot([qi + 0.5, qi + 0.5], [1.0, 1.5],
                              color=tile_col, linewidth=2.5, alpha=0.75,
                              solid_capstyle="round", zorder=1)

            else:
                # Data qubit tile: idle (grey)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center", fontsize=7, color="#bbbbbb")

                # Ancilla slot above: faint dashed outline (no ancilla here)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 1.5), 1, 1,
                    facecolor="#f8f8f8", edgecolor="#e0e0e0",
                    linewidth=0.6, linestyle="--",
                ))

    plt.tight_layout()
    _save_or_show(fig, save_path, "Case 3 qubit occupancy", display)


def _save_or_show(
    fig:       plt.Figure,
    save_path: str | None,
    label:     str,
    display:   bool = False,
) -> None:
    """Save to file and/or display interactively.

    display=True  → always call plt.show(), even when a save_path is given.
    display=False → only show when save_path is None (original behaviour).
    """
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  {label.capitalize()} saved → {save_path}")
    if not save_path or display:
        plt.show()
    plt.close(fig)
