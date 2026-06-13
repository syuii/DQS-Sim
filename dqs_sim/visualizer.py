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
    from dqs_sim.cost import ResourceReport
    from dqs_sim.schedulers import (
        CZScheduler, CZSchedulerFixedAncilla, CZSchedulerLimited,
        MPPSchedulerLimited, SubstrateScheduler,
        ScheduledBatch, ScheduledGroup,
    )


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
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Two-panel figure showing ELUs as physical surface-code patches.

    Left  — Case 1: MIS nodes (green) vs to-measure (pink).
             Edges coloured by the Tock in which they are activated.
    Right — Case 2: Edges coloured by CZ round.

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

    # ── Case 3 panel (optional) ──────────────────────────────────────────────
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
                 vline_every=2)   # divider every CZ round

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


def plot_gantt_case3(cz_k: "CZSchedulerFixedAncilla", ax: plt.Axes) -> None:
    """Gantt chart for Case 3 — sub-batches of k CZ gates, 2 Tocks each."""
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


# ============================================================================
# Comparison dashboard  (bar charts — unchanged logic, cleaner styling)
# ============================================================================

def show_comparison_dashboard(
    report:    "ResourceReport",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Bar-chart resource comparison across all available cases.

    When Cases 4 & 5 are present (k-port switch), additional bars are shown.
    No SWAP overhead — the k-port switch is reconfigurable; the constraint
    is parallelism (how many operations fit in k ports per Tock).
    """
    has3 = report.tocks_case3 is not None
    has4 = report.tocks_case4 is not None
    has5 = report.tocks_case5 is not None
    has_limited = has4 or has5

    # ── Build case labels and colours ────────────────────────────────────
    cases  = ["Case 1\n(MPP)", "Case 2\n(CZ)"]
    colors = ["#4C72B0", "#DD8452"]
    if has3:
        cases.append(f"Case 3\n(CZ, k={report.ancilla_k})")
        colors.append("#55A868")
    if has4:
        k_str = report.switch_k or "?"
        cases.append(f"Case 4\n(MPP, k={k_str})")
        colors.append("#C44E52")
    if has5:
        k_str = report.switch_k or "?"
        cases.append(f"Case 5\n(CZ, k={k_str})")
        colors.append("#8172B2")

    n_cases = len(cases)

    # ── Figure layout ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(max(14, n_cases * 3 + 4), 5),
                             facecolor=_FIG_BG)

    subtitle = f"{report.graph_type}(n={report.graph_nodes}), d={report.code_distance}"
    if has_limited:
        subtitle += f", switch k={report.switch_k}"
    fig.suptitle(
        f"Resource Comparison — {subtitle}",
        fontsize=13, fontweight="bold",
    )

    def _panel(ax: plt.Axes, vals: list, case_labels: list,
               case_colors: list, title: str,
               ylabel: str, fmt: str = "{}") -> None:
        n = len(vals)
        bar_w = max(0.28, min(0.55, 2.0 / n))
        x = np.arange(n)

        bars = ax.bar(x, vals, color=case_colors, edgecolor="white",
                      width=bar_w, zorder=2)
        for bar, val in zip(bars, vals):
            lbl = fmt.format(val) if isinstance(val, int) else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    lbl, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.set_ylim(0, max(vals) * 1.35)

    # ── Tock values ──────────────────────────────────────────────────────
    tock_vals = [report.tocks_case1, report.tocks_case2]
    if has3:
        tock_vals.append(report.tocks_case3)
    if has4:
        tock_vals.append(report.tocks_case4)
    if has5:
        tock_vals.append(report.tocks_case5)

    # ── Ancilla values ───────────────────────────────────────────────────
    anc_vals = [report.peak_ancilla_ELUs_case1, report.peak_ancilla_ELUs_case2]
    if has3:
        anc_vals.append(report.peak_ancilla_ELUs_case3)
    if has4:
        anc_vals.append(report.peak_ancilla_ELUs_case1)  # same ELU model
    if has5:
        anc_vals.append(report.peak_ancilla_ELUs_case2)

    # ── Physical qubit values ────────────────────────────────────────────
    phys_vals = [report.total_phys_qubits_case1, report.total_phys_qubits_case2]
    if has3:
        phys_vals.append(report.total_phys_qubits_case3)
    if has4:
        phys_vals.append(report.total_phys_qubits_case1)
    if has5:
        phys_vals.append(report.total_phys_qubits_case2)

    # ── Draw panels ──────────────────────────────────────────────────────
    _panel(axes[0], tock_vals, cases, colors,
           "Total Tocks\n(logical time steps)", "Tocks")
    _panel(axes[1], anc_vals, cases, colors,
           "Peak Ancilla ELUs\n(simultaneous demand)", "ELUs")
    _panel(axes[2], phys_vals, cases, colors,
           f"Total Physical Qubits\n(d={report.code_distance}, "
           f"{report.phys_per_ELU}/ELU)",
           "Physical qubits", fmt="{:,}")

    plt.tight_layout()
    _save_or_show(fig, save_path, "dashboard", display)


# ============================================================================
# Shared save/show helper
# ============================================================================
# Qubit Occupancy  (per-tock tile strip, restored from original notebook)
# ============================================================================

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

    # Active tiles — one colour per measurement in this tock
    for bi, b in enumerate(tock):
        col      = palette[bi % len(palette)]
        L, R     = b["L"], b["R"]
        involved = [mapping[b["id"]]] + [mapping[nb] for nb in b["neighbors"]]

        for q in range(L, R + 1):          # ancilla span
            ax.add_patch(plt.Rectangle(
                (q, 1.5), 1, 1,
                facecolor=col, edgecolor="white", linewidth=1.5, alpha=0.88,
            ))
            ax.text(q + 0.5, 2.0, f"A{q}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

        for q in involved:                 # data qubits
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
    Per-tock qubit occupancy figure (Case 1 — MPP).

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
        axes = [axes]   # normalise to list of rows

    pos = scheduler._get_pos()

    for t, (row, tock) in enumerate(zip(axes, scheduler.schedule)):
        ax_graph, ax_tiles = row

        # ── Graph view ───────────────────────────────────────────────────────
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

        # ── Tile strip ───────────────────────────────────────────────────────
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
      Top row    — ancilla ELU: drawn ONLY above the data qubit it is docked
                   with this tock.  Idle slots show a faint dashed outline.
      Bottom row — data qubits: always present; highlighted when active.
      Connector  — a short vertical line links each active (ancilla, qubit) pair
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
        step      = t % 2    # 0 = Step 1  (measure Z_u⊗Z_anc)
                              # 1 = Step 2  (measure Z_v⊗Z_anc)
        batch     = cz_k.batches[batch_idx]
        col       = palette[batch_idx % len(palette)]

        active_step1 = {u for u, v in batch}
        active_step2 = {v for u, v in batch}
        active_now   = active_step1 if step == 0 else active_step2
        active_other = active_step2 if step == 0 else active_step1

        # ── Graph view ───────────────────────────────────────────────────────
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

        # ── Tile strip ───────────────────────────────────────────────────────
        # n_q columns — one per data qubit.
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

                # Data qubit tile — active (colored)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor=tile_col, edgecolor="white",
                    linewidth=1.5, alpha=0.88, zorder=2,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white", zorder=3)

                # Ancilla tile — docked directly above (same column, same color)
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
                # Data qubit tile — idle (grey)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center", fontsize=7, color="#bbbbbb")

                # Ancilla slot above — faint dashed outline (no ancilla here)
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 1.5), 1, 1,
                    facecolor="#f8f8f8", edgecolor="#e0e0e0",
                    linewidth=0.6, linestyle="--",
                ))

    plt.tight_layout()
    _save_or_show(fig, save_path, "Case 3 qubit occupancy", display)


# ============================================================================

def show_qubit_occupancy_case2(
    cz:        "CZScheduler",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    Per-tock qubit occupancy figure for Case 2 (CZ gates, unlimited ancillas).

    Layout is identical to Case 3: n_q columns, one per data qubit.
      Top row    — ancilla ELU drawn above the data qubit it is interacting with.
      Bottom row — data qubit tile, always present; highlighted when active.
      Connector  — vertical line linking each active (ancilla, qubit) pair.

    Since ancillas are unlimited in Case 2, every CZ gate in a round runs in
    parallel — the ancilla count per tock equals the number of edges in that
    colour class.
    """
    G       = cz.G
    n_q     = len(G.nodes)
    n_rounds = cz.chromatic_index
    n_tocks  = 2 * n_rounds
    if n_tocks == 0:
        print("  No tocks to display.")
        return

    nodes_list  = list(G.nodes)
    palette     = _PALETTE

    strip_w = max(8, min(n_q * 0.60 + 3, 30))
    fig, axes = plt.subplots(
        n_tocks, 2,
        figsize=(strip_w + 7, 5.5 * n_tocks),
        facecolor=_FIG_BG,
        gridspec_kw={"width_ratios": [1, max(1, n_q // 6)]},
    )
    fig.suptitle(
        "Qubit Occupancy per Tock  (Case 2 — CZ, unlimited ancillas)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if n_tocks == 1:
        axes = [axes]

    pos = cz._get_pos()

    import networkx as nx_local

    for t in range(n_tocks):
        ax_graph, ax_tiles = axes[t]
        round_idx = t // 2
        step      = t % 2          # 0 = Step 1 (Z_u⊗Z_anc), 1 = Step 2 (Z_v⊗Z_anc)
        round_edges = cz.cz_rounds[round_idx]
        round_col   = palette[round_idx % len(palette)]

        active_step1 = {u for u, v in round_edges}
        active_step2 = {v for u, v in round_edges}
        active_now   = active_step1 if step == 0 else active_step2
        active_other = active_step2 if step == 0 else active_step1

        # ── Graph view ───────────────────────────────────────────────────────
        node_colors = [
            "#FFCC80" if nd in active_now   else
            "#FFECB3" if nd in active_other else
            _COL_IDLE
            for nd in G.nodes
        ]
        round_edge_set = {(min(u, v), max(u, v)) for u, v in round_edges}
        active_edges   = [(u, v) for u, v in G.edges()
                          if (min(u, v), max(u, v)) in round_edge_set]
        inactive_edges = [(u, v) for u, v in G.edges()
                          if (min(u, v), max(u, v)) not in round_edge_set]

        nx_local.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                                     width=1.2, edge_color=_COL_EDGE_OFF, ax=ax_graph)
        nx_local.draw_networkx_edges(G, pos, edgelist=active_edges,
                                     width=2.5, edge_color=round_col, ax=ax_graph)
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
            f"Z_{u if step == 0 else v}⊗Z_anc{i}"
            for i, (u, v) in enumerate(round_edges)
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
        ax_graph.set_title(
            f"Tock {t + 1} — CZ Round {round_idx + 1}, {step_name}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax_graph.axis("off")
        ax_graph.margins(0.18)

        # ── Tile strip ───────────────────────────────────────────────────────
        # Build: node → (ancilla_index, tile_color) for qubits active this tock
        active_map: dict = {}
        for i, (u, v) in enumerate(round_edges):
            qubit = u if step == 0 else v
            active_map[qubit] = (i, palette[i % len(palette)])

        n_active = len(active_map)

        ax_tiles.set_xlim(-0.6, n_q + 0.2)
        ax_tiles.set_ylim(-0.3, 2.8)
        ax_tiles.set_aspect("equal")
        ax_tiles.axis("off")
        ax_tiles.set_title(
            f"Tock {t + 1} — Tile Occupancy  "
            f"({n_active} ancilla ELU(s) docked)",
            fontsize=10, fontweight="bold", pad=8,
        )

        ax_tiles.text(-0.3, 2.0, "Ancilla", ha="right", va="center",
                      fontsize=10, color="#555", fontweight="bold")
        ax_tiles.text(-0.3, 0.5, "Data",    ha="right", va="center",
                      fontsize=10, color="#555", fontweight="bold")

        for qi, nd in enumerate(nodes_list):
            if nd in active_map:
                ai, tile_col = active_map[nd]

                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor=tile_col, edgecolor="white",
                    linewidth=1.5, alpha=0.88, zorder=2,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white", zorder=3)

                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 1.5), 1, 1,
                    facecolor=tile_col, edgecolor="white",
                    linewidth=1.5, alpha=0.88, zorder=2,
                ))
                ax_tiles.text(qi + 0.5, 2.0, f"A{ai}",
                              ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white", zorder=3)

                ax_tiles.plot([qi + 0.5, qi + 0.5], [1.0, 1.5],
                              color=tile_col, linewidth=2.5, alpha=0.75,
                              solid_capstyle="round", zorder=1)
            else:
                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 0.0), 1, 1,
                    facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8,
                ))
                ax_tiles.text(qi + 0.5, 0.5, f"Q{nd}",
                              ha="center", va="center", fontsize=7, color="#bbbbbb")

                ax_tiles.add_patch(plt.Rectangle(
                    (qi, 1.5), 1, 1,
                    facecolor="#f8f8f8", edgecolor="#e0e0e0",
                    linewidth=0.6, linestyle="--",
                ))

    plt.tight_layout()
    _save_or_show(fig, save_path, "Case 2 qubit occupancy", display)


# ============================================================================
# Port Utilization Chart (Cases 4 & 5 — k-port switch)
# ============================================================================

def show_port_utilization(
    mpp_lim: "MPPSchedulerLimited | None" = None,
    cz_lim:  "CZSchedulerLimited | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Bar chart showing switch port utilization per group/batch.

    Left panel  — Case 4 (MPP): each bar = one scheduled group
    Right panel — Case 5 (CZ):  each bar = one sub-batch
    Bar height = ports used; dashed line = k (max ports available).
    """
    panels = []
    if mpp_lim is not None and mpp_lim.scheduled_groups:
        panels.append(("Case 4 — MPP (k-port)", mpp_lim.scheduled_groups,
                        mpp_lim.switch.k, "Group", "mpp"))
    if cz_lim is not None and cz_lim.scheduled_batches:
        panels.append(("Case 5 — CZ (k-port)", cz_lim.scheduled_batches,
                        cz_lim.switch.k, "Sub-batch", "cz"))
    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5),
                             facecolor=_FIG_BG, squeeze=False)
    fig.suptitle("Switch Port Utilization per Tock Group / Sub-batch",
                 fontsize=14, fontweight="bold")

    for idx, (title, schedule, k, xlabel, mode) in enumerate(panels):
        ax = axes[0][idx]
        n = len(schedule)
        x = np.arange(n)

        ports_used = [item.n_ports_used for item in schedule]
        util_pct = [p / k * 100 for p in ports_used]

        bars = ax.bar(x, ports_used, color="#4C72B0" if mode == "cz" else "#55A868",
                      edgecolor="white", zorder=2, alpha=0.85)
        ax.axhline(k, color="#C44E52", linewidth=2, linestyle="--",
                   label=f"k = {k} (max ports)", zorder=3)

        # Label each bar with port count
        for bar, pu, pct in zip(bars, ports_used, util_pct):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{pu}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Ports Used", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        _smart_ticks(ax, n, "x", [str(i + 1) for i in range(n)], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper right")
        ax.set_ylim(0, k * 1.3)

        # Avg utilization annotation
        avg_util = sum(util_pct) / len(util_pct) if util_pct else 0
        ax.text(0.02, 0.95,
                f"Avg utilization: {avg_util:.0f}%  "
                f"({n} {'groups' if mode == 'mpp' else 'batches'}, "
                f"{sum(s.tocks for s in schedule)} Tocks)",
                transform=ax.transAxes, fontsize=9, va="top",
                color="#444", fontstyle="italic")

    plt.tight_layout()
    _save_or_show(fig, save_path, "port utilization chart", display)


# ============================================================================
# Heatmap for Cases 4 & 5 (k-port switch)
# ============================================================================

def _build_heatmap_case5_kport(cz_lim: "CZSchedulerLimited") -> tuple[np.ndarray, list[str]]:
    """Matrix[qubit_row, tock] for Case 5 k-port model.

    0 = idle, 1 = CZ step 1, 2 = CZ step 2.
    Each sub-batch = 2 Tocks (step 1 + step 2).
    """
    G = cz_lim.G
    nds = list(G.nodes)
    ntr = {n: i for i, n in enumerate(nds)}
    total_tocks = cz_lim.n_tocks()
    m = np.zeros((len(nds), total_tocks), dtype=np.int8)
    x_labels = []

    for bi, sb in enumerate(cz_lim.scheduled_batches):
        t1 = 2 * bi
        t2 = 2 * bi + 1
        for u, v in sb.edges:
            m[ntr[u], t1] = 1
            m[ntr[v], t2] = 2
        x_labels.extend([f"B{bi+1}.1", f"B{bi+1}.2"])

    return m, x_labels


def _build_heatmap_case4_kport(mpp_lim: "MPPSchedulerLimited") -> tuple[np.ndarray, list[str]]:
    """Matrix[qubit_row, tock] for Case 4 k-port model.

    0 = idle, 1 = MPP measured (X), 2 = MPP neighbour (Z).
    Each group = 1 Tock.
    """
    G = mpp_lim.G
    nds = list(G.nodes)
    ntr = {n: i for i, n in enumerate(nds)}
    total_tocks = mpp_lim.n_tocks()
    m = np.zeros((len(nds), total_tocks), dtype=np.int8)
    x_labels = []

    for gi, sg in enumerate(mpp_lim.scheduled_groups):
        for meas in sg.measurements:
            m[ntr[meas["id"]], gi] = 2  # X-measured
            for nb in meas["neighbors"]:
                if m[ntr[nb], gi] == 0:
                    m[ntr[nb], gi] = 1  # Z-neighbour
        x_labels.append(f"G{gi+1}")

    return m, x_labels


def show_heatmaps_limited(
    mpp_lim: "MPPSchedulerLimited | None" = None,
    cz_lim:  "CZSchedulerLimited | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Schedule heatmaps for Cases 4 and/or 5 (k-port switch).

    Rows = logical qubits.
    Columns = Tocks.
    Colour = role: idle / active.
    """
    panels = []
    if mpp_lim is not None and mpp_lim.scheduled_groups:
        mat4, xlbl4 = _build_heatmap_case4_kport(mpp_lim)
        panels.append((mat4, xlbl4, "Case 4 — MPP (k-port switch)", "c4"))
    if cz_lim is not None and cz_lim.scheduled_batches:
        mat5, xlbl5 = _build_heatmap_case5_kport(cz_lim)
        panels.append((mat5, xlbl5, "Case 5 — CZ (k-port switch)", "c5"))
    if not panels:
        return

    n_panels = len(panels)
    n_q = max(m.shape[0] for m, _, _, _ in panels)
    height = max(5, min(n_q * 0.25 + 3, 18))
    total_cols = sum(m.shape[1] for m, _, _, _ in panels)
    fig_w = max(12, total_cols * 0.4 + 6)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, height),
                             facecolor=_FIG_BG, squeeze=False)

    k_str = ""
    if mpp_lim is not None:
        k_str = f"k={mpp_lim.switch.k}"
    elif cz_lim is not None:
        k_str = f"k={cz_lim.switch.k}"
    fig.suptitle(f"Schedule Heatmaps — k-Port Switch ({k_str})",
                 fontsize=14, fontweight="bold")

    # Case 4: white | light blue (Z-neighbour) | deep blue (X-measured)
    cmap4 = ListedColormap(["#F4F4F4", "#90CAF9", "#1565C0"])
    legend4 = [
        mpatches.Patch(facecolor="#F4F4F4", edgecolor="#ccc", label="Idle"),
        mpatches.Patch(facecolor="#90CAF9", label="Z-neighbour"),
        mpatches.Patch(facecolor="#1565C0", label="X-measured"),
    ]
    # Case 5: white | amber (CZ step 1) | deep orange (CZ step 2)
    cmap5 = ListedColormap(["#F4F4F4", "#FFCC80", "#E65100"])
    legend5 = [
        mpatches.Patch(facecolor="#F4F4F4", edgecolor="#ccc", label="Idle"),
        mpatches.Patch(facecolor="#FFCC80", label="CZ Step 1 (Z_u)"),
        mpatches.Patch(facecolor="#E65100", label="CZ Step 2 (Z_v)"),
    ]

    for idx, (mat, xlbl, title, mode) in enumerate(panels):
        ax = axes[0][idx]
        cmap = cmap4 if mode == "c4" else cmap5
        legend = legend4 if mode == "c4" else legend5
        sched = mpp_lim if mode == "c4" else cz_lim
        nds = list(sched.G.nodes)
        y_lbl = [f"Q{nd}" for nd in nds]

        plot_heatmap(mat, xlbl, y_lbl, title, cmap, legend, ax,
                     vline_every=2 if mode == "c5" else 0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save_or_show(fig, save_path, "k-port heatmap", display)


# ============================================================================
# Per-Tock Occupancy — Case 4 (MPP + k-port switch)
# ============================================================================

def _draw_switch_diagram(
    ax: plt.Axes,
    *,
    # Data ELU info
    nodes_list: list,
    n_q: int,
    data_active: set,       # nodes actively connected this tock
    data_other: set,        # nodes involved but not active this tock (Case 5 step)
    data_measured: set,     # X-measured nodes (Case 4) — empty for Case 5
    data_neighbor: set,     # Z-neighbour nodes (Case 4) — empty for Case 5
    # Switch info
    k: int,
    n_sw: int,
    sw_data_nodes: list[list],     # per-switch: which data nodes connect
    sw_anc_count: list[int],       # per-switch: how many ancillas connect
    # Title
    title: str,
    is_case4: bool = False,
) -> None:
    """
    Shared switch diagram renderer for Cases 4 and 5.

    Layout: Data ELUs (top) → Switch(es) side-by-side (middle, single row
    of k ports each) → Ancilla ELUs (bottom).

    Each switch has k generic ports. Data and ancilla ELUs connect to any
    port — they are NOT pre-split into data/ancilla rows.
    """
    _SW_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
                  "#00BCD4", "#E91E63", "#795548", "#607D8B"]
    if not is_case4:
        _SW_COLORS = ["#E65100", "#1565C0", "#2E7D32", "#6A1B9A",
                      "#00838F", "#C62828", "#4E342E", "#37474F"]

    port_w = 0.5
    port_sp = 0.7     # port spacing
    sw_box_w = k * port_sp + 0.4
    sw_gap = 1.5
    total_sw_w = n_sw * sw_box_w + (n_sw - 1) * sw_gap

    max_x = max(n_q + 1, total_sw_w + 1)

    # Y coordinates
    data_y = 4.0
    sw_y = 1.8         # single port row centre
    anc_y = -0.5

    ax.set_xlim(-1, max_x + 0.5)
    ax.set_ylim(anc_y - 1.2, data_y + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # ── DATA ELUs (top row) — all n_q shown ─────────────────────────────
    data_x_start = (max_x - n_q) / 2
    data_positions = {}
    for qi, nd in enumerate(nodes_list):
        x = data_x_start + qi
        if is_case4:
            if nd in data_measured:
                fc, ec = _COL_MEASURE, "#E53935"
            elif nd in data_neighbor:
                fc, ec = "#FFD54F", "#F9A825"
            else:
                fc, ec = "#F5F5F5", "#E0E0E0"
            tc = "white" if nd in data_measured else (
                 "#333" if nd in data_neighbor else "#CCC")
            role = "X" if nd in data_measured else (
                   "Z" if nd in data_neighbor else "")
            lbl = f"Q{nd}" + (f"\n{role}" if role else "")
        else:
            is_active = nd in data_active
            is_other = nd in data_other
            if is_active:
                fc, ec = "#FFCC80", "#F57C00"
            elif is_other:
                fc, ec = "#FFECB3", "#FFB74D"
            else:
                fc, ec = "#F5F5F5", "#E0E0E0"
            tc = "#333" if is_active else ("#888" if is_other else "#CCC")
            lbl = f"Q{nd}"
        ax.add_patch(plt.Rectangle(
            (x, data_y - 0.35), 0.8, 0.7,
            facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
        ))
        ax.text(x + 0.4, data_y, lbl, ha="center", va="center",
                fontsize=7, fontweight="bold", color=tc, zorder=3)
        if nd in data_active:
            data_positions[nd] = x + 0.4

    ax.text(data_x_start + n_q / 2, data_y + 0.6, "Data ELUs",
            ha="center", fontsize=8, color="#666", fontweight="bold")

    # ── ANCILLA ELUs (bottom row) — show only active ones ────────────────
    total_anc = sum(sw_anc_count)
    anc_x_start = (max_x - total_anc) / 2 if total_anc > 0 else max_x / 2
    anc_positions = {}
    anc_sw_map = {}   # anc_idx → switch_idx
    ai = 0
    for si in range(n_sw):
        for _ in range(sw_anc_count[si]):
            anc_sw_map[ai] = si
            ai += 1

    for ai_idx in range(total_anc):
        x = anc_x_start + ai_idx
        sw_c = _SW_COLORS[anc_sw_map[ai_idx] % len(_SW_COLORS)]
        fc = "#A5D6A7" if is_case4 else "#FFB74D"
        ec = sw_c
        ax.add_patch(plt.Rectangle(
            (x, anc_y - 0.35), 0.8, 0.7,
            facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
        ))
        tc = "#1B5E20" if is_case4 else "#333"
        ax.text(x + 0.4, anc_y, f"A{ai_idx}", ha="center", va="center",
                fontsize=7, fontweight="bold", color=tc, zorder=3)
        anc_positions[ai_idx] = x + 0.4

    if total_anc > 0:
        ax.text(anc_x_start + total_anc / 2, anc_y - 0.7, "Ancilla ELUs",
                ha="center", fontsize=8,
                color="#2E7D32" if is_case4 else "#E65100", fontweight="bold")

    # ── SWITCH boxes (middle, side-by-side, single row of k ports) ──────
    sw_row_x_start = (max_x - total_sw_w) / 2

    for si in range(n_sw):
        sw_color = _SW_COLORS[si % len(_SW_COLORS)]
        sw_color_bg = sw_color + "15"

        box_x = sw_row_x_start + si * (sw_box_w + sw_gap)
        sw_box_y0 = sw_y - 0.5
        sw_box_h = 1.0

        # Switch box background
        ax.add_patch(mpatches.FancyBboxPatch(
            (box_x, sw_box_y0), sw_box_w, sw_box_h,
            boxstyle="round,pad=0.1",
            facecolor=sw_color_bg, edgecolor=sw_color,
            linewidth=2, zorder=1,
        ))

        n_data_sw = len(sw_data_nodes[si])
        n_anc_sw = sw_anc_count[si]
        ports_used = n_data_sw + n_anc_sw
        label_text = (f"SWITCH {si+1}  (k={k}, {ports_used}/{k} ports)"
                      if n_sw > 1
                      else f"k-PORT SWITCH  (k={k}, {ports_used}/{k} ports)")
        ax.text(box_x + sw_box_w / 2, sw_box_y0 - 0.3, label_text,
                ha="center", fontsize=7, fontweight="bold", color=sw_color)

        # Draw k ports in a single row: first n_data_sw are data, next n_anc_sw are ancilla
        for pi in range(k):
            px = box_x + 0.2 + pi * port_sp
            if pi < n_data_sw:
                # Data port
                fc = sw_color + "55"
                ec = sw_color
                plbl = "D"
            elif pi < n_data_sw + n_anc_sw:
                # Ancilla port
                fc = "#A5D6A7" + "88" if is_case4 else "#FFB74D88"
                ec = sw_color
                plbl = "A"
            else:
                # Empty port
                fc = "#F5F5F5"
                ec = "#DDD"
                plbl = ""
            ax.add_patch(plt.Rectangle(
                (px, sw_y - 0.2), port_w, 0.4,
                facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2,
            ))
            if plbl:
                ax.text(px + port_w / 2, sw_y, plbl, ha="center", va="center",
                        fontsize=6, color=sw_color, fontweight="bold", zorder=3)

        # Lines: data ELU → data port
        data_sorted = sorted(sw_data_nodes[si])
        for pi, nd in enumerate(data_sorted):
            if nd not in data_positions:
                continue
            ex = data_positions[nd]
            px = box_x + 0.2 + pi * port_sp + port_w / 2
            lc = "#E53935" if nd in data_measured else (
                 "#F9A825" if nd in data_neighbor else "#F57C00")
            ax.plot([ex, px], [data_y - 0.35, sw_y + 0.2],
                    color=lc, linewidth=1.5, alpha=0.6, zorder=0)

        # Lines: ancilla port → ancilla ELU
        anc_gi_start = sum(sw_anc_count[:si])
        for ai_off in range(n_anc_sw):
            anc_gi = anc_gi_start + ai_off
            if anc_gi not in anc_positions:
                continue
            ax_pos = anc_positions[anc_gi]
            px = box_x + 0.2 + (n_data_sw + ai_off) * port_sp + port_w / 2
            ax.plot([px, ax_pos], [sw_y - 0.2, anc_y + 0.35],
                    color=sw_color, linewidth=1.5, alpha=0.5,
                    linestyle="--", zorder=0)


def show_qubit_occupancy_case4(
    mpp_lim: "MPPSchedulerLimited",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Per-Tock occupancy for Case 4 — MPP with k-port switch(es)."""
    import networkx as nx_local

    G = mpp_lim.G
    n_q = len(G.nodes)
    groups = mpp_lim.scheduled_groups
    n_tocks = len(groups)
    if n_tocks == 0:
        return

    nodes_list = list(G.nodes)
    k = mpp_lim.switch.k
    n_sw = mpp_lim.switch.n_switches

    half_k = k // 2
    port_sp = 0.7
    sw_box_w = max(half_k, 1) * port_sp + 0.4
    sw_gap = 1.2
    total_sw_w = n_sw * sw_box_w + (n_sw - 1) * sw_gap

    strip_w = max(10, min(max(n_q, total_sw_w) * 0.70 + 5, 30))
    fig, axes = plt.subplots(
        n_tocks, 2,
        figsize=(strip_w + 7, 7.0 * n_tocks),
        facecolor=_FIG_BG,
        gridspec_kw={"width_ratios": [1, max(1, n_q // 4)]},
    )
    sw_label = f"{n_sw}x k={k}" if n_sw > 1 else f"k={k}"
    fig.suptitle(
        f"Per-Tock Occupancy — Case 4 (MPP, {sw_label} port switch)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    if n_tocks == 1:
        axes = [axes]

    pos = nx_local.spring_layout(G, seed=42)

    for gi, sg in enumerate(groups):
        ax_graph, ax_sw = axes[gi]

        measured = {m["id"] for m in sg.measurements}
        neighbors = set()
        for m in sg.measurements:
            neighbors.update(m["neighbors"])
        neighbors -= measured
        connected = measured | neighbors

        # ── Graph view (left panel) ─────────────────────────────────────
        node_colors = [
            _COL_MEASURE if nd in measured else
            "#FFD54F"    if nd in neighbors else
            _COL_IDLE
            for nd in G.nodes
        ]
        active_edges = []
        for m in sg.measurements:
            for nb in m["neighbors"]:
                if G.has_edge(m["id"], nb):
                    active_edges.append((m["id"], nb))
                elif G.has_edge(nb, m["id"]):
                    active_edges.append((nb, m["id"]))
        all_edges = list(G.edges())
        inactive_edges = [e for e in all_edges
                          if e not in active_edges
                          and (e[1], e[0]) not in active_edges]
        nx_local.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                                     width=1.2, edge_color=_COL_EDGE_OFF, ax=ax_graph)
        nx_local.draw_networkx_edges(G, pos, edgelist=active_edges,
                                     width=2.5, edge_color=_COL_MEASURE, ax=ax_graph)
        nx_local.draw_networkx_nodes(G, pos, node_size=700,
                                     node_color=node_colors,
                                     edgecolors=_COL_BORDER, linewidths=1.2,
                                     ax=ax_graph)
        labels = {
            nd: (f"{nd}\nX" if nd in measured else
                 f"{nd}\nZ" if nd in neighbors else str(nd))
            for nd in G.nodes
        }
        nx_local.draw_networkx_labels(G, pos, labels,
                                      font_size=9, font_color="#222", ax=ax_graph)
        op_strs = []
        for m in sg.measurements:
            parts = [f"X{m['id']}"] + [f"Z{nb}" for nb in m["neighbors"]]
            op_strs.append(" ".join(parts))
        ax_graph.text(0.5, 0.02, "  |  ".join(op_strs),
                      transform=ax_graph.transAxes,
                      ha="center", fontsize=8, family="monospace",
                      color="#666", zorder=3)
        ax_graph.legend(
            handles=[
                mpatches.Patch(facecolor=_COL_MEASURE, edgecolor=_COL_BORDER,
                               lw=1.2, label="X (measured)"),
                mpatches.Patch(facecolor="#FFD54F", edgecolor=_COL_BORDER,
                               lw=1.2, label="Z (neighbour)"),
            ],
            fontsize=8, frameon=False, loc="upper right",
        )
        gen_ids = [m["id"] for m in sg.measurements]
        ax_graph.set_title(
            f"Group {gi+1} — generators {gen_ids}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax_graph.axis("off")
        ax_graph.margins(0.18)

        # ── Build per-switch data ───────────────────────────────────────
        sw_nodes: list[set] = [set() for _ in range(n_sw)]
        sw_gens_list: list[list[dict]] = [[] for _ in range(n_sw)]
        if sg.switch_gens and len(sg.switch_gens) == n_sw:
            for si in range(n_sw):
                sw_nodes[si] = sg.switch_support[si] if sg.switch_support else set()
                sw_gens_list[si] = sg.switch_gens[si]
        else:
            sw_nodes[0] = connected
            sw_gens_list[0] = sg.measurements

        sw_data_nodes = [sorted(sw_nodes[si]) for si in range(n_sw)]
        sw_anc_count = [len(sw_gens_list[si]) for si in range(n_sw)]

        total_ports = n_sw * k
        _draw_switch_diagram(
            ax_sw,
            nodes_list=nodes_list, n_q=n_q,
            data_active=connected, data_other=set(),
            data_measured=measured, data_neighbor=neighbors,
            k=k, n_sw=n_sw,
            sw_data_nodes=sw_data_nodes,
            sw_anc_count=sw_anc_count,
            title=(f"Group {gi+1} — {sg.n_ports_used}/{total_ports} ports "
                   f"({n_sw} switch{'es' if n_sw > 1 else ''}, k={k}), 1 Tock"),
            is_case4=True,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path, "Case 4 occupancy", display)


# ============================================================================
# Per-Tock Occupancy — Case 5 (CZ + k-port switch)
# ============================================================================

def show_qubit_occupancy_case5(
    cz_lim: "CZSchedulerLimited",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Per-Tock occupancy for Case 5 — CZ with k-port switch(es)."""
    import networkx as nx_local

    G = cz_lim.G
    n_q = len(G.nodes)
    batches = cz_lim.scheduled_batches
    n_tocks = cz_lim.n_tocks()
    if n_tocks == 0:
        return

    nodes_list = list(G.nodes)
    k = cz_lim.switch.k
    n_sw = cz_lim.switch.n_switches

    half_k = k // 2
    port_sp = 0.7
    sw_box_w = max(half_k, 1) * port_sp + 0.4
    sw_gap = 1.2
    total_sw_w = n_sw * sw_box_w + (n_sw - 1) * sw_gap

    strip_w = max(10, min(max(n_q, total_sw_w) * 0.70 + 5, 30))
    fig, axes = plt.subplots(
        n_tocks, 2,
        figsize=(strip_w + 7, 7.0 * n_tocks),
        facecolor=_FIG_BG,
        gridspec_kw={"width_ratios": [1, max(1, n_q // 4)]},
    )
    sw_label = f"{n_sw}x k={k}" if n_sw > 1 else f"k={k}"
    fig.suptitle(
        f"Per-Tock Occupancy — Case 5 (CZ, {sw_label} port switch)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    if n_tocks == 1:
        axes = [axes]

    pos = nx_local.spring_layout(G, seed=42)

    for t in range(n_tocks):
        ax_graph, ax_sw = axes[t]
        batch_idx = t // 2
        step = t % 2
        sb = batches[batch_idx]
        col = _PALETTE[batch_idx % len(_PALETTE)]

        active_step1 = {u for u, v in sb.edges}
        active_step2 = {v for u, v in sb.edges}
        active_now = active_step1 if step == 0 else active_step2
        active_other = active_step2 if step == 0 else active_step1

        # ── Graph view (left panel) ─────────────────────────────────────
        node_colors = [
            "#FFCC80" if nd in active_now else
            "#FFECB3" if nd in active_other else
            _COL_IDLE
            for nd in G.nodes
        ]
        batch_edge_set = {(min(u, v), max(u, v)) for u, v in sb.edges}
        active_edges = [(u, v) for u, v in G.edges()
                        if (min(u, v), max(u, v)) in batch_edge_set]
        inactive_edges = [(u, v) for u, v in G.edges()
                          if (min(u, v), max(u, v)) not in batch_edge_set]
        nx_local.draw_networkx_edges(G, pos, edgelist=inactive_edges,
                                     width=1.2, edge_color=_COL_EDGE_OFF, ax=ax_graph)
        nx_local.draw_networkx_edges(G, pos, edgelist=active_edges,
                                     width=2.5, edge_color=col, ax=ax_graph)
        nx_local.draw_networkx_nodes(G, pos, node_size=700,
                                     node_color=node_colors,
                                     edgecolors=_COL_BORDER, linewidths=1.2,
                                     ax=ax_graph)
        step_lbl = "Z_u" if step == 0 else "Z_v"
        labels = {
            nd: (f"{nd}\n{step_lbl}" if nd in active_now else str(nd))
            for nd in G.nodes
        }
        nx_local.draw_networkx_labels(G, pos, labels,
                                      font_size=8, font_color="#222", ax=ax_graph)
        edge_strs = [f"CZ({u},{v})" for u, v in sb.edges]
        ax_graph.text(0.5, 0.02, "  ".join(edge_strs),
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
        step_name = "Step 1 (Z_u x Z_anc)" if step == 0 else "Step 2 (Z_v x Z_anc)"
        ax_graph.set_title(
            f"Tock {t+1} — Sub-batch {batch_idx+1}, {step_name}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax_graph.axis("off")
        ax_graph.margins(0.18)

        # ── Build per-switch data ───────────────────────────────────────
        n_gates = len(sb.edges)
        sw_edge_lists: list[list[tuple]] = [[] for _ in range(n_sw)]
        if sb.switch_edges and len(sb.switch_edges) == n_sw:
            sw_edge_lists = sb.switch_edges
        else:
            sw_edge_lists[0] = list(sb.edges)

        # Data nodes per switch (active this step)
        sw_data_nodes_list: list[list] = [[] for _ in range(n_sw)]
        for si in range(n_sw):
            for u, v in sw_edge_lists[si]:
                nd_now = u if step == 0 else v
                sw_data_nodes_list[si].append(nd_now)
            sw_data_nodes_list[si].sort()

        sw_anc_count = [len(sw_edge_lists[si]) for si in range(n_sw)]
        total_ports = n_sw * k

        _draw_switch_diagram(
            ax_sw,
            nodes_list=nodes_list, n_q=n_q,
            data_active=active_now, data_other=active_other,
            data_measured=set(), data_neighbor=set(),
            k=k, n_sw=n_sw,
            sw_data_nodes=sw_data_nodes_list,
            sw_anc_count=sw_anc_count,
            title=(f"Tock {t+1} — {n_gates} CZ gate(s), "
                   f"{sb.n_ports_used}/{total_ports} ports "
                   f"({n_sw} switch{'es' if n_sw > 1 else ''}, k={k})"),
            is_case4=False,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path, "Case 5 occupancy", display)


# ============================================================================

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
