"""
visualizer.py
=============
Figures for DQS-Sim schedules.

Every figure shares one visual language:

  * colour = operation family:  blue for MPP (Cases 1 and 4), orange for CZ
    (Cases 2, 3 and 5), green for MIS qubits that start in |+> for free;
  * two shades per family: the strong shade marks the measured / second-step
    qubit, the mid shade marks Z-neighbours / first-step qubits;
  * time order (Tock, round, sub-batch) runs light -> dark in the family hue;
  * idle elements are warm grey, text is near-black, titles sit top-left.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

if TYPE_CHECKING:
    from dqs_sim.cost import ResourceReport
    from dqs_sim.schedulers import (
        CZScheduler, CZSchedulerFixedAncilla, CZSchedulerLimited,
        MPPSchedulerLimited, SubstrateScheduler,
    )


# Palette

INK       = "#1d1d1b"  # primary text
INK_2     = "#52514e"  # secondary text, axis labels
INK_3     = "#8a8984"  # muted captions
RULE      = "#c9c8c2"  # axis lines, dividers
GRID      = "#ebeae6"  # gridlines
SURFACE   = "#ffffff"
BAND      = "#f7f6f3"  # alternating background bands

IDLE      = "#f3f2ef"
IDLE_EDGE = "#e2e1dc"
IDLE_TEXT = "#b3b2ac"
EDGE_OFF  = "#dddcd7"  # inactive graph edges

MPP       = "#2a78d6"
MPP_MID   = "#86b6ef"
MPP_TINT  = "#e6f0fc"
MPP_DARK  = "#104281"

CZ        = "#eb6834"
CZ_MID    = "#f5a582"
CZ_TINT   = "#fdeee7"
CZ_DARK   = "#8f3a17"

FREE      = "#1baf7a"
FREE_MID  = "#a5e0c9"

_FAMILY = {
    "mpp": {"strong": MPP, "mid": MPP_MID, "tint": MPP_TINT, "dark": MPP_DARK},
    "cz":  {"strong": CZ,  "mid": CZ_MID,  "tint": CZ_TINT,  "dark": CZ_DARK},
}

_FS_TITLE    = 12
_FS_SUBTITLE = 9
_FS_LABEL    = 10
_FS_TICK     = 9
_FS_SMALL    = 8
_FS_TINY     = 7


def _init_plot_style() -> None:
    """Shared Matplotlib defaults for every DQS-Sim figure."""
    mpl.rcParams.update({
        "figure.facecolor":    SURFACE,
        "savefig.facecolor":   SURFACE,
        "axes.facecolor":      SURFACE,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                                "DejaVu Sans"],
        "mathtext.fontset":    "dejavusans",
        "text.color":          INK,
        "axes.labelcolor":     INK_2,
        "axes.titlecolor":     INK,
        "axes.edgecolor":      RULE,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.titlesize":      _FS_TITLE,
        "axes.titleweight":    "bold",
        "axes.titlelocation":  "left",
        "axes.labelsize":      _FS_LABEL,
        "xtick.color":         RULE,
        "ytick.color":         RULE,
        "xtick.labelcolor":    INK_2,
        "ytick.labelcolor":    INK_2,
        "xtick.labelsize":     _FS_TICK,
        "ytick.labelsize":     _FS_TICK,
        "xtick.major.size":    0,
        "ytick.major.size":    0,
        "legend.frameon":      False,
        "legend.fontsize":     _FS_SMALL,
        "legend.handlelength": 1.2,
        "legend.handleheight": 1.0,
        "savefig.dpi":         200,
    })


_init_plot_style()


# Small helpers

def _shades(family: str, n: int) -> list[str]:
    """n colours in the family hue, ordered light -> dark (time order)."""
    f = _FAMILY[family]
    if n <= 0:
        return []
    if n == 1:
        return [f["strong"]]
    cmap = LinearSegmentedColormap.from_list(
        family, [f["mid"], f["strong"], f["dark"]])
    return [mpl.colors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Left-aligned bold title with an optional muted subtitle underneath."""
    if subtitle:
        ax.set_title(title, pad=24)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=_FS_SUBTITLE, color=INK_3, ha="left", va="bottom")
    else:
        ax.set_title(title, pad=10)


def _legend_above(ax: plt.Axes, handles: list) -> None:
    """Legend in one row above the top-right corner of the axes."""
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=len(handles), borderaxespad=0.3, columnspacing=1.2,
              handletextpad=0.5)


def _swatch(colour: str, label: str, edge: str | None = None) -> mpatches.Patch:
    return mpatches.Patch(facecolor=colour, edgecolor=edge or colour,
                          linewidth=0.8, label=label)


def _text_on(fill: str) -> str:
    """White text on strong fills, ink on light ones."""
    r, g, b = mpl.colors.to_rgb(fill)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.55 else INK


def _positions(G: nx.Graph) -> dict:
    """One layout per graph, shared by every figure (grids come out as grids)."""
    key = (tuple(G.nodes), tuple(G.edges))
    cached = _POS_CACHE.get(id(G))
    if cached and cached[0] == key:
        return cached[1]
    n = len(G)
    if n == 0:
        pos = {}
    elif 2 < n <= 120 and nx.is_connected(G):
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    _POS_CACHE[id(G)] = (key, pos)
    return pos


_POS_CACHE: dict = {}


def _tick_step(n: int, target: int = 20) -> int:
    return max(1, math.ceil(n / target))


def _save_or_show(
    fig:       plt.Figure,
    save_path: str | None,
    label:     str,
    display:   bool = False,
) -> None:
    """Save to file and/or display interactively.

    display=True  -> always call plt.show(), even when a save_path is given.
    display=False -> only show when save_path is None.
    """
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  {label.capitalize()} saved -> {save_path}")
    if not save_path or display:
        plt.show()
    plt.close(fig)


# Graph panels

def _node_size(n: int) -> float:
    return float(max(120, min(620, 5200 / max(n, 1))))


def _draw_graph(
    ax:          plt.Axes,
    G:           nx.Graph,
    pos:         dict,
    node_fill:   dict,
    edge_colour: dict,
    caption:     str | None = None,
    labels:      bool = True,
) -> None:
    """Graph view: inactive edges thin grey, active edges thick and coloured."""
    off = [e for e in G.edges() if (min(e), max(e)) not in edge_colour]
    on  = [e for e in G.edges() if (min(e), max(e)) in edge_colour]
    nx.draw_networkx_edges(G, pos, edgelist=off, width=1.0,
                           edge_color=EDGE_OFF, ax=ax)
    if on:
        nx.draw_networkx_edges(G, pos, edgelist=on, width=2.6,
                               edge_color=[edge_colour[(min(e), max(e))] for e in on],
                               ax=ax)
    fills = [node_fill.get(v, IDLE) for v in G.nodes]
    edges = [RULE if node_fill.get(v, IDLE) == IDLE else SURFACE for v in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=_node_size(len(G)), node_color=fills,
                           edgecolors=edges, linewidths=1.2, ax=ax)
    if labels and len(G) <= 40:
        for v, (x, y) in pos.items():
            fill = node_fill.get(v, IDLE)
            ax.text(x, y, str(v), ha="center", va="center",
                    fontsize=_FS_SMALL if len(G) <= 20 else _FS_TINY,
                    fontweight="bold",
                    color=_text_on(fill) if fill != IDLE else INK_3, zorder=5)
    if caption:
        ax.text(0.0, -0.02, caption, transform=ax.transAxes, ha="left", va="top",
                fontsize=_FS_TINY, family="monospace", color=INK_3)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.axis("off")


def _draw_tile_strip(
    ax:        plt.Axes,
    nodes:     list,
    data_fill: dict,
    ancillas:  dict,
    anc_fill:  str,
    rows:      tuple[str, str] = ("Ancilla", "Data"),
) -> None:
    """
    Two rows of tiles: ancilla QPUs docked above the data QPUs they touch.

    data_fill : {node: fill} for active data QPUs (others drawn idle)
    ancillas  : {node: label} ancilla docked above that data QPU
    """
    n = len(nodes)
    ax.set_xlim(-1.9, n + 0.1)
    ax.set_ylim(-0.25, 2.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.25, 2.05, rows[0], ha="right", va="center",
            fontsize=_FS_SMALL, color=INK_2)
    ax.text(-0.25, 0.45, rows[1], ha="right", va="center",
            fontsize=_FS_SMALL, color=INK_2)
    label_fs = _FS_TINY if n <= 24 else 5
    for i, v in enumerate(nodes):
        fill = data_fill.get(v)
        ax.add_patch(FancyBboxPatch(
            (i + 0.06, 0.06), 0.88, 0.78, boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=fill or IDLE, edgecolor=fill or IDLE_EDGE, linewidth=0.8))
        if n <= 48:
            ax.text(i + 0.5, 0.45, f"Q{v}", ha="center", va="center",
                    fontsize=label_fs, fontweight="bold" if fill else "normal",
                    color=_text_on(fill) if fill else IDLE_TEXT)
        if v in ancillas:
            ax.plot([i + 0.5, i + 0.5], [0.84, 1.66], color=anc_fill,
                    linewidth=1.4, solid_capstyle="round", zorder=0)
            ax.add_patch(FancyBboxPatch(
                (i + 0.06, 1.66), 0.88, 0.78, boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=anc_fill, edgecolor=anc_fill, linewidth=0.8))
            if n <= 48:
                ax.text(i + 0.5, 2.05, ancillas[v], ha="center", va="center",
                        fontsize=label_fs, fontweight="bold", color=_text_on(anc_fill))
        else:
            ax.add_patch(FancyBboxPatch(
                (i + 0.06, 1.66), 0.88, 0.78, boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=SURFACE, edgecolor=IDLE_EDGE, linewidth=0.8, linestyle=(0, (2, 2))))


def _operators(measurements: list) -> str:
    """'X1 Z0 Z2 Z4' per measurement, shortened when there are many."""
    ops = "   ".join(" ".join([f"X{m['id']}"] + [f"Z{nb}" for nb in m["neighbors"]])
                     for m in measurements)
    if len(ops) <= 60:
        return ops
    n = len(measurements)
    return f"{n} measurement{'s' if n != 1 else ''}"


def _per_tock_figure(n_rows: int, n_q: int, right_ratio: float = 1.8):
    width = max(9.0, min(6.0 + n_q * 0.42, 22.0))
    height = 2.6 * n_rows + 0.6
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(width, height), layout="constrained",
        gridspec_kw={"width_ratios": [1, right_ratio]}, squeeze=False,
    )
    return fig, axes


# 2D patch layout

def _draw_patch_layout(
    ax:          plt.Axes,
    G:           nx.Graph,
    pos:         dict,
    node_fill:   dict,
    node_edge:   dict,
    edge_colour: dict,
) -> None:
    """QPUs as rounded surface-code patches; edges drawn border to border."""
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    pts = list(pos.values())
    min_d = min((math.hypot(a[0] - b[0], a[1] - b[1])
                 for i, a in enumerate(pts) for b in pts[i + 1:]), default=span)
    hw = max(0.012 * span, min(0.06 * span, 0.28 * min_d))

    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        col = edge_colour.get(key, EDGE_OFF)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=col, linewidth=2.6 if key in edge_colour else 1.0,
                solid_capstyle="round", zorder=1)

    for v, (x, y) in pos.items():
        fill = node_fill.get(v, IDLE)
        ax.add_patch(FancyBboxPatch(
            (x - hw, y - hw), 2 * hw, 2 * hw,
            boxstyle=f"round,pad=0,rounding_size={hw * 0.35}",
            facecolor=fill, edgecolor=node_edge.get(v, RULE),
            linewidth=1.2, zorder=2))
        if len(G) <= 40:
            ax.text(x, y, str(v), ha="center", va="center",
                    fontsize=_FS_SMALL if len(G) <= 20 else _FS_TINY,
                    fontweight="bold", color=_text_on(fill), zorder=3)

    pad = hw * 2.2
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _order_legend(ax: plt.Axes, colours: list[str], word: str,
                  extra: list | None = None) -> None:
    """Time-order legend in one row under the panel (first ... last if long)."""
    n = len(colours)
    idx = list(range(n)) if n <= 8 else [0, 1, 2, n - 2, n - 1]
    handles = []
    for j, i in enumerate(idx):
        if n > 8 and j == 3:
            handles.append(Line2D([], [], linestyle="none", label="..."))
        handles.append(Line2D([], [], color=colours[i], linewidth=3, label=str(i + 1)))
    leg = ax.legend(handles=handles, title=word, loc="upper left",
                    bbox_to_anchor=(0.0, 0.0), ncol=len(handles), columnspacing=0.9,
                    handlelength=1.4, alignment="left", title_fontsize=_FS_SMALL)
    if extra:
        ax.add_artist(leg)
        ax.legend(handles=extra, loc="upper right")


def show_2d_layout(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    QPUs drawn as surface-code patches, one panel per case.

    Case 1: MIS QPUs (green) start in |+>; measured QPUs (blue).  Edges are
    shaded by the Tock that covers them.  Cases 2/3: edges shaded by CZ round
    or sub-batch.
    """
    G   = scheduler.G
    pos = _positions(G)

    tock_of: dict[tuple, int] = {}
    for t, tock in enumerate(scheduler.schedule):
        for b in tock:
            for v in b["neighbors"]:
                tock_of.setdefault((min(b["id"], v), max(b["id"], v)), t)
    t_cols = _shades("mpp", len(scheduler.schedule))

    round_of = {(min(u, v), max(u, v)): r
                for r, edges in enumerate(cz.cz_rounds) for u, v in edges}
    r_cols = _shades("cz", len(cz.cz_rounds))

    panels = 3 if cz_k is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.4 * panels, 6.0),
                             layout="constrained")

    mis = set(scheduler.mis)
    fill1 = {v: (FREE_MID if v in mis else MPP_MID) for v in G.nodes}
    edge1 = {v: (FREE if v in mis else MPP) for v in G.nodes}
    _draw_patch_layout(axes[0], G, pos, fill1, edge1,
                       {e: t_cols[t] for e, t in tock_of.items()})
    _title(axes[0], "Case 1: MPP",
           f"{len(scheduler.schedule)} Tocks, edges shaded by Tock")
    _order_legend(axes[0], t_cols, "Tock", extra=[
        _swatch(FREE_MID, "MIS, starts in |+>", FREE),
        _swatch(MPP_MID, "measured", MPP),
    ])

    _draw_patch_layout(axes[1], G, pos, {}, {},
                       {e: r_cols[r] for e, r in round_of.items()})
    _title(axes[1], "Case 2: CZ",
           f"{len(cz.cz_rounds)} rounds, edges shaded by round")
    _order_legend(axes[1], r_cols, "CZ round")

    if cz_k is not None:
        batch_of = {(min(u, v), max(u, v)): b
                    for b, batch in enumerate(cz_k.batches) for u, v in batch}
        b_cols = _shades("cz", len(cz_k.batches))
        _draw_patch_layout(axes[2], G, pos, {}, {},
                           {e: b_cols[b] for e, b in batch_of.items()})
        _title(axes[2], f"Case 3: CZ, k = {cz_k.k} ancillas",
               f"{len(cz_k.batches)} sub-batches, edges shaded by sub-batch")
        _order_legend(axes[2], b_cols, "Sub-batch")

    _save_or_show(fig, save_path, "2D layout", display)


# Schedule heatmaps

def _build_heatmap_case1(scheduler: "SubstrateScheduler") -> np.ndarray:
    """Matrix[qubit_row, tock] = 0 (idle) | 1 (Z-neighbour) | 2 (X-measured)."""
    nds = list(scheduler.G.nodes)
    row = {n: i for i, n in enumerate(nds)}
    m = np.zeros((len(nds), len(scheduler.schedule)), dtype=np.int8)
    for t, tock in enumerate(scheduler.schedule):
        for b in tock:
            m[row[b["id"]], t] = 2
            for nb in b["neighbors"]:
                if m[row[nb], t] == 0:
                    m[row[nb], t] = 1
    return m


def _build_heatmap_cz(G: nx.Graph, batches: list) -> np.ndarray:
    """Matrix[qubit_row, tock] = 0 (idle) | 1 (step-1 qubit) | 2 (step-2 qubit)."""
    row = {n: i for i, n in enumerate(G.nodes)}
    m = np.zeros((len(G), 2 * len(batches)), dtype=np.int8)
    for b, edges in enumerate(batches):
        for u, v in edges:
            m[row[u], 2 * b] = 1
            m[row[v], 2 * b + 1] = 2
    return m


def _build_heatmap_case2(cz: "CZScheduler") -> np.ndarray:
    return _build_heatmap_cz(cz.G, cz.cz_rounds)


def _build_heatmap_case3(cz_k: "CZSchedulerFixedAncilla") -> np.ndarray:
    return _build_heatmap_cz(cz_k.G, cz_k.batches)


def _mpp_legend() -> list:
    return [_swatch(IDLE, "idle", IDLE_EDGE), _swatch(MPP_MID, "Z neighbour"),
            _swatch(MPP, "X measured")]


def _cz_legend() -> list:
    return [_swatch(IDLE, "idle", IDLE_EDGE), _swatch(CZ_MID, "CZ step 1"),
            _swatch(CZ, "CZ step 2")]


def plot_heatmap(
    matrix:   np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title:    str,
    cmap:     ListedColormap,
    legend_handles: list,
    ax:       plt.Axes,
    vline_every: int = 0,
    subtitle: str | None = None,
) -> None:
    """Qubit x Tock grid; one cell per (QPU, Tock), white gaps between cells."""
    n_q, n_t = matrix.shape
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=cmap.N - 1,
              interpolation="nearest", origin="upper")
    gap = 1.6 if max(n_q, n_t) <= 40 else 0.6
    ax.set_xticks(np.arange(-0.5, n_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_q, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=gap)
    ax.tick_params(which="minor", length=0)
    if vline_every > 0:
        for x in range(vline_every, n_t, vline_every):
            ax.axvline(x - 0.5, color=RULE, linewidth=1.2, zorder=4)
    xs = _tick_step(n_t, 24)
    ys = _tick_step(n_q, 30)
    ax.set_xticks(range(0, n_t, xs))
    ax.set_xticklabels(x_labels[::xs], fontsize=_FS_TINY)
    ax.set_yticks(range(0, n_q, ys))
    ax.set_yticklabels(y_labels[::ys], fontsize=_FS_TINY)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Tock")
    _title(ax, title, subtitle)
    _legend_above(ax, legend_handles)


def show_heatmaps(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Side-by-side schedule heatmaps for Cases 1, 2 (and 3)."""
    nds = list(scheduler.G.nodes)
    y_lbl = [f"Q{v}" for v in nds]
    mats = [(_build_heatmap_case1(scheduler), "mpp")]
    mats.append((_build_heatmap_case2(cz), "cz"))
    if cz_k is not None:
        mats.append((_build_heatmap_case3(cz_k), "cz"))

    widths = [max(m.shape[1], 3) for m, _ in mats]
    height = max(4.2, min(len(nds) * 0.24 + 2.2, 18))
    fig_w = max(11, sum(widths) * 0.32 + 3.5 * len(mats))
    fig, axes = plt.subplots(1, len(mats), figsize=(fig_w, height), layout="constrained",
                             gridspec_kw={"width_ratios": widths})

    cm_mpp = ListedColormap([IDLE, MPP_MID, MPP])
    cm_cz  = ListedColormap([IDLE, CZ_MID, CZ])

    m1 = mats[0][0]
    plot_heatmap(m1, [f"T{t+1}" for t in range(m1.shape[1])], y_lbl,
                 "Case 1: MPP", cm_mpp, _mpp_legend(), axes[0],
                 subtitle=f"{m1.shape[1]} Tocks")
    m2 = mats[1][0]
    plot_heatmap(m2, [f"T{t+1}" for t in range(m2.shape[1])], y_lbl,
                 "Case 2: CZ", cm_cz, _cz_legend(), axes[1], vline_every=2,
                 subtitle=f"{m2.shape[1]} Tocks, {len(cz.cz_rounds)} rounds of 2 Tocks")
    if cz_k is not None:
        m3 = mats[2][0]
        plot_heatmap(m3, [f"T{t+1}" for t in range(m3.shape[1])], y_lbl,
                     f"Case 3: CZ, k = {cz_k.k}", cm_cz, _cz_legend(), axes[2],
                     vline_every=2,
                     subtitle=f"{m3.shape[1]} Tocks, {len(cz_k.batches)} sub-batches")
    _save_or_show(fig, save_path, "heatmap", display)


# Gantt charts

def _gantt_axes(ax: plt.Axes, n_t: int, nodes: list, title: str,
                subtitle: str, legend: list) -> dict:
    """Common frame: Q0 at the top, a header lane for group labels."""
    n_q = len(nodes)
    ax.set_xlim(0, max(n_t, 1))
    ax.set_ylim(n_q - 0.5, -1.4)  # inverted: Q0 on top, header lane above it
    step = _tick_step(n_t, 24)
    ax.set_xticks([t + 0.5 for t in range(0, n_t, step)])
    ax.set_xticklabels([f"T{t+1}" for t in range(0, n_t, step)], fontsize=_FS_TINY)
    ys = _tick_step(n_q, 30)
    ax.set_yticks(range(0, n_q, ys))
    ax.set_yticklabels([f"Q{v}" for v in nodes[::ys]], fontsize=_FS_TINY)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Tock")
    _title(ax, title, subtitle)
    _legend_above(ax, legend)
    return {v: i for i, v in enumerate(nodes)}


def _bar(ax: plt.Axes, t: float, row: int, fill: str, text: str | None,
         show_text: bool) -> None:
    ax.add_patch(FancyBboxPatch(
        (t + 0.08, row - 0.36), 0.84, 0.72,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=fill, edgecolor=fill, linewidth=0, zorder=2))
    if show_text and text:
        ax.text(t + 0.5, row, text, ha="center", va="center", fontsize=_FS_TINY,
                fontweight="bold", color=_text_on(fill), zorder=3)


def plot_gantt_case1(scheduler: "SubstrateScheduler", ax: plt.Axes) -> None:
    """Gantt chart for Case 1: X on the measured QPU, Z on its neighbours."""
    nodes = list(scheduler.G.nodes)
    n_t = len(scheduler.schedule)
    row = _gantt_axes(ax, n_t, nodes, "Case 1: MPP", f"{n_t} Tocks", _mpp_legend()[1:])
    show = len(nodes) <= 40
    for t, tock in enumerate(scheduler.schedule):
        if t % 2:
            ax.axvspan(t, t + 1, color=BAND, zorder=0, linewidth=0)
        for b in tock:
            _bar(ax, t, row[b["id"]], MPP, "X", show)
            for nb in b["neighbors"]:
                _bar(ax, t, row[nb], MPP_MID, "Z", show)


def _gantt_cz(ax: plt.Axes, G: nx.Graph, batches: list, word: str,
              title: str, subtitle: str) -> None:
    nodes = list(G.nodes)
    n_t = 2 * len(batches)
    row = _gantt_axes(ax, n_t, nodes, title, subtitle, _cz_legend()[1:])
    show = len(nodes) <= 40
    for b, edges in enumerate(batches):
        t1 = 2 * b
        if b % 2:
            ax.axvspan(t1, t1 + 2, color=BAND, zorder=0, linewidth=0)
        if n_t <= 60:
            ax.text(t1 + 1, -0.95, f"{word}{b+1}", ha="center", va="center",
                    fontsize=_FS_TINY, color=INK_3)
        for u, v in edges:
            _bar(ax, t1, row[u], CZ_MID, None, show)
            _bar(ax, t1 + 1, row[v], CZ, None, show)


def plot_gantt_case2(cz: "CZScheduler", ax: plt.Axes) -> None:
    """Gantt chart for Case 2: each CZ round takes 2 Tocks."""
    _gantt_cz(ax, cz.G, cz.cz_rounds, "R", "Case 2: CZ",
              f"{2 * len(cz.cz_rounds)} Tocks, {len(cz.cz_rounds)} rounds")


def plot_gantt_case3(cz_k: "CZSchedulerFixedAncilla", ax: plt.Axes) -> None:
    """Gantt chart for Case 3: sub-batches of at most k CZ gates, 2 Tocks each."""
    _gantt_cz(ax, cz_k.G, cz_k.batches, "B", f"Case 3: CZ, k = {cz_k.k}",
              f"{cz_k.n_tocks()} Tocks, {len(cz_k.batches)} sub-batches")


def show_gantt(
    scheduler: "SubstrateScheduler",
    cz:        "CZScheduler",
    cz_k:      "CZSchedulerFixedAncilla | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Side-by-side Gantt charts for Cases 1, 2 (and 3)."""
    n_q = len(scheduler.G.nodes)
    widths = [max(len(scheduler.schedule), 3), max(2 * len(cz.cz_rounds), 3)]
    if cz_k is not None:
        widths.append(max(cz_k.n_tocks(), 3))
    height = max(4.2, min(n_q * 0.30 + 2.4, 20))
    fig_w = max(12, sum(widths) * 0.36 + 3.5 * len(widths))
    fig, axes = plt.subplots(1, len(widths), figsize=(fig_w, height),
                             layout="constrained", gridspec_kw={"width_ratios": widths})
    plot_gantt_case1(scheduler, axes[0])
    plot_gantt_case2(cz, axes[1])
    if cz_k is not None:
        plot_gantt_case3(cz_k, axes[2])
    _save_or_show(fig, save_path, "Gantt chart", display)


# Comparison dashboard

def show_comparison_dashboard(
    report:    "ResourceReport",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Tocks, peak ancillas and physical qubits for every case that was run."""
    entries = [
        ("Case 1", "MPP", MPP, report.tocks_case1,
         report.peak_ancilla_ELUs_case1, report.total_phys_qubits_case1),
        ("Case 2", "CZ", CZ, report.tocks_case2,
         report.peak_ancilla_ELUs_case2, report.total_phys_qubits_case2),
    ]
    if report.tocks_case3 is not None:
        entries.append(("Case 3", f"CZ, k={report.ancilla_k}", CZ_MID,
                        report.tocks_case3, report.peak_ancilla_ELUs_case3,
                        report.total_phys_qubits_case3))
    if report.tocks_case4 is not None:
        entries.append(("Case 4", f"MPP, {report.switch_k} ports", MPP_MID,
                        report.tocks_case4, report.peak_ancilla_ELUs_case4,
                        report.total_phys_qubits_case4))
    if report.tocks_case5 is not None:
        entries.append(("Case 5", f"CZ, {report.switch_k} ports", CZ_MID,
                        report.tocks_case5, report.peak_ancilla_ELUs_case5,
                        report.total_phys_qubits_case5))

    labels  = [f"{c}\n{d}" for c, d, *_ in entries]
    colours = [e[2] for e in entries]
    n = len(entries)
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.4 * n * 3 + 2), 4.6),
                             layout="constrained")

    panels = [
        ("Time", "Tocks (logical time steps)", [e[3] for e in entries], "{:,}"),
        ("Peak ancilla QPUs", "ancillas in use at the same time",
         [e[4] for e in entries], "{:,}"),
        ("Physical qubits", f"d = {report.code_distance}, "
         f"{report.phys_per_ELU} physical qubits per QPU",
         [e[5] for e in entries], "{:,}"),
    ]
    for ax, (title, subtitle, vals, fmt) in zip(axes, panels):
        x = np.arange(n)
        clean = [v if v is not None else 0 for v in vals]
        ax.bar(x, clean, width=0.62, color=colours, edgecolor=SURFACE,
               linewidth=2, zorder=2)
        top = max(clean) if clean and max(clean) > 0 else 1
        for xi, v in zip(x, vals):
            ax.text(xi, (v or 0) + top * 0.02, "-" if v is None else fmt.format(v),
                    ha="center", va="bottom", fontsize=_FS_SMALL,
                    fontweight="bold", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=_FS_SMALL)
        ax.set_ylim(0, top * 1.15)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
        _title(ax, title, subtitle)

    fig.legend(handles=[_swatch(MPP, "MPP"), _swatch(CZ, "CZ"),
                        _swatch(RULE, "lighter shade = constrained (k ancillas or k ports)")],
               loc="outside lower left", ncol=3)
    _save_or_show(fig, save_path, "dashboard", display)


# Per-Tock occupancy: Case 1

def _occupancy_legend(fig: plt.Figure, handles: list) -> None:
    fig.legend(handles=handles, loc="outside upper left", ncol=len(handles))


def show_qubit_occupancy(
    scheduler: "SubstrateScheduler",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """
    One row per Tock (Case 1, MPP).  Left: the graph with the measured QPU (X)
    and its Z-neighbours.  Right: ancilla and data QPUs in use.
    """
    if not scheduler.schedule:
        print("  No tocks to display.")
        return
    G = scheduler.G
    nodes = list(G.nodes)
    pos = _positions(G)
    layout_free = getattr(scheduler, "layout_free", False)
    fig, axes = _per_tock_figure(len(scheduler.schedule), len(nodes))
    _occupancy_legend(fig, [_swatch(MPP, "X measured"), _swatch(MPP_MID, "Z neighbour"),
                            _swatch(IDLE, "idle", IDLE_EDGE)])

    for t, tock in enumerate(scheduler.schedule):
        ax_g, ax_t = axes[t]
        measured = {b["id"] for b in tock}
        nbrs = {nb for b in tock for nb in b["neighbors"]} - measured
        fill = {**{v: MPP_MID for v in nbrs}, **{v: MPP for v in measured}}
        edges = {(min(b["id"], nb), max(b["id"], nb)): MPP
                 for b in tock for nb in b["neighbors"]}
        _draw_graph(ax_g, G, pos, fill, edges)
        _title(ax_g, f"Tock {t + 1}", _operators(tock))

        if layout_free:
            ancillas = {b["id"]: f"A{i}" for i, b in enumerate(tock)}
        else:  # substrate model: the ancilla strip spans positions L..R
            by_pos = {scheduler.mapping[v]: v for v in nodes}
            ancillas = {}
            for i, b in enumerate(tock):
                for p in range(b["L"], b["R"] + 1):
                    if p in by_pos:
                        ancillas[by_pos[p]] = f"A{i}"
        _draw_tile_strip(ax_t, nodes, fill, ancillas, MPP)
        _title(ax_t, "QPUs in use",
               f"{len(measured) + len(nbrs)} data, {len(tock)} ancilla")

    _save_or_show(fig, save_path, "qubit occupancy", display)


# Per-Tock occupancy: Cases 2 and 3

def _occupancy_cz(G: nx.Graph, batches: list, word: str, pool: int | None,
                  save_path: str | None, display: bool, label: str) -> None:
    n_t = 2 * len(batches)
    if n_t == 0:
        print("  No tocks to display.")
        return
    nodes = list(G.nodes)
    pos = _positions(G)
    fig, axes = _per_tock_figure(n_t, len(nodes))
    _occupancy_legend(fig, [_swatch(CZ, "active this Tock"),
                            _swatch(CZ_MID, "other end of the CZ"),
                            _swatch(IDLE, "idle", IDLE_EDGE)])
    for t in range(n_t):
        ax_g, ax_t = axes[t]
        b, step = divmod(t, 2)
        edges = batches[b]
        now   = {u if step == 0 else v for u, v in edges}
        other = {v if step == 0 else u for u, v in edges}
        fill = {**{v: CZ_MID for v in other}, **{v: CZ for v in now}}
        _draw_graph(ax_g, G, pos, fill,
                    {(min(u, v), max(u, v)): CZ for u, v in edges})
        _title(ax_g, f"Tock {t + 1}",
               f"{word} {b + 1}, step {step + 1}: "
               + (r"$Z_u \otimes Z_{anc}$" if step == 0 else r"$Z_v \otimes Z_{anc}$"))
        ancillas = {(u if step == 0 else v): f"A{i}" for i, (u, v) in enumerate(edges)}
        _draw_tile_strip(ax_t, nodes, {v: CZ for v in now}, ancillas, CZ_DARK)
        in_use = f"{len(edges)} of {pool}" if pool else f"{len(edges)}"
        _title(ax_t, "QPUs in use", f"{len(now)} data, {in_use} ancillas")
    _save_or_show(fig, save_path, label, display)


def show_qubit_occupancy_case2(
    cz:        "CZScheduler",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """One row per Tock for Case 2 (CZ, unlimited ancillas)."""
    _occupancy_cz(cz.G, cz.cz_rounds, "Round", None, save_path, display,
                  "Case 2 qubit occupancy")


def show_qubit_occupancy_case3(
    cz_k:      "CZSchedulerFixedAncilla",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """One row per Tock for Case 3 (CZ, fixed pool of k ancillas)."""
    _occupancy_cz(cz_k.G, cz_k.batches, "Sub-batch", cz_k.k, save_path, display,
                  "Case 3 qubit occupancy")


# k-port switch: utilisation and heatmaps

def show_port_utilization(
    mpp_lim: "MPPSchedulerLimited | None" = None,
    cz_lim:  "CZSchedulerLimited | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Ports used per Tock group (Case 4) and per CZ sub-batch (Case 5)."""
    panels = []
    if mpp_lim is not None and mpp_lim.scheduled_groups:
        panels.append(("Case 4: MPP through the switch", mpp_lim.scheduled_groups,
                       mpp_lim.switch.k * mpp_lim.switch.n_switches, "Group", MPP))
    if cz_lim is not None and cz_lim.scheduled_batches:
        panels.append(("Case 5: CZ through the switch", cz_lim.scheduled_batches,
                       cz_lim.switch.k * cz_lim.switch.n_switches, "Sub-batch", CZ))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.4),
                             layout="constrained", squeeze=False)
    for ax, (title, sched, k, xlabel, colour) in zip(axes[0], panels):
        used = [item.n_ports_used for item in sched]
        x = np.arange(len(used))
        ax.bar(x, used, width=0.62, color=colour, edgecolor=SURFACE,
               linewidth=2, zorder=2)
        ax.axhline(k, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
        if len(used) <= 30:
            for xi, u in zip(x, used):
                ax.text(xi, u + k * 0.02, str(u), ha="center", va="bottom",
                        fontsize=_FS_SMALL, fontweight="bold", color=INK)
        step = _tick_step(len(used), 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(i + 1) for i in x[::step]])
        ax.set_xlim(-0.6, len(used) - 0.4)
        ax.set_ylim(0, k * 1.18)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ports used")
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        avg = sum(used) / (len(used) * k) * 100
        tocks = sum(item.tocks for item in sched)
        _title(ax, title, f"dashed line = k = {k} ports, average {avg:.0f}% used, {len(used)} "
               f"{'groups' if xlabel == 'Group' else 'sub-batches'}, {tocks} Tocks")
    _save_or_show(fig, save_path, "port utilization chart", display)


def _build_heatmap_case5_kport(cz_lim: "CZSchedulerLimited") -> tuple[np.ndarray, list[str]]:
    """Matrix[qubit_row, tock] for Case 5: 0 idle, 1 CZ step 1, 2 CZ step 2."""
    m = _build_heatmap_cz(cz_lim.G, [sb.edges for sb in cz_lim.scheduled_batches])
    labels = [f"T{t + 1}" for t in range(m.shape[1])]
    return m, labels


def _build_heatmap_case4_kport(mpp_lim: "MPPSchedulerLimited") -> tuple[np.ndarray, list[str]]:
    """Matrix[qubit_row, tock] for Case 4: 0 idle, 1 Z-neighbour, 2 X-measured."""
    row = {n: i for i, n in enumerate(mpp_lim.G.nodes)}
    m = np.zeros((len(row), len(mpp_lim.scheduled_groups)), dtype=np.int8)
    for gi, sg in enumerate(mpp_lim.scheduled_groups):
        for meas in sg.measurements:
            m[row[meas["id"]], gi] = 2
            for nb in meas["neighbors"]:
                if m[row[nb], gi] == 0:
                    m[row[nb], gi] = 1
    return m, [f"T{g + 1}" for g in range(m.shape[1])]


def show_heatmaps_limited(
    mpp_lim: "MPPSchedulerLimited | None" = None,
    cz_lim:  "CZSchedulerLimited | None" = None,
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """Schedule heatmaps for Cases 4 and/or 5 (k-port switch)."""
    panels = []
    if mpp_lim is not None and mpp_lim.scheduled_groups:
        m, xl = _build_heatmap_case4_kport(mpp_lim)
        panels.append((m, xl, mpp_lim.G, f"Case 4: MPP, {mpp_lim.switch.k} ports",
                       ListedColormap([IDLE, MPP_MID, MPP]), _mpp_legend(), 0,
                       f"{m.shape[1]} Tocks"))
    if cz_lim is not None and cz_lim.scheduled_batches:
        m, xl = _build_heatmap_case5_kport(cz_lim)
        panels.append((m, xl, cz_lim.G, f"Case 5: CZ, {cz_lim.switch.k} ports",
                       ListedColormap([IDLE, CZ_MID, CZ]), _cz_legend(), 2,
                       f"{m.shape[1]} Tocks, {len(cz_lim.scheduled_batches)} sub-batches"))
    if not panels:
        return
    widths = [max(p[0].shape[1], 3) for p in panels]
    n_q = max(p[0].shape[0] for p in panels)
    height = max(4.2, min(n_q * 0.24 + 2.2, 18))
    fig, axes = plt.subplots(1, len(panels), figsize=(max(10, sum(widths) * 0.32 + 4 * len(panels)), height),
                             layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": widths})
    for ax, (m, xl, G, title, cmap, legend, every, sub) in zip(axes[0], panels):
        plot_heatmap(m, xl, [f"Q{v}" for v in G.nodes], title, cmap, legend, ax,
                     vline_every=every, subtitle=sub)
    _save_or_show(fig, save_path, "k-port heatmap", display)


# k-port switch: per-Tock wiring diagrams (Cases 4 and 5)

def _draw_switch_diagram(
    ax: plt.Axes,
    *,
    nodes_list:    list,
    data_fill:     dict,
    sw_data_nodes: list[list],
    sw_anc_count:  list[int],
    k:             int,
    family:        str,
) -> None:
    """
    Data QPUs (top row) wired through k-port switch(es) (middle) to the
    ancilla QPUs in use (bottom row).
    """
    strong = _FAMILY[family]["strong"]
    mid    = _FAMILY[family]["mid"]
    n_q  = len(nodes_list)
    n_sw = len(sw_data_nodes)
    port_sp, port_w = 0.62, 0.46
    sw_w = k * port_sp + 0.3
    sw_gap = 0.9
    total_sw = n_sw * sw_w + (n_sw - 1) * sw_gap
    width = max(n_q, total_sw, sum(sw_anc_count))
    data_y, sw_y, anc_y = 3.6, 1.8, 0.0

    ax.set_xlim(-2.2, width + 0.3)
    ax.set_ylim(anc_y - 0.6, data_y + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    for y, name in [(data_y, "Data"), (sw_y, "Switch"), (anc_y, "Ancilla")]:
        ax.text(-0.35, y, name, ha="right", va="center", fontsize=_FS_SMALL, color=INK_2)

    x0 = (width - n_q) / 2
    data_x = {}
    for i, v in enumerate(nodes_list):
        fill = data_fill.get(v)
        ax.add_patch(FancyBboxPatch(
            (x0 + i + 0.06, data_y - 0.39), 0.88, 0.78,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=fill or IDLE, edgecolor=fill or IDLE_EDGE, linewidth=0.8, zorder=2))
        if n_q <= 48:
            ax.text(x0 + i + 0.5, data_y, f"Q{v}", ha="center", va="center",
                    fontsize=_FS_TINY if n_q <= 24 else 5,
                    fontweight="bold" if fill else "normal",
                    color=_text_on(fill) if fill else IDLE_TEXT, zorder=3)
        data_x[v] = x0 + i + 0.5

    total_anc = sum(sw_anc_count)
    a0 = (width - total_anc) / 2
    anc_x = [a0 + j + 0.5 for j in range(total_anc)]
    for j, ax_x in enumerate(anc_x):
        ax.add_patch(FancyBboxPatch(
            (ax_x - 0.44, anc_y - 0.39), 0.88, 0.78,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=_FAMILY[family]["dark"], edgecolor=_FAMILY[family]["dark"],
            linewidth=0.8, zorder=2))
        ax.text(ax_x, anc_y, f"A{j}", ha="center", va="center", fontsize=_FS_TINY,
                fontweight="bold", color="white", zorder=3)

    s0 = (width - total_sw) / 2
    anc_j = 0
    for si in range(n_sw):
        bx = s0 + si * (sw_w + sw_gap)
        ax.add_patch(FancyBboxPatch(
            (bx, sw_y - 0.42), sw_w, 0.84, boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=BAND, edgecolor=RULE, linewidth=1.0, zorder=1))
        data_nodes = sorted(sw_data_nodes[si], key=nodes_list.index)
        n_anc = sw_anc_count[si]
        for p in range(k):
            px = bx + 0.15 + p * port_sp
            if p < len(data_nodes):
                fc = data_fill.get(data_nodes[p], mid)
            elif p < len(data_nodes) + n_anc:
                fc = _FAMILY[family]["dark"]
            else:
                fc = SURFACE
            ax.add_patch(plt.Rectangle((px, sw_y - 0.2), port_w, 0.4, facecolor=fc,
                                       edgecolor=fc if fc != SURFACE else IDLE_EDGE,
                                       linewidth=0.8, zorder=2))
        for p, v in enumerate(data_nodes):
            if v in data_x:
                ax.plot([data_x[v], bx + 0.15 + p * port_sp + port_w / 2],
                        [data_y - 0.39, sw_y + 0.2], color=mid, linewidth=1.2, zorder=0)
        for a in range(n_anc):
            px = bx + 0.15 + (len(data_nodes) + a) * port_sp + port_w / 2
            ax.plot([px, anc_x[anc_j]], [sw_y - 0.2, anc_y + 0.39], color=strong,
                    linewidth=1.2, zorder=0)
            anc_j += 1


def show_qubit_occupancy_case4(
    mpp_lim:   "MPPSchedulerLimited",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """One row per Tock for Case 4: MPP measurements wired through the switch."""
    groups = mpp_lim.scheduled_groups
    if not groups:
        return
    G = mpp_lim.G
    nodes = list(G.nodes)
    pos = _positions(G)
    k, n_sw = mpp_lim.switch.k, mpp_lim.switch.n_switches
    fig, axes = _per_tock_figure(len(groups), max(len(nodes), k + 2), right_ratio=2.0)
    _occupancy_legend(fig, [_swatch(MPP, "X measured"), _swatch(MPP_MID, "Z neighbour"),
                            _swatch(MPP_DARK, "ancilla"), _swatch(IDLE, "idle", IDLE_EDGE)])
    for gi, sg in enumerate(groups):
        ax_g, ax_s = axes[gi]
        measured = {m["id"] for m in sg.measurements}
        nbrs = {nb for m in sg.measurements for nb in m["neighbors"]} - measured
        fill = {**{v: MPP_MID for v in nbrs}, **{v: MPP for v in measured}}
        edges = {(min(m["id"], nb), max(m["id"], nb)): MPP
                 for m in sg.measurements for nb in m["neighbors"]}
        _draw_graph(ax_g, G, pos, fill, edges)
        _title(ax_g, f"Tock {gi + 1}", _operators(sg.measurements))

        if sg.switch_gens and len(sg.switch_gens) == n_sw:
            sw_nodes = [list(sg.switch_support[si]) for si in range(n_sw)]
            sw_anc = [len(sg.switch_gens[si]) for si in range(n_sw)]
        else:
            sw_nodes = [list(measured | nbrs)] + [[] for _ in range(n_sw - 1)]
            sw_anc = [len(sg.measurements)] + [0] * (n_sw - 1)
        _draw_switch_diagram(ax_s, nodes_list=nodes, data_fill=fill,
                             sw_data_nodes=sw_nodes, sw_anc_count=sw_anc, k=k,
                             family="mpp")
        _title(ax_s, "Switch wiring", f"{sg.n_ports_used} of {k * n_sw} ports used")
    _save_or_show(fig, save_path, "Case 4 occupancy", display)


def show_qubit_occupancy_case5(
    cz_lim:    "CZSchedulerLimited",
    save_path: str | None = None,
    display:   bool = False,
) -> None:
    """One row per Tock for Case 5: CZ steps wired through the switch."""
    batches = cz_lim.scheduled_batches
    if not batches:
        return
    G = cz_lim.G
    nodes = list(G.nodes)
    pos = _positions(G)
    k, n_sw = cz_lim.switch.k, cz_lim.switch.n_switches
    n_t = 2 * len(batches)
    fig, axes = _per_tock_figure(n_t, max(len(nodes), k + 2), right_ratio=2.0)
    _occupancy_legend(fig, [_swatch(CZ, "active this Tock"),
                            _swatch(CZ_MID, "other end of the CZ"),
                            _swatch(CZ_DARK, "ancilla"), _swatch(IDLE, "idle", IDLE_EDGE)])
    for t in range(n_t):
        ax_g, ax_s = axes[t]
        b, step = divmod(t, 2)
        sb = batches[b]
        now   = {u if step == 0 else v for u, v in sb.edges}
        other = {v if step == 0 else u for u, v in sb.edges}
        fill = {**{v: CZ_MID for v in other}, **{v: CZ for v in now}}
        _draw_graph(ax_g, G, pos, fill,
                    {(min(u, v), max(u, v)): CZ for u, v in sb.edges})
        _title(ax_g, f"Tock {t + 1}",
               f"sub-batch {b + 1}, step {step + 1}: "
               + (r"$Z_u \otimes Z_{anc}$" if step == 0 else r"$Z_v \otimes Z_{anc}$"))

        per_sw = sb.switch_edges if (sb.switch_edges and len(sb.switch_edges) == n_sw) \
            else [list(sb.edges)] + [[] for _ in range(n_sw - 1)]
        sw_nodes = [[u if step == 0 else v for u, v in e] for e in per_sw]
        sw_anc = [len(e) for e in per_sw]
        _draw_switch_diagram(ax_s, nodes_list=nodes, data_fill={v: CZ for v in now},
                             sw_data_nodes=sw_nodes, sw_anc_count=sw_anc, k=k,
                             family="cz")
        _title(ax_s, "Switch wiring",
               f"{len(sb.edges)} CZ gate{'s' if len(sb.edges) != 1 else ''}, "
               f"{sb.n_ports_used} of {k * n_sw} ports used")
    _save_or_show(fig, save_path, "Case 5 occupancy", display)
