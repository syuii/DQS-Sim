"""
figures.py
==========
Purpose-built, page-native result figures, drawn with
one shared visual language so they read as a set:

  fig_c6_substrate.pdf   Case-1 substrate occupancy for C6: data + ancilla
                         strips per Tock, with the ancilla routing interval
                         and hatched blocked positions.
  fig_c8_switches.pdf    One Tock of the Case-4 schedule for C8 on S=2
                         four-port switches: two MPP measurements wired in
                         parallel, one per switch.
  fig_switch_count.pdf   Measured T5 versus switch count S at fixed k=4.

Usage:  python examples/figures.py OUTPUT_DIR
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dqs_sim.schedulers import (CZSchedulerLimited, MPPConflictScheduler,
                                MPPSchedulerLimited)
from dqs_sim.topology import KPortSwitch

mpl.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#5A6473", "axes.labelcolor": "#1A1A1A",
    "axes.titlecolor": "#1A1A1A", "text.color": "#1A1A1A",
    "xtick.color": "#5A6473", "ytick.color": "#5A6473",
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

_W = 6.3

# Shared semantic colours (identical to the schedule heatmaps)
C_MEAS   = "#285A96"  # X-measured        (dark navy, white text)
C_NBR    = "#B8CDE4"  # Z-neighbour       (light navy, black text)
C_IDLE   = "#F2F3F5"  # idle
C_IDLE_E = "#D8DBE0"
C_ANC    = "#8B95A5"  # ancilla patch / ancilla QPU (slate)
C_SLATE  = "#5A6473"
C_SWFILL = "#EDF1F6"  # switch body fill
CASE2    = "#C1853C"  # ochre (Case-2 floors)
CASE1    = "#285A96"


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def qpu_box(ax, x, y, w, h, fill, edge, label, tcol, fs=8, lw=1.1, hatch=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                         facecolor=fill, edgecolor=edge, linewidth=lw,
                         hatch=hatch, zorder=3)
    ax.add_patch(box)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tcol, zorder=4)


# Figure A: C6 Case-1 substrate occupancy

def fig_c6_substrate(path: str) -> None:
    s = SubstrateScheduler(nx.cycle_graph(6))
    _quiet(s.phase_1_reduction)
    _quiet(s.phase_2_mapping)
    s.schedule = _quiet(s.phase_3_scheduling)
    n = len(s.G)

    fig, axes = plt.subplots(len(s.schedule), 1, figsize=(_W, 3.4),
                             facecolor="white")

    for t, (ax, tock) in enumerate(zip(axes, s.schedule)):
        b = tock[0]  # one measurement per Tock
        meas = s.mapping[b["id"]]
        nbrs = {s.mapping[v] for v in b["neighbors"]}
        L, R = b["L"], b["R"]

        ax.set_xlim(-2.1, n + 0.1)
        ax.set_ylim(-0.25, 2.45)
        ax.axis("off")

        # row labels
        ax.text(-0.25, 1.85, "Ancilla", ha="right", va="center",
                fontsize=8, color=C_SLATE)
        ax.text(-0.25, 0.55, "Data", ha="right", va="center",
                fontsize=8, color=C_SLATE)
        op = (f"$X_{{{meas}}}" +
              "".join(f"Z_{{{p}}}" for p in sorted(nbrs)) + "$")
        ax.text(-2.05, 1.2, f"Tock {t+1}\n{op}", ha="left", va="center",
                fontsize=9, linespacing=1.5)

        for p in range(n):
            # data row
            if p == meas:
                qpu_box(ax, p, 0.1, 0.92, 0.9, C_MEAS, C_MEAS,
                        f"$Q_{p}$\nX", "white", fs=7)
            elif p in nbrs:
                qpu_box(ax, p, 0.1, 0.92, 0.9, C_NBR, "#7FA6CE",
                        f"$Q_{p}$\nZ", "#1A1A1A", fs=7)
            else:
                qpu_box(ax, p, 0.1, 0.92, 0.9, C_IDLE, C_IDLE_E,
                        f"$Q_{p}$", "#AEB4BE", fs=7)
            # ancilla row
            if L <= p <= R:
                busy_below = (p == meas or p in nbrs)
                qpu_box(ax, p, 1.4, 0.92, 0.9, C_ANC, C_SLATE,
                        f"$A_{p}$", "white", fs=7,
                        hatch=None if busy_below else "///")
            else:
                qpu_box(ax, p, 1.4, 0.92, 0.9, C_IDLE, C_IDLE_E,
                        f"$A_{p}$", "#AEB4BE", fs=7)

        # routing interval bracket above the ancilla row
        ax.plot([L + 0.06, R + 0.86], [2.42, 2.42], color=C_SLATE,
                linewidth=1.2)
        ax.plot([L + 0.06, L + 0.06], [2.30, 2.42], color=C_SLATE, linewidth=1.2)
        ax.plot([R + 0.86, R + 0.86], [2.30, 2.42], color=C_SLATE, linewidth=1.2)
        ax.text((L + R + 0.92) / 2, 2.58, f"ancilla interval $[{L},{R}]$",
                ha="center", va="bottom", fontsize=7.5, color=C_SLATE)

    # shared legend
    import matplotlib.patches as mpatches
    fig.legend(handles=[
        mpatches.Patch(facecolor=C_MEAS, edgecolor=C_MEAS, label="measured ($X$)"),
        mpatches.Patch(facecolor=C_NBR, edgecolor="#7FA6CE", label="neighbour ($Z$)"),
        mpatches.Patch(facecolor=C_ANC, edgecolor=C_SLATE, label="ancilla patch"),
        mpatches.Patch(facecolor=C_ANC, edgecolor=C_SLATE, hatch="///",
                       label="blocked (data idle)"),
    ], fontsize=7.5, frameon=False, ncol=4, loc="lower center",
       bbox_to_anchor=(0.54, -0.015), handlelength=1.2, columnspacing=1.0)

    fig.tight_layout(rect=[0, 0.045, 1, 1], h_pad=2.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("intervals:",
          [(s.mapping[tock[0]['id']], tock[0]['L'], tock[0]['R'])
           for tock in s.schedule])


# Figure B: C8 Case-4, one Tock on two switches

def fig_c8_switches(path: str) -> None:
    G = nx.cycle_graph(8)
    m = MPPSchedulerLimited(G, KPortSwitch(k=4, n_elus=16, n_switches=2))
    _quiet(m.schedule)
    g0 = m.scheduled_groups[0]

    fig, ax = plt.subplots(figsize=(_W, 2.9), facecolor="white")
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-0.6, 6.4)
    ax.axis("off")

    n = 8
    dx, dw = 2.0, 1.3  # data spacing / width
    data_y = 5.0

    roles = {}  # vertex -> ('X'|'Z', switch idx)
    for si, gens in enumerate(g0.switch_gens):
        for gen in gens:
            roles[gen["id"]] = ("X", si)
            for v in gen["neighbors"]:
                roles[v] = ("Z", si)

    # data QPUs
    centres = {}
    for v in range(n):
        x = 0.2 + v * dx
        centres[v] = x + dw / 2
        if v in roles:
            r, _ = roles[v]
            if r == "X":
                qpu_box(ax, x, data_y, dw, 1.0, C_MEAS, C_MEAS,
                        f"$q_{v}$", "white", fs=9)
            else:
                qpu_box(ax, x, data_y, dw, 1.0, C_NBR, "#7FA6CE",
                        f"$q_{v}$", "#1A1A1A", fs=9)
        else:
            qpu_box(ax, x, data_y, dw, 1.0, C_IDLE, C_IDLE_E,
                    f"$q_{v}$", "#AEB4BE", fs=9)
    ax.text(0.2, 6.25, "data QPUs", fontsize=8, color=C_SLATE)

    # two switches
    sw_w, sw_h, sw_y = 5.6, 1.15, 2.3
    sw_x = [1.4, 9.4]
    port_xs = []
    for si in range(2):
        x0 = sw_x[si]
        box = FancyBboxPatch((x0, sw_y), sw_w, sw_h, boxstyle="round,pad=0.03",
                             facecolor=C_SWFILL, edgecolor=C_SLATE,
                             linewidth=1.3, zorder=2)
        ax.add_patch(box)
        ax.text(x0 + sw_w / 2, sw_y + sw_h / 2,
                f"switch {si+1}  ($k=4$)", ha="center", va="center",
                fontsize=8.5, color=C_SLATE, fontweight="bold")
        # exactly k = 4 ports: three data ports on top, one ancilla port below
        pxs = [x0 + sw_w * f for f in (0.14, 0.38, 0.62, 0.86)]
        port_xs.append(pxs)
        for px in pxs[:3]:
            ax.plot(px, sw_y + sw_h, "o", ms=4.5, color=C_SLATE, zorder=4)
        ax.plot(pxs[3], sw_y, "o", ms=4.5, color=C_SLATE, zorder=4)

    # wires: support qubits -> top ports, ancilla -> bottom port
    for si, gens in enumerate(g0.switch_gens):
        sup = sorted(g0.switch_support[si])
        for j, v in enumerate(sup):
            col = C_MEAS if roles[v][0] == "X" else "#7FA6CE"
            ax.plot([centres[v], port_xs[si][j]], [data_y, sw_y + sw_h],
                    color=col, linewidth=1.3, zorder=1)
        # ancilla
        ax_x = sw_x[si] + sw_w / 2
        qpu_box(ax, ax_x - 0.65, 0.15, 1.3, 1.0, C_ANC, C_SLATE,
                f"$a_{si}$", "white", fs=9)
        ax.plot([ax_x, port_xs[si][3]], [1.15, sw_y],
                color=C_SLATE, linewidth=1.3, linestyle="--", zorder=1)
        op = gens[0]
        lbl = (f"$X_{{{op['id']}}}" +
               "".join(f"Z_{{{v}}}" for v in sorted(op["neighbors"])) + "$")
        ax.text(ax_x + 1.0, 0.62, lbl, ha="left", va="center", fontsize=8.5)
    ax.text(0.2, -0.5, "ancilla QPUs", fontsize=8, color=C_SLATE)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Figure C: T5 versus switch count

def fig_switch_count(path: str) -> None:
    from matplotlib.ticker import MaxNLocator
    graphs = [("$C_{12}$", nx.cycle_graph(12), CASE1, "o"),
              ("$K_8$", nx.complete_graph(8), CASE2, "s")]
    Ss = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(0.62 * _W, 2.6), facecolor="white")
    for lbl, G, col, mk in graphs:
        t5, floor = [], None
        for S in Ss:
            c = CZSchedulerLimited(G, KPortSwitch(k=4, n_elus=2 * len(G),
                                                  n_switches=S))
            _quiet(c.schedule_batches)
            t5.append(c.n_tocks())
            floor = 2 * c.chromatic_index
        ax.plot(Ss, t5, "-" + mk, color=col, markersize=4.5, linewidth=1.4,
                label=lbl, zorder=3)
        ax.axhline(floor, color=col, linestyle=":", linewidth=1.0,
                   alpha=0.65, zorder=1)
    ax.set_xlabel("Switch count $S$  (at $k=4$)", fontsize=8.5)
    ax.set_ylabel("Tocks", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.25, zorder=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Single-panel schedule heatmaps and port charts (one PDF per case)

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

from dqs_sim.schedulers import CZScheduler, CZSchedulerFixedAncilla

MPP_CMAP = ListedColormap([C_IDLE, C_NBR, C_MEAS])
CZ_CMAP  = ListedColormap([C_IDLE, "#E6C79A", "#B0743C"])

MPP_LEG = [mpatches.Patch(facecolor=C_IDLE, edgecolor=C_IDLE_E, label="idle"),
           mpatches.Patch(facecolor=C_NBR,  label="neighbour ($Z$)"),
           mpatches.Patch(facecolor=C_MEAS, label="measured ($X$)")]
CZ_LEG  = [mpatches.Patch(facecolor=C_IDLE, edgecolor=C_IDLE_E, label="idle"),
           mpatches.Patch(facecolor="#E6C79A", label="CZ step 1"),
           mpatches.Patch(facecolor="#B0743C", label="CZ step 2")]


def heat_panel(mat, xlabels, ylabels, title, cmap, legend, path,
               width, vline_every=0):
    """One self-contained heatmap panel in the style of the line charts."""
    n_q, n_t = mat.shape
    fig, ax = plt.subplots(figsize=(width, 0.34 * n_q + 1.15),
                           facecolor="white")
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=cmap.N - 1,
              interpolation="nearest", origin="upper")
    ax.set_xticks(np.arange(-0.5, n_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_q, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    if vline_every:
        for x in range(vline_every - 1, n_t - 1, vline_every):
            ax.axvline(x + 0.5, color=C_SLATE, linewidth=0.9)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    for s in ax.spines.values():
        s.set_color(C_IDLE_E)
    ax.legend(handles=legend, fontsize=7, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              handlelength=1.1, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def port_panel(ports_used, k, title, xlabel, path, width, color):
    fig, ax = plt.subplots(figsize=(width, 2.1), facecolor="white")
    x = np.arange(1, len(ports_used) + 1)
    ax.bar(x, ports_used, width=0.62, color=color, edgecolor="white",
           linewidth=0.6, zorder=2)
    for xi, p in zip(x, ports_used):
        ax.text(xi, p + 0.12, str(p), ha="center", va="bottom", fontsize=7.5)
    ax.axhline(k, color=C_SLATE, linewidth=1.2, linestyle="--",
               label=f"$k={k}$ ports", zorder=3)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("Ports used", fontsize=8.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.set_xticks(x)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, k * 1.38)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _mat_mpp(schedule, mapping, n):
    # Rows are vertex ids (consistent with the CZ panels), not substrate slots.
    m = np.zeros((n, len(schedule)), dtype=int)
    for t, tock in enumerate(schedule):
        for b in tock:
            m[b["id"], t] = 2
            for v in b["neighbors"]:
                if m[v, t] == 0:
                    m[v, t] = 1
    return m


def _mat_cz(batches, n):
    m = np.zeros((n, 2 * len(batches)), dtype=int)
    for i, batch in enumerate(batches):
        for (u, v) in batch:
            m[u, 2 * i] = 1
            m[v, 2 * i + 1] = 2
    return m


def _mat_case4(groups, n):
    m = np.zeros((n, len(groups)), dtype=int)
    for t, g in enumerate(groups):
        for gen in g.measurements:
            m[gen["id"], t] = 2
            for v in gen["neighbors"]:
                if m[v, t] == 0:
                    m[v, t] = 1
    return m


def panels_c6(out: str) -> None:
    G = nx.cycle_graph(6)
    n = 6
    HALF, FULL = 3.05, 6.3

    s = MPPConflictScheduler(G)
    _quiet(s.schedule_mpp)
    ylab = [f"$Q_{i}$" for i in range(n)]
    m1 = _mat_mpp(s.schedule, s.mapping, n)
    heat_panel(m1, [f"T{t+1}" for t in range(m1.shape[1])], ylab,
               "(a) Case 1 — MPP (3 Tocks)", MPP_CMAP, MPP_LEG,
               os.path.join(out, "c6_heat_c1.pdf"), HALF)

    c2 = CZScheduler(G); _quiet(c2.schedule_cz_rounds)
    m2 = _mat_cz(c2.cz_rounds, n)
    heat_panel(m2, [f"T{t+1}" for t in range(m2.shape[1])], ylab,
               "(b) Case 2 — CZ (4 Tocks)", CZ_CMAP, CZ_LEG,
               os.path.join(out, "c6_heat_c2.pdf"), HALF, vline_every=2)

    c3 = CZSchedulerFixedAncilla(G, k=1); _quiet(c3.schedule_batches)
    m3 = _mat_cz(c3.batches, n)
    heat_panel(m3, [f"T{t+1}" for t in range(m3.shape[1])], ylab,
               "(c) Case 3 — CZ, $k_a=1$ (12 Tocks)", CZ_CMAP, CZ_LEG,
               os.path.join(out, "c6_heat_c3.pdf"), FULL, vline_every=2)

    sw = KPortSwitch(k=4, n_elus=2 * n, n_switches=1)
    m4s = MPPSchedulerLimited(G, sw); _quiet(m4s.schedule)
    m4 = _mat_case4(m4s.scheduled_groups, n)
    heat_panel(m4, [f"G{t+1}" for t in range(m4.shape[1])], ylab,
               "(a) Case 4 — MPP, switch (3 Tocks)", MPP_CMAP, MPP_LEG,
               os.path.join(out, "c6_heat_c4.pdf"), HALF)

    c5 = CZSchedulerLimited(G, sw); _quiet(c5.schedule_batches)
    m5 = _mat_cz([b.edges for b in c5.scheduled_batches], n)
    heat_panel(m5, [f"T{t+1}" for t in range(m5.shape[1])], ylab,
               "(b) Case 5 — CZ, switch (8 Tocks)", CZ_CMAP, CZ_LEG,
               os.path.join(out, "c6_heat_c5.pdf"), HALF, vline_every=2)

    port_panel([g.n_ports_used for g in m4s.scheduled_groups], 4,
               "(a) Case 4 — MPP", "Group",
               os.path.join(out, "c6_port_c4.pdf"), HALF, "#9C5A8C")
    port_panel([b.n_ports_used for b in c5.scheduled_batches], 4,
               "(b) Case 5 — CZ", "Sub-batch",
               os.path.join(out, "c6_port_c5.pdf"), HALF, "#7FA6CE")


def panels_k5(out: str) -> None:
    G = nx.complete_graph(5)
    n = 5
    HALF, FULL = 3.05, 6.3
    ylab = [f"$Q_{i}$" for i in range(n)]

    s = MPPConflictScheduler(G)
    _quiet(s.schedule_mpp)
    m1 = _mat_mpp(s.schedule, s.mapping, n)
    heat_panel(m1, [f"T{t+1}" for t in range(m1.shape[1])], ylab,
               "(a) Case 1 — MPP (4 Tocks)", MPP_CMAP, MPP_LEG,
               os.path.join(out, "k5_heat_c1.pdf"), HALF)

    c2 = CZScheduler(G); _quiet(c2.schedule_cz_rounds)
    m2 = _mat_cz(c2.cz_rounds, n)
    heat_panel(m2, [f"T{t+1}" for t in range(m2.shape[1])], ylab,
               "(b) Cases 2, 3, 5 — CZ (14 Tocks)", CZ_CMAP, CZ_LEG,
               os.path.join(out, "k5_heat_c2.pdf"), FULL, vline_every=2)

    sw = KPortSwitch(k=4, n_elus=2 * n, n_switches=1)
    c5 = CZSchedulerLimited(G, sw); _quiet(c5.schedule_batches)
    port_panel([b.n_ports_used for b in c5.scheduled_batches], 4,
               "Case 5 — CZ port usage", "Sub-batch",
               os.path.join(out, "k5_port_c5.pdf"), 0.55 * _W, "#7FA6CE")


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "Results/sweeps"
    os.makedirs(out, exist_ok=True)
    fig_c6_substrate(os.path.join(out, "fig_c6_substrate.pdf"))
    fig_c8_switches(os.path.join(out, "fig_c8_switches.pdf"))
    fig_switch_count(os.path.join(out, "fig_switch_count.pdf"))
    panels_c6(out)
    panels_k5(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
