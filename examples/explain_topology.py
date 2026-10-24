"""
explain_topology.py
===================
Generate an explanatory diagram showing the k-port reconfigurable switch model
used in Cases 4 & 5.

Shows:
  1. The logical graph G (what qubits NEED to interact)
  2. The k-port switch concept (any k of N ELUs connected per Tock)
  3. A worked example of CZ bin-packing (Case 5)
  4. A worked example of MPP bin-packing (Case 4)

Usage:
    python examples/explain_topology.py
    python examples/explain_topology.py --save kport_explained.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Colours
_BG       = "white"
_LOGICAL  = "#4C72B0"  # blue: logical graph
_SWITCH   = "#DD8452"  # orange: switch
_CZ_COL   = "#55A868"  # green: CZ operation
_MPP_COL  = "#C44E52"  # red: MPP operation
_IDLE     = "#F4F4F4"
_BORDER   = "#555555"
_PALETTE  = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _draw_elu_circle(ax, x, y, r, label, color, text_color="white"):
    """Draw a single ELU as a filled circle."""
    circle = plt.Circle((x, y), r, facecolor=color, edgecolor=_BORDER,
                         linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color, zorder=4)


def make_figure(save_path=None):
    fig = plt.figure(figsize=(22, 18), facecolor=_BG)
    fig.suptitle("k-Port Reconfigurable Switch - How Cases 4 & 5 Work",
                 fontsize=16, fontweight="bold", y=0.98)

    # ROW 1 LEFT: Logical graph + explanation
    ax_concept = fig.add_axes([0.03, 0.72, 0.45, 0.22])
    ax_concept.set_xlim(-0.5, 10)
    ax_concept.set_ylim(-1, 5)
    ax_concept.set_aspect("equal")
    ax_concept.axis("off")
    ax_concept.set_title("The k-Port Switch Concept",
                         fontsize=13, fontweight="bold", pad=10)

    # Draw N=8 ELUs in a row
    n_elus = 8
    elu_r = 0.35
    for i in range(n_elus):
        color = _IDLE if i >= 4 else _PALETTE[i % len(_PALETTE)]
        _draw_elu_circle(ax_concept, i * 1.2 + 0.5, 3.5, elu_r,
                         f"E{i}", color,
                         "white" if i < 4 else "#999")

    # Draw switch box
    switch_box = mpatches.FancyBboxPatch(
        (0.0, 0.8), 9.1, 1.8,
        boxstyle="round,pad=0.15",
        facecolor="#FFF3E0", edgecolor=_SWITCH, linewidth=2.5, zorder=1)
    ax_concept.add_patch(switch_box)
    ax_concept.text(4.55, 1.7, "RECONFIGURABLE SWITCH",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=_SWITCH)
    ax_concept.text(4.55, 1.2, "k = 4 ports active per Tock",
                    ha="center", va="center", fontsize=10,
                    color=_SWITCH, fontstyle="italic")

    # Connect first 4 ELUs to switch (active)
    for i in range(4):
        ax_concept.plot([i * 1.2 + 0.5, i * 1.2 + 0.5], [3.15, 2.6],
                        color=_PALETTE[i], linewidth=2.5, zorder=2)
    # Disconnected ELUs (dashed)
    for i in range(4, 8):
        ax_concept.plot([i * 1.2 + 0.5, i * 1.2 + 0.5], [3.15, 2.6],
                        color="#CCCCCC", linewidth=1.5, linestyle="--", zorder=2)

    # Labels
    ax_concept.text(2.15, 4.3, "Connected this Tock (any k of N)",
                    ha="center", fontsize=9, color=_CZ_COL, fontweight="bold")
    ax_concept.text(7.3, 4.3, "Idle this Tock",
                    ha="center", fontsize=9, color="#999")

    ax_concept.text(4.55, 0.3,
                    "Next Tock: switch reconnects to ANY different k ELUs",
                    ha="center", fontsize=9, color="#666", fontstyle="italic")

    # ROW 1 RIGHT: Explanation text
    ax_text = fig.add_axes([0.52, 0.72, 0.45, 0.22])
    ax_text.axis("off")
    explanation = (
        "Cases 1-3: ALL-TO-ALL connectivity (k = N)\n"
        "  Any ELU can talk to any other ELU in 1 Tock.\n\n"
        "Cases 4 & 5: k-PORT SWITCH (k < N)\n"
        "  Each Tock, the switch connects any k of N ELUs.\n"
        "  Those k ELUs have full all-to-all connectivity.\n"
        "  The remaining N-k ELUs are idle that Tock.\n\n"
        "  No fixed wiring, no SWAPs needed.\n"
        "  The challenge is BIN-PACKING: fitting operations\n"
        "  into Tocks where each uses at most k ports.\n\n"
        "  CLI:  --switch-k K    (set port count)\n"
        "  Auto: k defaults to |V|/2 if not specified"
    )
    ax_text.text(0.0, 0.95, explanation, transform=ax_text.transAxes,
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FF",
                           edgecolor="#99AACC", alpha=0.9))

    # ROW 2: Case 5 - CZ Bin-Packing Example
    ax_cz = fig.add_axes([0.03, 0.38, 0.94, 0.28])
    ax_cz.set_xlim(-0.5, 24)
    ax_cz.set_ylim(-1.5, 4.5)
    ax_cz.set_aspect("equal")
    ax_cz.axis("off")
    ax_cz.set_title("Case 5 - CZ Bin-Packing (k=4, floor(k/2)=2 CZ gates per sub-batch)",
                     fontsize=12, fontweight="bold", pad=10)

    # Show 3 sub-batches, each with 2 CZ gates, each taking 2 Tocks
    batch_data = [
        [("Q0", "Q1"), ("Q2", "Q3")],  # Sub-batch 1
        [("Q4", "Q5"), ("Q0", "Q3")],  # Sub-batch 2
        [("Q1", "Q4")],  # Sub-batch 3 (partial)
    ]

    for bi, batch in enumerate(batch_data):
        x_off = bi * 8
        # Sub-batch box
        box_w = 6.5
        box = mpatches.FancyBboxPatch(
            (x_off, -0.5), box_w, 4.0,
            boxstyle="round,pad=0.1",
            facecolor="#F0FFF0" if bi < 2 else "#FFF0F0",
            edgecolor=_CZ_COL, linewidth=1.5, zorder=0)
        ax_cz.add_patch(box)

        n_gates = len(batch)
        ports = n_gates * 2
        ax_cz.text(x_off + box_w / 2, 3.8,
                    f"Sub-batch {bi+1}  ({n_gates} CZ, {ports}/4 ports, 2 Tocks)",
                    ha="center", fontsize=9, fontweight="bold", color=_CZ_COL)

        # Tock labels
        ax_cz.text(x_off + 1.5, 2.8, f"Tock {bi*2+1}", ha="center",
                    fontsize=8, color="#666", fontstyle="italic")
        ax_cz.text(x_off + 1.5, 2.4, "Z_u x Z_anc", ha="center",
                    fontsize=8, color="#666")
        ax_cz.text(x_off + 4.5, 2.8, f"Tock {bi*2+2}", ha="center",
                    fontsize=8, color="#666", fontstyle="italic")
        ax_cz.text(x_off + 4.5, 2.4, "Z_v x Z_anc", ha="center",
                    fontsize=8, color="#666")

        for gi, (qu, qv) in enumerate(batch):
            y = 1.5 - gi * 1.8
            col = _PALETTE[gi % len(_PALETTE)]

            # Step 1: data qubit u
            _draw_elu_circle(ax_cz, x_off + 1.0, y, 0.35, qu, col)
            ax_cz.text(x_off + 1.0, y - 0.55, "data", ha="center",
                        fontsize=7, color="#888")

            # Ancilla between
            _draw_elu_circle(ax_cz, x_off + 2.2, y, 0.3, f"A{gi}", "#FFB74D")
            ax_cz.text(x_off + 2.2, y - 0.55, "anc", ha="center",
                        fontsize=7, color="#888")

            # Arrow from u to anc
            ax_cz.annotate("", xy=(x_off + 1.9, y), xytext=(x_off + 1.38, y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

            # Step 2: data qubit v
            _draw_elu_circle(ax_cz, x_off + 4.0, y, 0.35, qv, col)
            ax_cz.text(x_off + 4.0, y - 0.55, "data", ha="center",
                        fontsize=7, color="#888")

            # Same ancilla
            _draw_elu_circle(ax_cz, x_off + 5.2, y, 0.3, f"A{gi}", "#FFB74D")
            ax_cz.text(x_off + 5.2, y - 0.55, "anc", ha="center",
                        fontsize=7, color="#888")

            # Arrow from v to anc
            ax_cz.annotate("", xy=(x_off + 4.88, y), xytext=(x_off + 4.38, y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    # Summary
    ax_cz.text(12, -1.2,
               "Total: 3 sub-batches x 2 Tocks = 6 Tocks  |  "
               "Case 2 (k=inf): 1 colour class = 2 Tocks per round",
               ha="center", fontsize=10, color="#444", fontstyle="italic")

    # ROW 3: Case 4 - MPP Bin-Packing Example
    ax_mpp = fig.add_axes([0.03, 0.04, 0.94, 0.28])
    ax_mpp.set_xlim(-0.5, 24)
    ax_mpp.set_ylim(-1.5, 4.5)
    ax_mpp.set_aspect("equal")
    ax_mpp.axis("off")
    ax_mpp.set_title("Case 4 - MPP Bin-Packing (k=4, generators must fit in k ports)",
                      fontsize=12, fontweight="bold", pad=10)

    # Show groups of generators
    group_data = [
        # Group 1: g_1 has support {1, 0, 2} (3 ports) - fits alone in k=4
        {"label": "Group 1", "generators": [
            {"id": 1, "support": ["Q1", "Q0", "Q2"], "op": "X1 Z0 Z2"},
        ], "ports": 3},
        # Group 2: g_3 has support {3, 0, 4} (3 ports) - fits alone
        {"label": "Group 2", "generators": [
            {"id": 3, "support": ["Q3", "Q0", "Q4"], "op": "X3 Z0 Z4"},
        ], "ports": 3},
        # Group 3: g_5 has support {5, 4} (2 ports), g_6 has {6, 0} (2 ports) - combined 4 ports, fits!
        {"label": "Group 3", "generators": [
            {"id": 5, "support": ["Q5", "Q4"], "op": "X5 Z4"},
            {"id": 6, "support": ["Q6", "Q0"], "op": "X6 Z0"},
        ], "ports": 4},
    ]

    for gi, group in enumerate(group_data):
        x_off = gi * 8
        box_w = 7.0
        box = mpatches.FancyBboxPatch(
            (x_off, -0.3), box_w, 4.0,
            boxstyle="round,pad=0.1",
            facecolor="#F0F4FF", edgecolor=_MPP_COL, linewidth=1.5, zorder=0)
        ax_mpp.add_patch(box)

        ax_mpp.text(x_off + box_w / 2, 4.0,
                     f"{group['label']}  ({group['ports']}/4 ports, 1 Tock)",
                     ha="center", fontsize=9, fontweight="bold", color=_MPP_COL)

        for mi, gen in enumerate(group["generators"]):
            y = 2.5 - mi * 2.0
            col = _PALETTE[(gi * 2 + mi) % len(_PALETTE)]

            # Draw support qubits
            support = gen["support"]
            for si, sq in enumerate(support):
                x = x_off + 0.8 + si * 1.8
                is_measured = si == 0  # first qubit is X-measured
                fc = col if is_measured else "#90CAF9"
                _draw_elu_circle(ax_mpp, x, y, 0.35, sq, fc)
                role = "X" if is_measured else "Z"
                ax_mpp.text(x, y - 0.55, role, ha="center",
                            fontsize=8, color=col if is_measured else "#4C72B0",
                            fontweight="bold")

            # Operator string
            ax_mpp.text(x_off + box_w / 2, y + 0.7, gen["op"],
                         ha="center", fontsize=9, fontfamily="monospace",
                         color="#444")

    ax_mpp.text(12, -1.2,
                "Generators with disjoint supports share a Tock if combined ELUs <= k  |  "
                "Case 1 (k=inf): all non-conflicting generators run in 1 Tock",
                ha="center", fontsize=10, color="#444", fontstyle="italic")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"  Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None, help="Save to file instead of displaying")
    args = p.parse_args()
    make_figure(save_path=args.save)
