"""Clean scaling chart: Case 1 and Case 2 defaults only."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "Results", "benchmarks")

BLUE = "#285A96"
ORANGE = "#B0743C"
FAMILY_MARK = {"cycle": "o", "grid": "s", "random": "^"}
FAMILY_LABEL = {"cycle": "cycle", "grid": "grid", "random": "random"}

with open(os.path.join(OUT, "scalability.json")) as f:
    rows = [r for r in json.load(f) if r.get("status") == "ok"]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
for case, color, label in (("case1", BLUE, "Case 1 (MPP)"),
                           ("case2", ORANGE, "Case 2 (CZ)")):
    for family in ("cycle", "grid", "random"):
        pts = sorted((r["n_nodes"], r["t_total"]) for r in rows
                     if r["case"] == case and r["family"] == family)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ys = [max(y, 5e-4) for y in ys]
        ax.plot(xs, ys, marker=FAMILY_MARK[family], ms=5, lw=1.6, color=color,
                label=f"{label}, {FAMILY_LABEL[family]}")

ax.axhline(300, color="#5A6473", lw=1, ls=":")
ax.text(9e4, 360, "5-minute budget", fontsize=9, color="#5A6473", ha="right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("graph size $n$ (vertices)")
ax.set_ylabel("scheduling time (s)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_scaling_defaults.{ext}"), dpi=180)
print("wrote fig_scaling_defaults.{pdf,png}")
