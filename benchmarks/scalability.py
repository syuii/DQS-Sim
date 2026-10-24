"""
benchmarks/scalability.py - runtime scaling study for DQS-Sim.

Measures wall-clock scheduling time (no visualization, no verification) as a
function of graph size, for:

  case1        - MPP pipeline, greedy MIS + conflict-graph colouring
  case1_exact  - MPP pipeline with exact branch-and-bound MIS
  case2        - CZ greedy edge colouring
  case2_vizing - CZ Misra-Gries (guaranteed Delta+1) edge colouring

Each task runs in a subprocess with a timeout, so exponential blowups and
memory walls are recorded as 'timeout' rather than hanging the sweep.

Usage:
  python benchmarks/scalability.py run    [--timeout 300] [--out Results/benchmarks]
  python benchmarks/scalability.py plot   [--out Results/benchmarks]
  python benchmarks/scalability.py worker '<json task>'     (internal)
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DEFAULT_OUT = os.path.join(ROOT, "Results", "benchmarks")

# graph builders

def build_graph(family: str, n: int):
    import networkx as nx
    if family == "cycle":
        return nx.cycle_graph(n)
    if family == "grid":  # n = side; graph has n*n nodes
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(n, n))
    if family == "random":  # sparse, average degree ~4
        return nx.erdos_renyi_graph(n, 4.0 / n, seed=42)
    if family == "complete":
        return nx.complete_graph(n)
    raise ValueError(family)


# worker: run ONE (case, family, n) measurement, print a JSON line

def worker(task: dict) -> dict:
    from dqs_sim.schedulers import CZScheduler, SubstrateScheduler

    family, n, case = task["family"], task["n"], task["case"]
    G = build_graph(family, n)
    res = {**task, "n_nodes": len(G), "n_edges": G.number_of_edges()}

    t0 = time.perf_counter()
    if case in ("case1", "case1_exact"):
        # layout-free distributed Case 1: MIS + conflict-graph colouring
        from dqs_sim.schedulers import mpp_conflict_schedule
        mode = "exact" if case == "case1_exact" else "greedy"
        tocks = mpp_conflict_schedule(G, mis_mode=mode)
        cover = sum(len(t) for t in tocks)
        res.update(tocks=len(tocks), alpha=len(G) - cover)
    elif case in ("case2", "case2_vizing"):
        mode = "vizing" if case == "case2_vizing" else "greedy"
        s = CZScheduler(G, coloring_mode=mode)
        rounds = s.schedule_cz_rounds()
        res.update(tocks=2 * len(rounds), chi_prime=len(rounds))
    else:
        raise ValueError(case)
    res["t_total"] = time.perf_counter() - t0
    return res


# driver

# sizes are the *parameter* passed to the builder (grid: side length)
SIZES = {
    # layout-free MPP: MIS reduction + conflict-graph colouring (DSATUR)
    ("case1", "cycle"):   [50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
    ("case1", "grid"):    [7, 10, 14, 22, 32, 45, 71, 100, 173],  # 49 .. 30k nodes
    ("case1", "random"):  [50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
    # exact MPP: branch-and-bound MIS
    ("case1_exact", "cycle"):  [50, 100, 200, 500, 1000, 2000, 5000],
    ("case1_exact", "grid"):   [7, 10, 14, 22, 32, 45],
    ("case1_exact", "random"): [50, 100, 200, 500, 1000, 2000],
    # CZ colouring: near-linear, push to 100k nodes
    ("case2", "cycle"):   [100, 300, 1000, 3000, 10000, 30000, 100000],
    ("case2", "grid"):    [10, 18, 32, 55, 100, 173, 316],  # 100 .. ~100k nodes
    ("case2", "random"):  [100, 300, 1000, 3000, 10000, 30000, 100000],
    ("case2", "complete"): [50, 100, 200, 300],
    ("case2_vizing", "cycle"):   [100, 300, 1000, 3000, 10000, 30000, 100000],
    ("case2_vizing", "grid"):    [10, 18, 32, 55, 100, 173, 316],
    ("case2_vizing", "random"):  [100, 300, 1000, 3000, 10000, 30000, 100000],
    ("case2_vizing", "complete"): [50, 100, 200, 300],
}


def run(timeout: float, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scalability.json")
    results: list[dict] = []
    skip_larger: set[tuple] = set()

    for (case, family), sizes in SIZES.items():
        for n in sizes:
            if (case, family) in skip_larger:
                results.append({"case": case, "family": family, "n": n,
                                "status": "skipped (smaller size timed out)"})
                continue
            task = {"case": case, "family": family, "n": n}
            cmd = [sys.executable, os.path.abspath(__file__), "worker",
                   json.dumps(task)]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, cwd=ROOT)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    results.append({**task, "status": "error",
                                    "stderr": proc.stderr[-400:]})
                    print(f"{case:13s} {family:8s} n={n:<7d} ERROR")
                    skip_larger.add((case, family))
                    continue
                line = proc.stdout.strip().splitlines()[-1]
                res = json.loads(line)
                res["status"] = "ok"
                res["t_wall"] = wall
                results.append(res)
                print(f"{case:13s} {family:8s} n={n:<7d} "
                      f"nodes={res['n_nodes']:<7d} t={res['t_total']:9.3f}s "
                      f"tocks={res.get('tocks','-')}")
            except subprocess.TimeoutExpired:
                results.append({**task, "status": "timeout", "t_wall": timeout})
                print(f"{case:13s} {family:8s} n={n:<7d} TIMEOUT (> {timeout:.0f}s)")
                skip_larger.add((case, family))
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nWrote {out_path} ({len(results)} rows)")


# plots

FAMILY_LABEL = {"cycle": "cycle $C_n$", "grid": "grid", "random": "random (deg 4)",
                "complete": "complete $K_n$"}
CASE_LABEL = {"case1": "Case 1 (MPP, heuristic)",
              "case1_exact": "Case 1 (MPP, exact MIS)",
              "case2": "Case 2 (CZ, greedy)",
              "case2_vizing": "Case 2 (CZ, Misra--Gries)"}
CASE_COLOR = {"case1": "#285A96", "case1_exact": "#7B2D43",
              "case2": "#B0743C", "case2_vizing": "#4E7E4E"}
FAMILY_MARK = {"cycle": "o", "grid": "s", "random": "^", "complete": "D"}


def plot(out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out_dir, "scalability.json")) as f:
        rows = [r for r in json.load(f) if r.get("status") == "ok"]

    def series(case, family):
        pts = [(r["n_nodes"], r["t_total"]) for r in rows
               if r["case"] == case and r["family"] == family]
        return sorted(pts)

    # figure 1: runtime vs size, heuristic pipeline
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for case in ("case1", "case2", "case2_vizing"):
        for family in ("cycle", "grid", "random"):
            pts = series(case, family)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=FAMILY_MARK[family], ms=4, lw=1.4,
                    color=CASE_COLOR[case],
                    label=f"{CASE_LABEL[case]}, {FAMILY_LABEL[family]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("graph size $n$ (vertices)")
    ax.set_ylabel("scheduling time (s)")
    ax.set_title("(a) heuristic schedulers")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=1, loc="upper left")

    # figure 1b: exact vs heuristic MIS (total scheduling time)
    ax = axes[1]
    for case in ("case1", "case1_exact"):
        for family in ("cycle", "grid", "random"):
            pts = series(case, family)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ys = [max(y, 1e-6) for y in ys]
            ax.plot(xs, ys, marker=FAMILY_MARK[family], ms=4, lw=1.4,
                    color=CASE_COLOR[case],
                    linestyle="-" if case == "case1" else "--",
                    label=f"{'greedy' if case=='case1' else 'exact'} MIS, "
                          f"{FAMILY_LABEL[family]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("graph size $n$ (vertices)")
    ax.set_ylabel("scheduling time (s)")
    ax.set_title("(b) exact vs heuristic MIS")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_scalability.{ext}"), dpi=180)
    print("wrote fig_scalability.{pdf,png}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
        sys.exit(0)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "plot"])
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    if args.mode == "run":
        run(args.timeout, args.out)
    else:
        plot(args.out)
