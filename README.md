# DQS-Sim

**Distributed graph-state preparation simulator** for fault-tolerant quantum computing.

DQS-Sim estimates the cost of preparing an arbitrary fault-tolerant graph state on a
distributed architecture, where many quantum processing units (QPUs), each holding one
logical qubit as a surface-code patch, are linked through a photonic optical switch.
It is built on Litinski's *Game of Surface Codes* (GoSC) / lattice-surgery model and
reports, for any target graph:

- **time cost** in Tocks (schedule depth),
- **peak ancilla demand**, and
- **total physical qubit count** for a given code distance.

Developed for my master's thesis in the AQUA lab (Prof. Rodney Van Meter), Keio University SFC, 2026.

---

## Why

Measurement-based quantum computing consumes fault-tolerant graph states, and preparing
them can dominate the cost of a computation. Prior work (the Substrate Scheduler,
Liu et al.) scheduled this preparation with multi-qubit Pauli product measurements (MPP)
on a single monolithic processor. In a distributed machine the cost depends on several
interacting choices:

1. **Entangling operation:** MPP stabilizer measurements vs. pairwise CZ gates
2. **Ancilla QPU pool size**
3. **Switch connectivity:** how many ports the optical switch has

DQS-Sim turns these trade-offs into concrete numbers.

## The five cases

| Case | Operation | Constraint | Schedule | Depth |
|------|-----------|------------|----------|-------|
| 1 | MPP | unlimited ancillas | MIS reduction + conflict-graph colouring | ≥ ω(conflict graph) |
| 2 | CZ  | unlimited ancillas | edge colouring | 2 · χ′(G) |
| 3 | CZ  | fixed ancilla pool *k*ₐ | colour classes split into batches of *k*ₐ | 2 · Σᵢ ⌈\|Cᵢ\| / kₐ⌉ |
| 4 | MPP | *k*-port switch | Case 1, each measurement needs Δ+2 ports | infeasible if a generator needs > *k* ports |
| 5 | CZ  | *k*-port switch | at most ⌊k/2⌋ CZ gates per round | 2 · Σᵢ ⌈\|Cᵢ\| / ⌊k/2⌋⌉ |

Each ELU (elementary logic unit) holds one distance-*d* surface-code patch.

## Main findings

- **Neither operation dominates.** MPP is faster on common graph families
  (paths, cycles, stars, grids, cliques), e.g. ×14 on an 8-vertex star and ×2 on
  square grids at every size tested.
- **CZ wins on projective-plane incidence graphs**, where measurement conflicts grow
  faster than vertex degree. CZ is guaranteed to be no slower once
  ω(C) ≥ 2 · χ′(G).
- **MPP needs at least Δ+2 switch ports**, so on narrow switches it is often infeasible
  while CZ (2 ports per gate) still runs.

## Install

```bash
git clone https://github.com/syuii/DQS-Sim.git
cd DQS-Sim
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

```bash
# Defaults: star graph, n = 10, Cases 1 + 2
python main.py

# All cases on a 3x3 grid (2D cluster state)
python main.py --graph grid --n 3 --case all

# k-port switch (Cases 4 and 5)
python main.py --graph cycle --n 8 --case all --switch-k 4

# Misra-Gries (≤ Δ+1) edge colouring instead of greedy
python main.py --graph star --n 8 --case all --colouring vizing

# Your own graph from an edge-list file
python main.py --graph-file examples/graphs/my_graph.txt --case all

# Large graph, skip verification, save figures + JSON report
python main.py --graph random --n 200 --no-verify --save-all Results/
```

Run `python main.py --help` for all options.

### Main options

| Flag | Description | Default |
|------|-------------|---------|
| `--graph` | path, cycle, star, complete, grid, random, petersen, hypercube | `star` |
| `--n` | node count (grid = n×n, hypercube = 2ⁿ) | `10` |
| `--p` | edge probability for `random` | `0.3` |
| `--graph-file` | edge-list file (one `u v` pair per line) | – |
| `--case` | `1`–`5`, `both` (1+2), `all`, `limited` (4+5) | `both` |
| `--ancillas K` | ancilla pool size for Case 3 | auto |
| `--switch-k K` | switch port count for Cases 4 and 5 | – |
| `--colouring` | `greedy` or `vizing` (Misra–Gries) | `greedy` |
| `--mis` | maximum independent set: `auto`, `exact`, `greedy` | `auto` |
| `--code-distance D` | surface-code distance | `7` |
| `--no-verify` | skip bound checks (faster on large graphs) | off |
| `--save-all DIR` | save all figures and `report.json` | – |
| `--export FILE.json` | export the resource report | – |

## Correctness checks

- Every schedule is checked at runtime against theoretical bounds:
  ω(C) ≤ T₁ for MPP, and Δ ≤ χ′ ≤ 2Δ−1 for greedy colouring.
- Cases 3 and 5 are implemented independently and must agree when kₐ = ⌊k/2⌋.
- `examples/mg_validation/test_mg.py` checks that the Misra–Gries colouring is proper,
  complete and uses at most Δ+1 colours.
- Experiments are seeded and reproducible.

```bash
python examples/mg_validation/test_mg.py
```

## Project structure

```
DQS-Sim/
├── main.py                  # CLI entry point
├── dqs_sim/
│   ├── schedulers.py        # Cases 1–5, edge colouring, MIS
│   ├── cost.py              # ResourceReport, physical-qubit formulae
│   ├── topology.py          # k-port switch model
│   └── visualizer.py        # Gantt, heatmaps, 2D layout, occupancy, dashboard
├── examples/                # demo runs, figures and sweeps, validation
├── benchmarks/              # scalability benchmarks
├── Results/                 # generated figures and reports
└── legacy/                  # earlier single-file prototype (kept for reference)
```

## Limitations and future work

- Greedy edge colouring can use up to 2Δ−1 colours; use `--colouring vizing` for ≤ Δ+1.
- Single optical switch; multi-switch network fabrics are future work.
- Clifford operations only (no magic-state distillation cost).
- Static compilation: the whole graph state is built before any measurement.
  Dynamic generation and measurement is the natural next step.

## Citation

If you use DQS-Sim, please cite:

> Gordon Tianyu Cui, *Cost Estimator for Distributed Fault-Tolerant Graph State Preparation*,
> Master's thesis, Graduate School of Media and Governance, Keio University, 2026.

## License

MIT. See [LICENSE](LICENSE).
