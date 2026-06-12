# DQS-Sim

Graph state generation simulator based on the **Game of Surface Codes (GoSC)** / Lattice Surgery framework. Compares three scheduling strategies for fault-tolerant distributed quantum computing, estimating physical qubit cost and circuit depth (Tocks) for arbitrary graph topologies.

Built for the AQUA Lab, Keio University SFC (Prof. Rodney Van Meter).

---

## What it does

Given a graph G = (V, E) representing a target graph state, DQS-Sim simulates three cases:

| Case | Method | Ancillas | Total Tocks |
|------|--------|----------|-------------|
| 1 | MPP stabilizer measurements | Unlimited | Greedy interval schedule |
| 2 | CZ gates, edge colouring | Unlimited | 2 × χ'(G) |
| 3 | CZ gates, edge colouring | Fixed k | 2 × Σᵢ ⌈\|Cᵢ\|/k⌉ |

Each ELU (Elementary Logic Unit) holds one distance-d surface code patch (2d²−1 physical qubits).

---

## Quickstart

```bash
# Install dependencies
pip install networkx matplotlib numpy

# Run with defaults: star(10), d=7, Cases 1+2
python main.py

# Run all 3 cases on a cycle graph, show all figures
python main.py --graph cycle --n 8 --case all --show

# Save all outputs to Results/
python main.py --graph star --n 10 --case all --save-all Results/

# Skip slow verification for large graphs
python main.py --graph random --n 200 --no-verify --save-all Results/
```

See `examples/demo.py` for preset example runs across multiple graph types.

---

## Project structure

```
DQS-Sim/
├── main.py               # CLI entry point
├── dqs_sim/
│   ├── schedulers.py     # Case 1: SubstrateScheduler
│   │                     # Case 2: CZScheduler
│   │                     # Case 3: CZSchedulerFixedAncilla
│   ├── cost.py           # ResourceReport + compute_report()
│   └── visualizer.py     # Gantt, heatmap, 2D layout, occupancy, dashboard
├── examples/
│   └── demo.py           # Preset runs for common graph types
└── Results/              # Output figures and JSON reports
```

---

## CLI reference

### Graph options
| Flag | Description | Default |
|------|-------------|---------|
| `--graph TYPE` | path, cycle, star, complete, grid, random, petersen, hypercube | `star` |
| `--n N` | Node count (grid = n×n, hypercube = 2ⁿ nodes) | `10` |
| `--p P` | Edge probability (random graph only) | `0.3` |

### Simulation options
| Flag | Description | Default |
|------|-------------|---------|
| `--case` | `1`, `2`, `3`, `both` (1+2), `all` (1+2+3) | `both` |
| `--ancillas K` | Ancilla pool size k for Case 3. Auto-selects ⌊peak/2⌋ if omitted with `--case all` | auto |
| `--code-distance D` | Surface code distance d | `7` |
| `--no-verify` | Skip chromatic/Vizing verification (faster for large graphs) | off |

### Output options
| Flag | Description |
|------|-------------|
| `--show` | Display all figures interactively after running |
| `--save-all DIR` | Save all figures + `report.json` to `DIR/{graph}_n{n}/` |
| `--gantt` / `--save-gantt FILE` | Gantt chart of qubit activity |
| `--heatmap` / `--save-heatmap FILE` | Schedule heatmap |
| `--layout` / `--save-layout FILE` | 2D surface-code patch layout |
| `--occupancy` / `--save-occupancy FILE` | Case 1 qubit occupancy strip |
| `--save-occupancy3 FILE` | Case 3 qubit occupancy strip |
| `--dashboard` / `--save-dashboard FILE` | Resource comparison bar chart |
| `--export FILE.json` | Export resource report as JSON |

---

## Output figures

- **Gantt** — horizontal bars showing which qubit is active in which Tock
- **Heatmap** — compact matrix (qubits × Tocks), colour-coded by role
- **2D layout** — ELU patches drawn as physical surface-code tiles
- **Occupancy** — per-Tock tile strip; Case 3 shows ancillas docked above their paired data qubit
- **Dashboard** — bar chart comparing Tocks, peak ancilla ELUs, and total physical qubits across cases
