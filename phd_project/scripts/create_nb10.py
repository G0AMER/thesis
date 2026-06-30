#!/usr/bin/env python3
"""
Create NB10 (Final Benchmarking) from NB09 (CPG-Net Experiment).

Extracts the final comparison cells, removes ablation (v2 vs v3) content,
and renames v3 references to present CPG-Net as the single final algorithm.
"""

import json
import copy
import uuid
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
SRC = NOTEBOOK_DIR / "09_cpg_net_experiment.ipynb"
DST = NOTEBOOK_DIR / "10_final_benchmarking.ipynb"

# Cells to SKIP (0-indexed): cell 8 (ablation markdown), cell 10 (ablation runner), cell 11 (ablation viz)
SKIP_CELLS = {7, 9, 10}  # 0-indexed

# New title markdown for the notebook
NEW_TITLE = """\
# NB10 -- Final Benchmarking: CPG-Net vs Continual Learning Baselines

## Shared Autonomy -- Incremental Learning of Joint-Space Policies

This notebook runs the **final comparison** of our proposed CPG-Net
(Contextual Policy Gating Network) against 5 established continual
learning baselines on the sequential shared-autonomy task.

### Strategies Compared

| # | Strategy | Type | Key Mechanism |
|---|----------|------|---------------|
| 1 | Naive Fine-Tune | Baseline | No CL protection |
| 2 | Joint Training  | Upper bound | Retrains on all data |
| 3 | Online EWC | Regularization | Fisher-weighted penalty |
| 4 | A-GEM | Memory | Gradient projection |
| 5 | DER++ | Replay | Dark knowledge distillation |
| 6 | **CPG-Net (ours)** | **Hybrid** | **FiLM + Replay + Regularization + Multi-Head** |

### CPG-Net Architecture

```
Input -> [Shared Layer 1] -> ReLU -> FiLM(task) -> Dropout
      -> [Shared Layer 2] -> ReLU -> FiLM(task) -> Dropout
      -> [Shared Head + Task Head (blended)] -> Output
```

**Key features:**
1. **FiLM Conditioning** -- per-task feature modulation (gamma * h + beta)
2. **Task-Aware Replay** -- DER++-style buffer with correct FiLM routing
3. **Importance Regularization** -- Fisher-based backbone weight protection
4. **RetroBoost** -- retroactive importance boosting across tasks
5. **FiLM Warm-Start** -- initialize new task FiLM from most similar existing task
6. **Adaptive Replay** -- forgetting-weighted replay sampling
7. **Multi-Head Output** -- per-task output heads blended with shared head
"""

# Cleaned section 7 markdown (run header)
NEW_RUN_HEADER = """\
---
## Run Full Comparison

All 6 strategies (5 baselines + CPG-Net) run sequentially. Each in its own cell.
Metrics: **ACC**, **F** (forgetting), **BWT**, **FWT**, **R²**, **Memory (MB)**.
"""

# Cleaned overfitting markdown (section 21)
NEW_OVERFITTING_HEADER = """\
---
## Overfitting & Learning Anomaly Diagnostics (CPG-Net)

We re-train CPG-Net while recording **train loss** and **val loss** per task per epoch.
Five diagnostic checks are run:
1. **Final train < val** (normal generalization gap)
2. **Loss ratio** final_val / final_train < 3× (no severe overfit)
3. **Monotonic decrease** in val loss (no training instability)
4. **Spike detection** in val loss curve
5. **Early stopping** fired before max epochs
"""


def make_cell_id():
    return uuid.uuid4().hex[:8]


def clean_source(lines):
    """Remove v2/v3 version references from source code lines, keeping class names intact."""
    cleaned = []
    for line in lines:
        # Clean comments and strings — but NOT class/function names like CPGNetV3
        # Remove "v2 vs v3" style comments
        line = line.replace("v2 vs v3", "Base vs Full CPG-Net")
        line = line.replace("v2→v3", "Base→Full")
        line = line.replace("v2 baseline", "base")
        line = line.replace("'v3 (all improvements)'", "'CPG-Net (all features)'")
        line = line.replace('"v3 (all improvements)"', '"CPG-Net (all features)"')
        # Clean print statements referencing versions
        line = line.replace("CPG-Net v3 models defined", "CPG-Net models defined")
        line = line.replace("CPG-Net v3", "CPG-Net")
        line = line.replace("CPG-Net v2", "CPG-Net (base)")
        cleaned.append(line)
    return cleaned


def main():
    with open(SRC) as f:
        nb = json.load(f)

    new_cells = []

    for i, cell in enumerate(nb["cells"]):
        if i in SKIP_CELLS:
            continue

        new_cell = copy.deepcopy(cell)

        # Remove execution outputs and counts
        if new_cell["cell_type"] == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None

        # Generate fresh cell IDs
        if "id" in new_cell:
            new_cell["id"] = make_cell_id()

        # Replace title cell (cell 0)
        if i == 0:
            new_cell["source"] = [l + "\n" for l in NEW_TITLE.split("\n")]
            # Remove trailing newline from last line
            if new_cell["source"][-1] == "\n":
                new_cell["source"] = new_cell["source"][:-1]

        # Replace run header markdown (cell 6, 0-indexed)
        elif i == 6:
            new_cell["source"] = [l + "\n" for l in NEW_RUN_HEADER.split("\n")]
            if new_cell["source"][-1] == "\n":
                new_cell["source"] = new_cell["source"][:-1]

        # Replace overfitting header markdown (cell 20, 0-indexed)
        elif i == 20:
            new_cell["source"] = [l + "\n" for l in NEW_OVERFITTING_HEADER.split("\n")]
            if new_cell["source"][-1] == "\n":
                new_cell["source"] = new_cell["source"][:-1]

        # Clean source code in code cells
        elif new_cell["cell_type"] == "code":
            new_cell["source"] = clean_source(new_cell["source"])

        new_cells.append(new_cell)

    # Reorder: move CPG-Net run cell to be FIRST among the strategy runs.
    # Strategy run cells come right after the CPGNetV3 defs cell.
    # Find CPG-Net run cell (contains "CPG-Net (ours) -- NOVEL METHOD")
    # and the first baseline run cell (contains "Naive Fine-Tune").
    cpg_idx = None
    first_baseline_idx = None
    for j, c in enumerate(new_cells):
        src = "".join(c.get("source", []))
        if "NOVEL METHOD" in src:
            cpg_idx = j
        if "Naive Fine-Tune" in src and first_baseline_idx is None and c["cell_type"] == "code":
            first_baseline_idx = j

    if cpg_idx is not None and first_baseline_idx is not None and cpg_idx > first_baseline_idx:
        cpg_cell = new_cells.pop(cpg_idx)
        new_cells.insert(first_baseline_idx, cpg_cell)
        print(f"  Reordered: CPG-Net run moved from position {cpg_idx+1} to {first_baseline_idx+1}")

    # Build new notebook
    new_nb = copy.deepcopy(nb)
    new_nb["cells"] = new_cells

    with open(DST, "w") as f:
        json.dump(new_nb, f, indent=1, ensure_ascii=False)

    print(f"Created {DST}")
    print(f"  Source cells: {len(nb['cells'])}")
    print(f"  Skipped:      {len(SKIP_CELLS)} (ablation cells {[x+1 for x in sorted(SKIP_CELLS)]})")
    print(f"  Output cells: {len(new_cells)}")

    # Verify: check for syntax errors
    errors = []
    for j, cell in enumerate(new_cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            compile(src, f"cell_{j+1}", "exec")
        except SyntaxError as e:
            errors.append(f"  Cell {j+1}: line {e.lineno}: {e.msg}")

    if errors:
        print("\nSyntax errors found:")
        for e in errors:
            print(e)
    else:
        print("  Syntax:       All code cells OK")

    # Verify: check CPGNetLearnerV3 class exists
    all_src = "".join("".join(c.get("source", [])) for c in new_cells if c["cell_type"] == "code")
    checks = {
        "class CPGNetV3(CPGNet)": "CPGNetV3 model class",
        "class CPGNetLearnerV3(ContinualLearner)": "CPGNetLearnerV3 learner class",
        "def make_cpg_v3_model": "make_cpg_v3_model factory",
        "def run_strategy_v3": "run_strategy_v3 function",
        "all_results": "all_results dict",
    }
    for pattern, desc in checks.items():
        status = "OK" if pattern in all_src else "MISSING"
        print(f"  {status}: {desc}")


if __name__ == "__main__":
    main()
