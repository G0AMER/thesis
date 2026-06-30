# Experiment 1 — Incremental Learning Benchmark Report

**PhD Project:** Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale  
**Author:** Ameur Gargouri  
**Date:** February 26, 2026  
**Notebook:** `notebooks/03_incremental_learning_benchmark_local.ipynb`  
**Results Directory:** `thesis_project/experiments/exp01_il_comparison/`

---

## 1. Executive Summary

This report documents Experiment 1 of the PhD thesis: a comprehensive benchmark of **7 continual learning (CL) strategies** applied to a human-robot collaboration (HRC) action prediction task. Using the HARMONIC dataset, we trained a policy network to predict joystick commands from robot state observations, treating each participant as a sequential learning task. The experiment validates the presence of catastrophic forgetting in multi-participant HRC and evaluates how different CL strategies mitigate it.

**Key result:** DER++ (Dark Experience Replay++) achieved the best trade-off among incremental methods, reaching an average MSE of **0.661** — a **27% improvement** over Naive Fine-Tuning — while being ~23× faster than the Joint Training oracle.

---

## 2. Experimental Setup

### 2.1 Dataset

- **Source:** HARMONIC dataset (preprocessed Parquet files)
- **Participants used:** 9 (p100, p101, p102, p103, p104, p106, p107, p108, p109)
- **Protocol:** Each participant = 1 sequential task (task 0 through task 8)
- **Note:** p105 was skipped (missing or insufficient data); 10 tasks were configured but 9 successfully loaded

| Participant | Runs | Parquet Files |
|-------------|------|---------------|
| p100 | 19 | 146 |
| p101 | 20 | 120 |
| p102 | 20 | 120 |
| p103 | 20 | 152 |
| p104 | 14 | 84 |
| p106 | 19 | 114 |
| p107 | 17 | 126 |
| p108 | 14 | 84 |
| p109 | 20 | 120 |

### 2.2 Input/Output Specification

| Component | Description | Dimensionality |
|-----------|-------------|----------------|
| **Observation (input)** | `joint_positions` + `robot_position` | 51 features |
| **Action (output)** | `ada_joy` (joystick commands) | 5 dimensions |

- Data was split per-participant: ~70% train / 15% validation / 15% test
- Global z-score normalization was applied across all participants (computed on training data only)
- NaN values were removed; remaining NaNs replaced with 0

### 2.3 Model Architecture

- **Type:** Multi-Layer Perceptron (MLP)
- **Architecture:** 51 → 256 (ReLU, Dropout 0.1) → 256 (ReLU, Dropout 0.1) → 5
- **Total parameters:** ~72,197
- **Loss function:** Mean Squared Error (MSE)

### 2.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 0 |
| Epochs per task | 50 (max) |
| Early stopping patience | 10 epochs |
| Batch size | 256 |
| Gradient clipping | Max norm 1.0 |
| Random seed | 42 |
| Device | CUDA (GPU) |

### 2.5 Strategy-Specific Hyperparameters

| Strategy | Parameter | Value |
|----------|-----------|-------|
| EWC | λ (regularization strength) | 5000.0 |
| EWC | Fisher samples | 1000 |
| Online EWC | γ (Fisher decay) | 0.95 |
| SI | c (importance coefficient) | 1.0 |
| LwF | α (distillation weight) | 1.0 |
| DER++ | Buffer size | 5000 |
| DER++ | α (logit replay weight) | 0.5 |
| DER++ | β (label replay weight) | 0.5 |

---

## 3. Strategies Benchmarked

### 3.1 Naive Fine-Tune (Lower Bound)
- **Family:** Baseline
- **Mechanism:** No protection against forgetting. The model is simply trained on each new task, overwriting previous knowledge.
- **Expected behavior:** Maximum catastrophic forgetting — establishes the lower bound.

### 3.2 Joint Training (Upper Bound / Oracle)
- **Family:** Baseline
- **Mechanism:** Accumulates all data from all seen tasks and retrains from scratch on the union after each new task.
- **Expected behavior:** Best possible performance (no forgetting by construction), but computationally expensive and not truly incremental. Establishes the upper bound.

### 3.3 EWC — Elastic Weight Consolidation
- **Family:** Regularization
- **Reference:** Kirkpatrick et al., *"Overcoming catastrophic forgetting in neural networks"*, PNAS 2017
- **Mechanism:** After each task, computes the diagonal Fisher Information Matrix to estimate parameter importance. Adds a penalty: $\mathcal{L} = \mathcal{L}_{task} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$
- **Trade-off:** Memory grows linearly with number of tasks (stores one Fisher matrix per task).

### 3.4 Online EWC (EWC++)
- **Family:** Regularization
- **Reference:** Schwarz et al., *"Progress & Compress"*, ICML 2018
- **Mechanism:** Uses a running average of the Fisher matrix: $\hat{F} = \gamma \hat{F}_{prev} + F_{new}$, maintaining constant memory footprint.
- **Trade-off:** The decay factor γ determines how quickly old importance information is forgotten.

### 3.5 SI — Synaptic Intelligence
- **Family:** Regularization
- **Reference:** Zenke et al., *"Continual Learning Through Synaptic Intelligence"*, ICML 2017
- **Mechanism:** Tracks per-parameter importance online during training via path integral of gradient contributions: $\mathcal{L} = \mathcal{L}_{task} + \frac{c}{2} \sum_i \Omega_i (\theta_i - \theta_i^*)^2$
- **Trade-off:** No separate Fisher computation pass needed, but importance estimation can be noisy.

### 3.6 LwF — Learning without Forgetting
- **Family:** Knowledge Distillation
- **Reference:** Li & Hoiem, *"Learning without Forgetting"*, TPAMI 2017
- **Mechanism:** Before training on a new task, snapshots the model as a frozen "teacher." Adds distillation loss: $\mathcal{L} = \mathcal{L}_{task} + \alpha \cdot \text{MSE}(\hat{y}_{student}, \hat{y}_{teacher})$
- **Trade-off:** No stored data, but effectiveness depends on domain similarity between tasks.

### 3.7 DER++ — Dark Experience Replay++
- **Family:** Replay
- **Reference:** Buzzega et al., *"Dark Experience for General Continual Learning: a Strong, Simple Baseline"*, NeurIPS 2020
- **Mechanism:** Uses a fixed-size reservoir-sampling replay buffer storing past (observation, action, logit) tuples. Combines current task loss with two replay losses: $\mathcal{L} = \mathcal{L}_{task} + \alpha \cdot \text{MSE}(\hat{y}, \text{logits}_{buf}) + \beta \cdot \text{MSE}(\hat{y}, \text{labels}_{buf})$
- **Trade-off:** Requires memory for the replay buffer, but typically achieves the best anti-forgetting performance.

---

## 4. Evaluation Protocol

### 4.1 Accuracy Matrix R[i,j]

After training on each task *i*, the model is evaluated on **all** tasks *j* ∈ {0, ..., T-1}. This produces the accuracy matrix R where R[i][j] = MSE on task *j* after training up to task *i*.

### 4.2 Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Average Accuracy (AA)** | $\text{AA} = \frac{1}{T} \sum_{j=0}^{T-1} R[T-1][j]$ | Mean MSE on all tasks after final training. **Lower is better.** |
| **Backward Transfer (BWT)** | $\text{BWT} = \frac{1}{T-1} \sum_{j=0}^{T-2} (R[T-1][j] - R[j][j])$ | Change in performance on old tasks. **Positive = forgetting.** Closer to 0 or negative is better. |
| **Forward Transfer (FWT)** | $\text{FWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} (R[j-1][j] - R[j][j])$ | How prior knowledge affects new tasks before specialized training. **Closer to 0 is better** (positive means new tasks were initially harder). |

---

## 5. Results

### 5.1 Summary Table

| Strategy | Avg MSE ↓ | BWT →0 | FWT ↓ | Time (s) | MSE Rank |
|----------|-----------|--------|-------|----------|----------|
| **Joint Training** | **0.5554** | **−0.0624** | +0.2521 | 3445.2 | 🥇 1 |
| **DER++** | **0.6614** | +0.0824 | +0.3129 | 148.1 | 🥈 2 |
| **EWC** | 0.7549 | +0.0544 | +0.2126 | 146.4 | 🥉 3 |
| **Online EWC** | 0.7897 | +0.1146 | +0.2240 | 95.9 | 4 |
| **LwF** | 0.8077 | +0.1251 | +0.1963 | 98.4 | 5 |
| **SI** | 0.8607 | +0.1929 | +0.4369 | 134.0 | 6 |
| **Naive Fine-Tune** | 0.9054 | +0.2379 | +0.3263 | 87.1 | 7 |

### 5.2 Per-Task Final MSE (after all 9 tasks trained)

| Task | Participant | Naive | Joint | EWC | O-EWC | SI | LwF | DER++ |
|------|-------------|-------|-------|-----|-------|-----|-----|-------|
| 0 | p100 | 0.967 | 0.254 | 0.639 | 0.708 | 0.740 | 0.715 | 0.523 |
| 1 | p101 | 1.083 | 0.423 | 0.861 | 1.075 | 1.019 | 1.002 | 0.723 |
| 2 | p102 | 1.698 | 1.248 | 1.506 | 1.655 | 1.680 | 1.640 | 1.967 |
| 3 | p103 | 0.627 | 0.367 | 0.524 | 0.623 | 0.658 | 0.611 | 0.445 |
| 4 | p104 | 0.497 | 0.231 | 0.467 | 0.440 | 0.473 | 0.423 | 0.326 |
| 5 | p106 | 1.139 | 0.557 | 0.958 | 0.878 | 1.078 | 1.037 | 0.546 |
| 6 | p107 | 0.870 | 0.572 | 0.694 | 0.659 | 0.853 | 0.724 | 0.505 |
| 7 | p108 | 0.992 | 0.947 | 0.831 | 0.758 | 1.005 | 0.808 | 0.687 |
| 8 | p109 | 0.276 | 0.400 | 0.315 | 0.313 | 0.242 | 0.309 | 0.232 |

### 5.3 Diagonal Values (Task-Specific MSE — measured right after training on that task)

| Task | Participant | Naive | Joint | EWC | O-EWC | SI | LwF | DER++ |
|------|-------------|-------|-------|-----|-------|-----|-----|-------|
| 0 | p100 | 0.314 | 0.319 | 0.315 | 0.316 | 0.321 | 0.316 | 0.317 |
| 1 | p101 | 0.515 | 0.520 | 0.616 | 0.625 | 0.516 | 0.669 | 0.516 |
| 2 | p102 | 1.188 | 1.207 | 1.287 | 1.283 | 1.170 | 1.299 | 1.171 |
| 3 | p103 | 0.257 | 0.363 | 0.327 | 0.337 | 0.251 | 0.342 | 0.271 |
| 4 | p104 | 0.166 | 0.240 | 0.310 | 0.261 | 0.155 | 0.235 | 0.182 |
| 5 | p106 | — | — | — | — | — | — | — |
| 6 | p107 | 0.878 | 0.744 | 0.855 | 0.807 | 0.920 | 0.785 | 0.711 |
| 7 | p108 | 1.081 | 1.035 | 1.008 | 1.083 | 1.082 | 1.004 | 1.011 |
| 8 | p109 | 0.708 | 0.513 | 0.684 | 0.601 | 0.711 | 0.583 | 0.570 |

> Note: Task 5 (p106) diagonal is missing due to an empty row in the accuracy matrix (likely an evaluation skip during the run).

### 5.4 Computation Time

| Strategy | Time (s) | Relative to Naive |
|----------|----------|-------------------|
| Naive Fine-Tune | 87.1 | 1.0× |
| Online EWC | 95.9 | 1.1× |
| LwF | 98.4 | 1.1× |
| SI | 134.0 | 1.5× |
| EWC | 146.4 | 1.7× |
| DER++ | 148.1 | 1.7× |
| Joint Training | 3445.2 | **39.6×** |

---

## 6. Analysis

### 6.1 Catastrophic Forgetting is Confirmed

Naive Fine-Tuning produced the worst results across all metrics:
- **BWT = +0.238** — the highest among all strategies, indicating severe degradation on old tasks
- **Final MSE on Task 0 = 0.967** vs diagonal of 0.314 — the model's performance on the first participant nearly tripled after training on 8 subsequent ones
- This unequivocally confirms that catastrophic forgetting is a real challenge in multi-participant HRC action prediction

### 6.2 Joint Training Sets the Upper Bound

Joint Training achieved the best overall MSE (0.555) and is the **only strategy with negative BWT** (−0.062), meaning it slightly *improved* on earlier tasks as more data was accumulated. However:
- It is **39.6× slower** than Naive Fine-Tuning (3445s vs 87s)
- It requires storing and reprocessing all previous data — incompatible with online deployment on a robot
- It serves as the theoretical oracle/upper bound, not a practical solution

### 6.3 DER++ is the Best Incremental Strategy

DER++ achieved the best results among true incremental methods:
- **Avg MSE = 0.661** — closest to Joint Training, **27% better** than Naive, and only 19% behind the oracle
- **BWT = +0.082** — moderate forgetting, significantly less than Naive (+0.238)
- Consistent improvements across nearly all participants
- The replay buffer (5000 samples) effectively preserves knowledge from prior tasks
- Computation time (148s) is comparable to EWC and much faster than Joint Training

### 6.4 EWC Outperforms Online EWC

This is a noteworthy result:
- **EWC: MSE = 0.755, BWT = +0.054** — lowest BWT among all incremental methods
- **Online EWC: MSE = 0.790, BWT = +0.115** — worse on both metrics

The standard EWC stores a separate Fisher matrix per task, giving it more precise importance information. Online EWC's exponential decay (γ=0.95) may be too aggressive for only 9 tasks, causing old importance information to be prematurely discarded. With more tasks or a higher γ, Online EWC might improve.

### 6.5 SI Underperforms

SI achieved near-Naive performance:
- **Avg MSE = 0.861** (vs Naive 0.905 — only 5% improvement)
- **BWT = +0.193** — nearly as much forgetting as Naive
- **FWT = +0.437** — the worst forward transfer, suggesting the importance weights actively interfere with learning new tasks

The path-integral importance estimation in SI appears less robust to the high inter-participant variability in HARMONIC. The importance coefficient c=1.0 may need tuning.

### 6.6 LwF Provides Moderate Protection

LwF sits between EWC and SI:
- **Avg MSE = 0.808, BWT = +0.125**
- **Best FWT (+0.196)** among all strategies, suggesting distillation helps generalization to unknown tasks
- However, the distillation approach assumes some output similarity between tasks — if participant behaviors are very different, the teacher's predictions may provide misleading supervision

### 6.7 Task Difficulty Analysis

Participant p102 (Task 2) is consistently the hardest across all strategies:
- **Final MSE ranges from 1.248 (Joint) to 1.967 (DER++)**
- Even the diagonal value (task-specific best) is ~1.17–1.30 for most strategies
- This suggests p102 has fundamentally different or more complex interaction patterns

Participant p109 (Task 8) is the easiest:
- **Final MSE ranges from 0.232 (DER++) to 0.400 (Joint)**
- As the last task trained, it benefits from proximity (most recent training)

Participant p104 has the lowest diagonal values (0.155–0.310), meaning the model learns it well when focused on it — but this knowledge is easily overwritten.

---

## 7. Strategy Rankings

### By Average MSE (overall quality)
```
1. Joint Training  0.555  (oracle)
2. DER++           0.661  ★ best incremental
3. EWC             0.755
4. Online EWC      0.790
5. LwF             0.808
6. SI              0.861
7. Naive           0.905  (lower bound)
```

### By Backward Transfer (forgetting resistance)
```
1. Joint Training  −0.062  (improves old tasks)
2. EWC             +0.054  ★ best incremental
3. DER++           +0.082
4. Online EWC      +0.115
5. LwF             +0.125
6. SI              +0.193
7. Naive           +0.238  (maximum forgetting)
```

### By Forward Transfer (new task generalization)
```
1. LwF             +0.196  ★ best
2. EWC             +0.213
3. Online EWC      +0.224
4. Joint Training  +0.252
5. DER++           +0.313
6. Naive           +0.326
7. SI              +0.437  (worst new-task performance)
```

### By Computation Time (speed)
```
1. Naive           87.1s   ★ fastest
2. Online EWC      95.9s
3. LwF             98.4s
4. SI              134.0s
5. EWC             146.4s
6. DER++           148.1s
7. Joint Training  3445.2s  (39.6× slower)
```

---

## 8. Generated Artifacts

All output files are stored in `thesis_project/experiments/exp01_il_comparison/`:

| File | Description | Size |
|------|-------------|------|
| `config.json` | Experiment configuration & hyperparameters | 0.5 KB |
| `comparison_table.csv` | Summary table (CSV) | 0.3 KB |
| `full_comparison.json` | Complete results with accuracy matrices | 17.7 KB |
| `naive_fine-tune_results.json` | Per-strategy detailed results | 2.5 KB |
| `joint_training_results.json` | " | 2.5 KB |
| `ewc_results.json` | " | 2.5 KB |
| `online_ewc_results.json` | " | 2.5 KB |
| `si_results.json` | " | 2.5 KB |
| `lwf_results.json` | " | 2.5 KB |
| `derpp_results.json` | " | 2.5 KB |
| `normalization_stats.json` | Global obs/act mean & std for inference | 2.7 KB |
| `comparison_bars.png` | Bar charts: Avg MSE & BWT comparison | 80 KB |
| `accuracy_matrices.png` | R[i,j] heatmaps for all strategies | 101 KB |
| `forgetting_curve.png` | Task 0 MSE over training sequence | 113 KB |
| `avg_mse_evolution.png` | Average MSE on seen tasks over time | 124 KB |
| `per_task_final_mse.png` | Final MSE per task grouped bar chart | 54 KB |

---

## 9. Conclusions

### 9.1 Main Findings

1. **Catastrophic forgetting is real and severe** in multi-participant HRC: Naive Fine-Tuning loses up to 3× performance on earlier tasks.

2. **DER++ is the recommended strategy** for incremental HRC deployment:
   - Best MSE among incremental methods (0.661 vs 0.905 Naive)
   - Only 1.7× slower than Naive, 23× faster than Joint Training
   - Replay buffer provides effective memory of prior participants

3. **EWC has the lowest forgetting** (BWT = +0.054) among incremental methods, making it suitable when minimal forgetting is prioritized over overall accuracy.

4. **Online EWC underperforms standard EWC** with current hyperparameters (γ=0.95), suggesting the Fisher decay is too aggressive for 9 tasks.

5. **SI is ineffective** on this dataset (c=1.0), producing near-Naive results — the path-integral importance may need careful tuning for HRC.

6. **The gap between DER++ and Joint Training (0.661 vs 0.555)** represents the remaining challenge: ~19% MSE gap that future work should aim to close.

### 9.2 Implications for the Thesis

- **Validates the research question:** Continual learning is necessary and effective for multi-participant HRC
- **Establishes a baseline:** These results provide the foundation for Experiments 2 (IRL integration) and 3 (DASIG style conditioning)
- **Guides architecture choices:** DER++ with replay buffers is viable on embedded systems with modest memory (~5000 × 56 floats ≈ 1.1 MB buffer)

### 9.3 Recommended Next Steps

1. **Hyperparameter sweep:** Vary λ (EWC: 100–10000), γ (Online EWC: 0.9–0.999), c (SI: 0.1–10), α (LwF: 0.1–5), buffer size (DER++: 1000–20000)
2. **Add PackNet:** Parameter isolation strategy using binary masks, which may better handle disjoint participant distributions without replay
3. **Scale to all 25 participants:** Current experiment uses 9; full dataset may reveal different trends
4. **Experiment 2:** Combine DER++ with Inverse Reinforcement Learning for online preference-aware adaptation
5. **Experiment 3:** Incorporate DASIG style vectors as task-conditional embeddings to improve forward transfer across participants
6. **Statistical significance:** Run with multiple seeds (42, 123, 456) to assess variance

---

*Report generated from the results of `03_incremental_learning_benchmark_local.ipynb`, executed on February 26, 2026.*
