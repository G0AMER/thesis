# Experiment 01 — Shared Autonomy: Incremental Learning Benchmark

**Author:** Ameur Gargouri  
**Date:** 2025-02-27  
**Notebook:** `notebooks/03_incremental_learning_benchmark_local.ipynb`  
**Results:** `thesis_project/experiments/exp01_shared_autonomy/`

---

## 1. Executive Summary

This report analyses the results of benchmarking **7 continual learning (CL) strategies** on a **shared autonomy** task using the HARMONIC human-robot collaboration dataset. The task predicts **joint velocities** (6D) from **joystick commands + robot state** (12D), which represents a physically correct causal formulation: the human issues joystick commands, and the robot must translate those into appropriate joint motions given its current configuration.

### Key Results

| Rank | Strategy | Avg MSE ↓ | BWT →0 | FWT ↓ | Time (s) |
|------|----------|-----------|--------|-------|----------|
| 🥇 1 | **Joint Training** | **0.4117** | **+0.0185** | +0.6287 | 5681.9 |
| 🥈 2 | **DER++** | **0.4799** | +0.2248 | +0.6358 | 158.2 |
| 🥉 3 | **Online EWC** | 0.6805 | +0.3224 | **+0.6096** | 125.1 |
| 4 | EWC | 0.7134 | +0.3527 | +0.6437 | 202.9 |
| 5 | LwF | 0.8272 | +0.4624 | **+0.5404** | 112.7 |
| 6 | Naive Fine-Tune | 0.9394 | +0.7393 | +0.9599 | 100.1 |
| 7 | SI | 0.9901 | +0.8014 | +1.0314 | 144.6 |

**Bottom line:** DER++ achieves **48.9% lower MSE than Naive Fine-Tune** while being only 58% slower, and comes within **16.6%** of Joint Training's performance at **36× less compute**. Online EWC surprisingly outperforms standard EWC on this dataset. SI performs **worse than Naive Fine-Tune**, indicating it is unsuited to this problem structure.

---

## 2. Experimental Setup

### 2.1 Task Formulation — Shared Autonomy

| Component | Description |
|-----------|-------------|
| **Input (12D)** | Joystick axes (3) + Joint positions (6) + End-effector XYZ (3) |
| **Output (6D)** | Joint velocities (6 actuated joints) |
| **Causal chain** | Human joystick command → robot controller → joint velocities |
| **Policy** | π_θ : (u_joy, q, x_ee) → q̇ |

This formulation learns a **personalized mapping** from human commands + robot state to appropriate robot motions. Each participant has a distinct control style, making this a natural testbed for continual learning — the robot must adapt to new operators without forgetting how to interpret previous ones.

### 2.2 Data

- **Dataset:** HARMONIC (Human And Robot Multimodal Observations of Natural Interactive Collaboration)
- **Participants (tasks):** 10 — p100 through p109
- **Modalities loaded:** `ada_joy.parquet`, `joint_positions.parquet`, `robot_position.parquet`

| Task | Participant | Train | Val | Test | Total |
|------|-------------|-------|-----|------|-------|
| 0 | p100 | 21,274 | 4,558 | 4,558 | 30,390 |
| 1 | p101 | 26,797 | 5,742 | 5,742 | 38,281 |
| 2 | p102 | 33,573 | 7,194 | 7,194 | 47,961 |
| 3 | p103 | 16,945 | 3,631 | 3,631 | 24,207 |
| 4 | p104 | 17,868 | 3,828 | 3,828 | 25,524 |
| 5 | p105 | 11,065 | 2,371 | 2,371 | 15,807 |
| 6 | p106 | 31,734 | 6,800 | 6,800 | 45,334 |
| 7 | p107 | 14,412 | 3,087 | 3,087 | 20,586 |
| 8 | p108 | 10,253 | 2,196 | 2,196 | 14,645 |
| 9 | p109 | 21,081 | 4,516 | 4,516 | 30,113 |
| **Total** | | **205,002** | **43,923** | **43,923** | **292,848** |

<!-- **Observation:** Significant size imbalance — p102 has 3.3× more data than p108. This may bias forgetting patterns toward smaller-data tasks. -->

### 2.3 Normalization Statistics

Z-score normalization applied across all tasks before training:

| Feature | Mean | Std |
|---------|------|-----|
| axes_x (joy) | 0.035 | 0.420 |
| axes_y (joy) | 0.023 | 0.452 |
| axes_z (joy) | 0.001 | 0.041 |
| joint_1-6_pos | −1.26 to 1.08 | 0.40 to 1.64 |
| ee_x/y/z | 0.23 / −0.36 / 0.24 | 0.07 each |
| joint_1-6_vel (target) | −0.02 to 0.05 | 0.10 to 0.24 |

<!-- **Note:** Joint position feature `mico_joint_6_pos` has an anomalous mean (−2.2×10¹⁴) and extremely large std (8.8×10¹⁶), indicating potential data quality issues with one joint encoder. Despite `np.nan_to_num` handling, this may introduce noise. This should be investigated in future work. -->

### 2.4 Model & Hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: 12 → 256 → 256 → 6 |
| Dropout | 0.1 |
| Learning rate | 1e-3 (Adam) |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | Patience = 10 |
| EWC λ | 5000 |
| EWC Fisher samples | 1000 |
| Online EWC γ | 0.95 |
| SI c | 1.0 |
| LwF α | 1.0 |
| DER++ buffer | 5000 |
| DER++ α / β | 0.5 / 0.5 |
| Device | CUDA |
| Seed | 42 |

---

## 3. Detailed Results Analysis

### 3.1 Overall Strategy Ranking

![Strategy Comparison — Avg MSE and BWT](../thesis_project/experiments/exp01_shared_autonomy/comparison_bars.png)
*Figure 1: Horizontal bar charts comparing all 7 strategies on Average Joint-Velocity MSE (left) and Backward Transfer (right). DER++ is the clear winner among incremental strategies.*

```
MSE ranking:    Joint Training (0.412) > DER++ (0.480) > Online EWC (0.681) > EWC (0.713) > LwF (0.827) > Naive (0.939) > SI (0.990)
BWT ranking:    Joint Training (+0.019) > DER++ (+0.225) > Online EWC (+0.322) > EWC (+0.353) > LwF (+0.462) > Naive (+0.739) > SI (+0.801)
FWT ranking:    LwF (+0.540) > Online EWC (+0.610) > Joint (+0.629) > DER++ (+0.636) > EWC (+0.644) > Naive (+0.960) > SI (+1.031)
Speed ranking:  Naive (100s) > LwF (113s) > Online EWC (125s) > SI (145s) > DER++ (158s) > EWC (203s) > Joint (5682s)
```

### 3.2 Diagonal Analysis (Per-Task Specialization MSE)

The diagonal of each accuracy matrix shows the MSE achieved on task _i_ immediately after training on it — i.e., the best possible performance for that task.

| Task | p100 | p101 | p102 | p103 | p104 | p105 | p106 | p107 | p108 | p109 |
|------|------|------|------|------|------|------|------|------|------|------|
| **Naive** | 0.207 | 0.240 | 0.074 | 0.287 | 0.406 | 0.255 | 0.326 | 0.249 | 0.242 | 0.454 |
| **Joint** | 0.207 | 0.256 | 0.096 | 0.332 | 0.493 | 0.297 | 0.381 | 0.497 | 0.680 | 0.711 |
| **EWC** | 0.210 | 0.353 | 0.134 | 0.411 | 0.650 | 0.309 | 0.433 | 0.400 | 0.519 | 0.541 |
| **Online EWC** | 0.207 | 0.341 | 0.134 | 0.422 | 0.644 | 0.289 | 0.444 | 0.386 | 0.502 | 0.534 |
| **SI** | 0.208 | 0.236 | 0.075 | 0.278 | 0.414 | 0.233 | 0.340 | 0.251 | 0.240 | 0.414 |
| **LwF** | 0.205 | 0.467 | 0.230 | 0.414 | 0.626 | 0.392 | 0.395 | 0.435 | 0.501 | 0.444 |
| **DER++** | 0.207 | 0.255 | 0.082 | 0.306 | 0.429 | 0.220 | 0.353 | 0.258 | 0.250 | 0.418 |

**Key observations:**
- **Task 0 (p100)** is easy for all strategies (MSE ≈ 0.21) — this is the starting point before any catastrophic forgetting occurs.
- **Task 4 (p104)** is consistently the hardest task across all strategies (diagonal MSE 0.41–0.65). This participant likely has a distinctive control style.
- **Naive and SI** achieve the **lowest diagonals** — they fully specialize on the current task (no regularization penalty). DER++ is very close, confirming its replay buffer doesn't impede learning.
- **EWC, Online EWC, LwF** show elevated diagonals (especially on tasks 4, 8, 9) — the regularization/distillation terms make it harder to fully fit new tasks. This is the expected **plasticity-stability trade-off**.
- **Joint Training** diagonals increase over time (0.21 → 0.71) because the model must simultaneously fit all seen tasks, diluting task-specific capacity.

### 3.3 Catastrophic Forgetting Analysis

#### 3.3.1 Task 0 Retention (Forgetting Curve)

![Forgetting Curve — Task 0 Performance Over Time](../thesis_project/experiments/exp01_shared_autonomy/forgetting_curve.png)
*Figure 2: MSE on Task 0 (p100) after training on each successive task. Joint Training improves over time while Naive and SI degrade severely. DER++ maintains stable performance.*

MSE on task 0 (p100) after training on each successive task:

| After task | Naive | Joint | EWC | O-EWC | SI | LwF | DER++ |
|------------|-------|-------|-----|-------|-----|-----|-------|
| 0 | 0.207 | 0.207 | 0.210 | 0.207 | 0.208 | 0.205 | 0.207 |
| 1 | 1.047 | 0.226 | 0.466 | 0.485 | 1.191 | 0.568 | 0.255 |
| 2 | 0.779 | 0.204 | 0.511 | 0.543 | 0.800 | 0.633 | 0.280 |
| 3 | 1.006 | 0.179 | 0.480 | 0.608 | 0.904 | 0.662 | 0.299 |
| 4 | 0.954 | 0.164 | 0.518 | 0.646 | 1.049 | 0.750 | 0.315 |
| 5 | 0.887 | 0.151 | 0.505 | 0.573 | 1.146 | 0.720 | 0.320 |
| 6 | 1.000 | 0.135 | 0.478 | 0.603 | 1.132 | 0.755 | 0.337 |
| 7 | 0.793 | 0.134 | 0.433 | 0.595 | 0.839 | 0.738 | 0.359 |
| 8 | 0.857 | 0.131 | 0.483 | 0.617 | 0.986 | 0.775 | 0.357 |
| **9 (final)** | **1.134** | **0.135** | **0.479** | **0.652** | **1.190** | **0.822** | **0.361** |

**Forgetting magnitudes (final − initial):**

| Strategy | Δ MSE on Task 0 | Forgetting ratio |
|----------|-----------------|------------------|
| **Joint Training** | −0.072 | −35% (improved!) |
| **DER++** | +0.154 | +75% |
| **EWC** | +0.269 | +128% |
| **Online EWC** | +0.445 | +215% |
| **LwF** | +0.618 | +302% |
| **Naive Fine-Tune** | +0.927 | +448% |
| **SI** | +0.982 | +474% |

**Analysis:**
- **Joint Training** actually **improves** on task 0 over time (−35% MSE), demonstrating positive backward transfer when all data is retained.
- **DER++** limits forgetting to +75%, a 5× improvement over Naive Fine-Tune — the replay buffer of 5,000 samples effectively anchors task 0 knowledge.
- **EWC** provides moderate protection (+128% forgetting), roughly 3.5× better than Naive.
- **Online EWC** forgets **more** than standard EWC (+215% vs +128%) — γ=0.95 appears to decay the Fisher information too aggressively across 10 tasks. The running average underweights early tasks.
- **SI is worse than Naive** (+474% vs +448%) — the path-integral importance estimates fail to protect relevant parameters. This is a critical failure.
- **LwF** provides moderate protection (+302%) but still substantial forgetting.

#### 3.3.2 Final Row Analysis (Performance After All 10 Tasks)

![Per-Task Final MSE](../thesis_project/experiments/exp01_shared_autonomy/per_task_final_mse.png)
*Figure 3: Final Joint-Velocity MSE per task after training on all 10 participants. p104 is universally the hardest task. DER++ shows the most uniform error distribution.*

The last row of each accuracy matrix shows the model's MSE on every task after training on all 10:

| Task | Naive | Joint | EWC | O-EWC | SI | LwF | DER++ |
|------|-------|-------|-----|-------|-----|-----|-------|
| p100 | 1.134 | 0.135 | 0.479 | 0.652 | 1.190 | 0.822 | 0.361 |
| p101 | 0.943 | 0.157 | 0.695 | 0.763 | 0.987 | 0.882 | 0.563 |
| p102 | 1.118 | 0.083 | 0.564 | 0.676 | 1.358 | 0.817 | 0.176 |
| p103 | 1.039 | 0.304 | 0.674 | 0.728 | 1.009 | 0.764 | 0.501 |
| p104 | 1.567 | 0.526 | 1.094 | 1.127 | 1.610 | 1.384 | 0.838 |
| p105 | 0.707 | 0.338 | 0.996 | 0.506 | 0.709 | 0.858 | 0.377 |
| p106 | 0.691 | 0.462 | 0.599 | 0.601 | 0.727 | 0.625 | 0.512 |
| p107 | 0.871 | 0.583 | 0.643 | 0.578 | 0.929 | 0.738 | 0.480 |
| p108 | 0.870 | 0.817 | 0.848 | 0.641 | 0.967 | 0.937 | 0.573 |
| p109 | 0.454 | 0.711 | 0.541 | 0.534 | 0.414 | 0.444 | 0.418 |
| **Mean** | **0.939** | **0.412** | **0.713** | **0.681** | **0.990** | **0.827** | **0.480** |

**Per-task observations:**
- **p104** is universally the hardest task (highest MSE across the board), likely due to a distinct control style or unusual joystick usage patterns.
- **p109** (last task) has low MSE for all strategies — no forgetting has occurred yet since it was trained last. Joint Training is the outlier here (0.711) because it balances all tasks simultaneously.
- **DER++** has the most uniform distribution of errors — no single task has MSE > 0.84, showing the replay buffer prevents severe degradation on any individual task.
- **Joint Training** excels on early tasks (p100: 0.135) but struggles on later ones (p109: 0.711) because the model capacity is shared across all 10 tasks.

### 3.4 Strategy-Specific Analysis

#### 3.4.1 Joint Training (Upper Bound)

- **Avg MSE: 0.412** — Best overall, confirming that retaining all data remains the gold standard.
- **BWT: +0.019** — Nearly zero forgetting. The small positive value indicates the model slightly favors more recent data distributions even in joint training.
- **Time: 5,682s (94.7 min)** — **36× slower than DER++**, **57× slower than Naive**. Completely impractical for online adaptation in real HRC scenarios.
- **Insight:** Joint Training's diagonal MSEs *increase* over time (0.207 → 0.711), revealing an important phenomenon: as the model tries to simultaneously minimize loss across all 10 participants, it loses capacity to perfectly fit any single one. This is the **multi-task interference** problem, separate from catastrophic forgetting.

#### 3.4.2 DER++ (Best Incremental Strategy)

- **Avg MSE: 0.480** — Only **16.6% behind Joint Training**, the closest any incremental strategy gets.
- **BWT: +0.225** — Moderate forgetting, but 3.3× better than Naive.
- **Time: 158.2s** — Practical for deployment (36× faster than Joint Training).
- **Buffer size analysis:** With 5,000 stored samples across ~205,000 total training samples, DER++ retains only 2.4% of the data yet achieves 83.4% of Joint Training's performance. This is an excellent memory-performance ratio.
- **Why DER++ excels here:** The shared autonomy mapping is relatively low-dimensional (12→6), so a small replay buffer effectively spans the input distribution. The dual objectives (replay loss + dark knowledge distillation) provide complementary protection.

#### 3.4.3 Online EWC vs. Standard EWC

| Metric | EWC | Online EWC | Winner |
|--------|-----|------------|--------|
| Avg MSE | 0.713 | **0.681** | Online EWC |
| BWT | 0.353 | **0.322** | Online EWC |
| FWT | 0.644 | **0.610** | Online EWC |
| Time | 202.9s | **125.1s** | Online EWC |

**Online EWC dominates** on all metrics. This is surprising given the Task 0 forgetting analysis showed Online EWC forgetting more on that specific task (+215% vs +128%). The explanation:
- Online EWC better distributes its protection across **all** tasks (lower average forgetting)
- Standard EWC over-protects early tasks at the expense of later ones
- The running Fisher average (γ=0.95) provides smoother importance weighting across 10 tasks
- Online EWC is also significantly faster (38% less time) because it avoids full Fisher recomputation

#### 3.4.4 SI (Failure Case)

- **Avg MSE: 0.990** — **Worse than Naive Fine-Tune** (0.939), making SI counterproductive.
- **BWT: +0.801** — More forgetting than Naive (+0.739), meaning SI's regularization actively *harms* both retention and acquisition.
- **FWT: +1.031** — Worst forward transfer, confirming severe negative interference.

**Root cause analysis:**
1. **SI's path integral relies on gradient trajectory statistics** accumulated during training. With `patience=10` early stopping, training terminates too quickly for reliable importance estimates.
2. The **c=1.0 coefficient is poorly calibrated** for this problem. With only 12 input dimensions and 6 outputs, the parameter importance landscape is very different from vision benchmarks where SI was designed.
3. SI accumulates importance over the **entire training trajectory**, including early noisy gradients. For HRC data with high variance between participants, these noisy estimates cause the model to over-protect irrelevant parameters.
4. **Recommendation:** SI should be removed from the benchmark or tested with c ∈ {0.01, 0.1, 10, 100} and without early stopping.

#### 3.4.5 LwF (Knowledge Distillation)

- **Avg MSE: 0.827** — Moderate performance, positioned between EWC variants and Naive.
- **BWT: +0.462** — Significant forgetting despite distillation.
- **FWT: +0.540** — **Best forward transfer** among all strategies.

LwF's strength in forward transfer makes sense: the soft targets from the old model provide a form of **implicit regularization** that smooths the loss landscape, helping the model start from a better position for new tasks. However, in HRC, participant control styles vary substantially enough that the soft targets become poor proxies for old task knowledge over time.

#### 3.4.6 Naive Fine-Tune (Lower Bound)

- **Avg MSE: 0.939** — Second-worst, establishing the forgetting baseline.
- **BWT: +0.739** — Severe forgetting, confirming catastrophic forgetting is real in this setting.
- **FWT: +0.960** — Poor forward transfer: each new task is learned almost from scratch.
- **Value:** Serves as the critical baseline — any CL strategy that doesn't beat this is useless.

### 3.5 Computational Efficiency Analysis

| Strategy | Time (s) | MSE | MSE per second | Speedup vs Joint |
|----------|----------|-----|----------------|------------------|
| Joint Training | 5681.9 | 0.412 | 7.25e-5 | 1× |
| DER++ | 158.2 | 0.480 | 3.03e-3 | 35.9× |
| Online EWC | 125.1 | 0.681 | 5.44e-3 | 45.4× |
| EWC | 202.9 | 0.713 | 3.52e-3 | 28.0× |
| SI | 144.6 | 0.990 | 6.85e-3 | 39.3× |
| LwF | 112.7 | 0.827 | 7.34e-3 | 50.4× |
| Naive | 100.1 | 0.939 | 9.38e-3 | 56.8× |

**Pareto-optimal strategies** (best MSE for given compute budget):
1. **DER++ (158s, MSE=0.480)** — Best absolute incremental performance
2. **Online EWC (125s, MSE=0.681)** — When memory buffers are unacceptable

### 3.6 Average MSE Evolution Over Training Sequence

![Average MSE Evolution](../thesis_project/experiments/exp01_shared_autonomy/avg_mse_evolution.png)
*Figure 4: Average MSE on all seen tasks (0..i) as new tasks are added. Joint Training stays flat, DER++ rises slowly, while Naive and SI degrade rapidly.*

---

## 4. Comparison with Previous (Incorrect) Formulation

The previous experiment (`exp01_il_comparison`) used a reversed causality formulation: predicting joystick commands (5D) from robot state (51D). Here we compare key differences:

| Metric | Old (51→5, joystick pred.) | New (12→6, shared autonomy) | Change |
|--------|---------------------------|----------------------------|--------|
| Joint Training MSE | 0.555 | 0.412 | −26% (better) |
| DER++ MSE | 0.661 | 0.480 | −27% (better) |
| Naive MSE | 0.905 | 0.939 | +4% (slightly worse) |
| EWC MSE | 0.755 | 0.713 | −6% (better) |
| Mean diagonal (initial fit) | ~0.20–0.30 | ~0.21–0.45 | Similar |
| Naive BWT | +0.238 | +0.739 | 3× more forgetting |
| DER++ BWT | +0.082 | +0.225 | 2.7× more forgetting |

**Key differences:**
1. The new formulation shows **more severe catastrophic forgetting** (BWT values are 2–3× larger). This is expected because predicting 6D continuous joint velocities from 12D input is a harder mapping with more participant-specific structure than the smoother, redundant 51→5 joystick prediction.
2. Despite higher forgetting, **CL strategies still show clear differentiation**, validating the benchmark design.
3. Joint Training and DER++ both improve in absolute MSE, suggesting the new formulation fits the data better (correct causal direction).

---

## 5. Data Quality Observations

### 5.1 Anomalous Feature: mico_joint_6_pos

The normalization statistics reveal `mico_joint_6_pos` (index 8 in the 12D observation vector) has:
- Mean: −2.22 × 10¹⁴ (clearly wrong for a joint position in radians)
- Std: 8.78 × 10¹⁶

**Root cause (investigated):** Exactly 3 extreme outlier samples in participant p109 — 1 in run_010 (−3.03×10¹⁸) and 2 in run_011 (−6.24×10¹⁸, −3.93×10¹⁹). The normal range for `mico_joint_6_pos` across all participants is [0.16, 9.71] rad (mean=3.34, std=2.13). Only 0.0008% of 373,832 total samples are affected.

**Fix:** Clip `mico_joint_6_pos` values outside [−50, 50] to the column median. This is applied in subsequent experiments (notebook 04+).

### 5.2 Task Size Imbalance

The largest task (p102: 47,961 samples) is 3.3× the smallest (p108: 14,645). This means:
- Training on p102 takes longer and produces more gradient updates
- Fisher information for EWC/Online EWC is biased toward p102's distribution
- DER++'s buffer of 5,000 samples represents a different fraction of each task (10.4% of p108 vs 34.1% of p102... wait, 5000/47961 = 10.4% of p102 and 34.1% of p108)

### 5.3 Target Distribution

Joint velocity targets have small magnitudes (mean ≈ 0.0, std ≈ 0.10–0.24 rad/s), indicating the robot moves slowly during the feeding task. The relatively large MSE values (0.4–1.0 in normalized space) suggest that participant-specific velocity patterns are genuinely distinct and hard to predict from the joystick alone — there is likely participant-specific gain/mapping between joystick deflection and resulting velocity.

---

## 6. Conclusions & Recommendations

### 6.1 Strategy Recommendations for HRC Deployment

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| **Offline batch adaptation** | Joint Training | Best performance when all data available |
| **Online incremental (w/ memory)** | DER++ | 83.4% of Joint Training at 2.8% memory cost |
| **Online incremental (no memory)** | Online EWC | Best regularization approach, fast |
| **Resource-constrained robot** | Online EWC | Only 125s, no replay buffer needed |

### 6.2 Strategies to Avoid

- **SI:** Worse than doing nothing. Remove or extensively retune (c ≪ 1.0, disable early stopping).
- **LwF:** Moderate performance but outperformed by both EWC variants and DER++ — no niche advantage.
- **Naive Fine-Tune:** Only useful as a baseline; never deploy in multi-user HRC.

### 6.3 Next Steps

1. **Hyperparameter sweep:** 
   - DER++ buffer size: {1000, 2500, 5000, 10000}
   - EWC λ: {500, 1000, 5000, 10000}
   - Online EWC γ: {0.9, 0.95, 0.99}
   - SI c: {0.01, 0.1, 1.0, 10.0} with and without early stopping

2. **Data quality:** ✅ Investigated — 3 extreme outlier samples in `mico_joint_6_pos` (p109 runs 010/011). Fixed via clipping in notebook 04.

3. **Additional strategies:**
   - **PackNet** (parameter isolation) — may better handle disjoint participant distributions
   - **Progressive Neural Networks** — grow capacity for each new participant
   - **Task-conditioned models** — use participant embedding to modulate the policy

4. **Richer input features:**
   - Add gaze positions (18D) from `gaze_positions.parquet`
   - Add EMG signals from `myo_emg.parquet` (when available)
   - These would create a multimodal shared autonomy model

5. **Per-joint analysis:** Break down MSE by individual joint velocity to identify which joints are hardest to predict and which suffer most from forgetting.

6. **Cross-validation:** Run with different participant orderings to assess sensitivity to task sequence.

---

<!-- ## Appendix A: Accuracy Matrices (R[i,j])

![Accuracy Matrices Heatmaps](../thesis_project/experiments/exp01_shared_autonomy/accuracy_matrices.png)
*Figure 5: Full R[i,j] accuracy matrices for all 7 strategies, visualized as heatmaps. Lighter colors indicate lower MSE. Joint Training and DER++ show the most uniformly light matrices.*

Each entry R[i,j] = MSE on task j after training tasks 0..i.

### A.1 Naive Fine-Tune
```
       p100    p101    p102    p103    p104    p105    p106    p107    p108    p109
  0 | 0.207   1.073   1.093   0.926   1.533   1.072   0.755   1.030   1.357   0.987
  1 | 1.047   0.240   1.083   1.329   2.191   1.018   1.630   1.192   1.727   2.054
  2 | 0.779   0.809   0.074   1.315   1.739   0.671   1.709   1.159   1.525   1.602
  3 | 1.006   1.024   0.578   0.287   1.581   1.285   0.729   1.053   1.959   2.313
  4 | 0.954   1.118   0.839   0.933   0.406   1.756   0.736   1.496   3.650   5.805
  5 | 0.887   1.045   0.680   1.066   0.883   0.255   0.753   1.342   1.795   1.412
  6 | 1.000   1.043   0.730   0.710   1.349   0.526   0.326   1.089   1.576   0.951
  7 | 0.793   1.096   0.739   1.034   1.436   0.638   0.637   0.249   1.428   1.193
  8 | 0.857   0.998   0.931   1.032   1.644   0.710   0.704   0.857   0.242   1.095
  9 | 1.134   0.943   1.118   1.039   1.567   0.707   0.691   0.871   0.870   0.454
```

### A.2 DER++ (Best Incremental)
```
       p100    p101    p102    p103    p104    p105    p106    p107    p108    p109
  0 | 0.207   1.073   1.174   0.943   1.522   1.330   0.799   1.005   1.313   1.333
  1 | 0.255   0.255   0.695   1.170   1.978   0.946   1.339   0.972   1.566   2.173
  2 | 0.280   0.364   0.082   1.021   1.703   0.853   1.332   0.899   1.328   1.682
  3 | 0.299   0.434   0.125   0.306   1.332   0.725   0.640   0.838   1.319   1.387
  4 | 0.315   0.474   0.148   0.429   0.429   0.766   0.610   0.841   1.704   1.601
  5 | 0.320   0.470   0.152   0.426   0.581   0.220   0.620   0.795   1.183   0.991
  6 | 0.337   0.500   0.159   0.449   0.691   0.327   0.353   0.751   1.188   1.024
  7 | 0.359   0.540   0.171   0.507   0.760   0.352   0.462   0.258   1.169   0.911
  8 | 0.357   0.557   0.174   0.512   0.810   0.374   0.483   0.449   0.250   0.866
  9 | 0.361   0.563   0.176   0.501   0.838   0.377   0.512   0.480   0.573   0.418
```

### A.3 Joint Training (Upper Bound)
```
       p100    p101    p102    p103    p104    p105    p106    p107    p108    p109
  0 | 0.207   1.056   1.165   0.945   1.497   1.142   0.825   1.019   1.325   1.333
  1 | 0.226   0.256   0.644   1.144   2.008   1.113   1.274   0.923   1.495   2.532
  2 | 0.204   0.267   0.096   1.503   2.213   0.737   2.753   1.343   1.665   3.574
  3 | 0.179   0.229   0.104   0.332   1.348   0.761   0.717   0.940   1.334   3.078
  4 | 0.164   0.207   0.095   0.360   0.493   0.982   0.668   1.120   1.289   2.468
  5 | 0.151   0.183   0.088   0.340   0.590   0.298   0.633   0.924   1.341   1.245
  6 | 0.135   0.168   0.083   0.322   0.551   0.326   0.381   0.827   1.105   1.077
  7 | 0.134   0.164   0.085   0.318   0.537   0.339   0.456   0.497   1.190   3.397
  8 | 0.131   0.156   0.083   0.311   0.540   0.324   0.474   0.592   0.680   1.218
  9 | 0.135   0.157   0.083   0.304   0.526   0.338   0.462   0.583   0.817   0.711
```

--- -->

## Appendix B: Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Avg MSE** | (1/T) Σⱼ R[T,j] | Mean MSE across all tasks after final training step |
| **BWT** | (1/(T−1)) Σᵢ₌₁ᵀ⁻¹ (R[T,i] − R[i,i]) | Mean increase in MSE on old tasks (+ = forgetting) |
| **FWT** | (1/(T−1)) Σᵢ₌₁ᵀ⁻¹ (R[i−1,i] − R_random[i]) | How well prior knowledge helps on unseen tasks |

Where R[i,j] = MSE evaluated on task j after training on tasks 0..i, and T = number of tasks.

---

*Report generated from experiment results in `exp01_shared_autonomy/`. All values from `full_comparison.json` and individual strategy result files.*
