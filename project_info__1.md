# PhD Thesis: Collaboration Humain-Robot — Codebase Analysis

**Author**: Ameur Gargouri  
**Title**: "Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale"  
**Repository**: https://github.com/G0AMER/thesis.git  
**Date of Analysis**: 2026-07-07

---

## Summary

This workspace contains the complete research ecosystem for a PhD thesis on **incremental human-robot collaboration** with behavioral adaptation. The work addresses how a collaborative robot (cobot) can learn operator preferences incrementally during physical collaboration — evolving from standardized interaction to a personalized partnership without explicit reprogramming. The project spans four research axes (operator modeling → inverse RL → behavior generation → alignment metrics) and delivers validated contributions on three public datasets (HARMONIC, DASIG, MultiPhysio-HRC), including a novel incremental learning algorithm called **FLAIR**, a state-of-the-art IMU-based safety detection system, and reproducible multimodal preprocessing pipelines.

---

## Architecture

### Thesis Structure — 4 Research Axes

| Axis | Goal | Key Methods |
|------|------|-------------|
| **A1 — Modeling human operational schemas** | Learn each operator's habits, action sequences, timing | LSTM / Transformers on hand trajectories; strategy clustering |
| **A2 — Incremental Inverse RL (IRL)** | Infer operator's reward function online, integrating physical/verbal/facial corrections | Online MaxEnt IRL / Deep MaxEnt IRL; adaptive forgetting; FLAIR algorithm |
| **A3 — Personalized behavior generation** | Produce robot trajectories, speeds, and timing aligned with learned operator style | CVAE conditioned on style vector; temporal scheduling; multi-objective optimization |
| **A4 — Alignment & fluidity metrics** | Quantify how well robot behavior matches human | Joint-action entropy; workspace overlap; mutual idle time |

### 8-Layer Physical Architecture (from `phd_project/ARCHITECTURE.md`)

```
Layer 0 — Sensor & Data Acquisition
Layer 1 — Human Perception & State Estimation (intent, gesture, cognitive load, phase seg.)
Layer 2 — Operator Style Modeling (sequence encoder, strategy clustering, profile manager)
Layer 3 — Incremental Inverse Reinforcement Learning (MaxEnt, Deep MaxEnt, AIRL, T-REX)
Layer 4 — Personalized Behavior Generation (CVAE, DMP, temporal scheduler)
Layer 5 — Motion Planning & Execution (MoveIt 2, impedance control)
Layer 6 — Safety & Compliance (always active, non-learnable, ISO/TS 15066)
Layer 7 — Collaboration Metrics & Evaluation
```

Key design principles: modularity (each layer independently testable), safety-first (Layer 6 wraps all motor commands), incremental by design (streaming data), multi-algorithm (each layer implements multiple candidates), ROS 2 native.

### Technology Stack

| Component | Choice |
|-----------|--------|
| Middleware | **ROS 2** (planned, not yet implemented in code) |
| Learning framework | **PyTorch** |
| IL methods | **DER/DER++** + replay buffer; **Adapter modules** for multi-user |
| IRL | Online MaxEnt IRL / Deep MaxEnt IRL / AIRL / T-REX |
| Novel algorithm | **FLAIR** (proposed: FiLM + replay + Fisher regularization + RetroBoost) |
| Behavior generator | **CVAE** conditioned on style vector |
| Safety detection | **ConvNeXt 1D, TCN, MLP-Mixer 1D, Transformer 1D, InceptionTime, LSTM-FCN, CRNN** |
| Classical ML | Random Forest, Gradient Boosting, XGBoost |

---

## Datasets Used

| Dataset | Modalities | Subjects | Role |
|---------|-----------|----------|------|
| **HARMONIC** | EMG, IMU, gaze, joystick, robot joints, video, body/hand/face pose | 24 | Intent prediction, style modeling, IRL (shared autonomy experiments) |
| **DASIG** | MIMU (accel+gyro+magneto on 5 segments = 65 channels), Arduino alarm timestamps | 60 | Safety-state detection (abrupt vs. standard gestures) |
| **MultiPhysio-HRC** | EEG, ECG, EDA, EMG, RESP, audio, video | 55 | Stress/cognitive load estimation, task type detection |

---

## Directory Structure

```
thesis/
├── README.md                          # Main thesis overview
├── README_RESEARCH.md                 # MultiPhysio-HRC research workflow
├── Pipeline_Architecture_Workflow.md  # Safety pipeline documentation
├── Final_Models_Summary.md            # Safety model rankings
├── Literature_Contributions.md        # SotA comparison + 5 contributions
├── SENSOR_RESEARCH_COMPREHENSIVE.md   # Sensor modality selection guide
│
├── phd_project/                       # MAIN PROJECT (structured codebase)
│   ├── pyproject.toml                 # Python project config
│   ├── requirements.txt               # Dependencies
│   ├── ARCHITECTURE.md                # Full 8-layer architecture (critical doc)
│   ├── src/
│   │   ├── perception/                # Layer 1 — intent recognition, gesture, human state
│   │   ├── style_modeling/            # Layer 2 — DASIG pipeline, sequence models, clustering
│   │   ├── incremental_learning/      # Layer 3 core — FLAIR? DER++, EWC, adapters, BC
│   │   │   ├── base.py                # ContinualLearner abstract class
│   │   │   ├── behavioral_cloning.py  # NaiveFineTune + JointTraining baselines
│   │   │   ├── replay_based/          # DER, DER++, iCaRL
│   │   │   ├── regularization_based/  # EWC, SI, LwF, MAS
│   │   │   └── adapter_based/         # Adapter modules, PackNet
│   │   ├── irl/                       # Layer 3 — MaxEnt, Deep MaxEnt, AIRL, online IRL
│   │   ├── behavior_generation/       # Layer 4 — CVAE, DMP, temporal scheduler
│   │   ├── metrics/                   # Layer 7 — alignment, fluidity, safety
│   │   ├── safety/                    # Layer 6 — collision, speed limiter, impedance
│   │   └── utils/                     # Data loading, visualization, logging, splits
│   ├── notebooks/
│   │   ├── 01_dataset_exploration.ipynb      # HARMONIC + DASIG exploration
│   │   ├── 02_preprocessing_pipeline.ipynb   # Preprocessing pipeline
│   │   ├── 03_incremental_learning_benchmark*.ipynb # IL benchmarks
│   │   ├── 04_hp_sweep_and_ordering*.ipynb   # Hyperparameter sweeps
│   │   ├── 05_*_definitive_benchmark*.ipynb  # Definitive benchmarks
│   │   ├── 06_scalability_analysis.ipynb     # Scalability
│   │   ├── 08_der_sa_experiment.ipynb        # DER + shared autonomy
│   │   ├── 09_cpg_net_experiment.ipynb       # CPG network experiments
│   │   └── 10_final_benchmarking*.ipynb      # Final benchmarks
│   ├── scripts/
│   │   ├── build_reference_report.py         # Builds Word + JSON reference reports
│   │   └── build_detailed_reference_report.py
│   ├── configs/                     # YAML configs (dasig_preprocess, harmonic_preprocess)
│   ├── experiments/                  # 6 experiment directories
│   ├── tests/                        # Unit tests (data loaders, preprocessing)
│   ├── reports/                      # Generated reports
│   └── data/                         # raw/, processed/, replay_buffer/, models/
│
├── cobot_safety_model/              # Standalone safety detection package
│   ├── __init__.py
│   ├── data_loader.py               # DASIG CSV loader with European decimal format
│   ├── features.py                  # Sliding window + handcrafted feature extraction
│   └── models.py                    # RF, GBM classifiers + evaluation + plotting
│
├── run_pipeline.py                  # CLI pipeline: DASIG → features → train RF/GBM
├── test_v4.py                       # Deep learning (TCN) training with 5-fold CV
├── generate_pipeline_pdfs.py        # 7 publication-quality PDF figures
│
├── IEEE-conference-template-062824/ # Conference paper artifacts
│   ├── flair_paper.tex              # FLAIR algorithm paper
│   ├── task_type_detection_paper.tex # MultiPhysio-HRC task detection paper
│   ├── flair_architecture.png       # FLAIR architecture diagram
│   └── task_pipeline.png            # Task detection pipeline
│
├── latex_project/                   # LaTeX thesis report
│   ├── main.tex                     # Main document (7 chapters)
│   ├── chapters/                    # Individual chapter sources
│   └── references.bib               # Bibliography
│
├── thesis_presentation.md           # Marp-based defense slides (comprehensive)
├── thesis_presentation.html         # Rendered HTML presentation
│
├── *.pth                            # Trained model weights
├── *_cm.png                         # Confusion matrices
├── dl_model_comparison*.png         # Model comparison plots
│
├── outputs/                         # Generated reports and figures
│   └── cobot_safety/                # RF + GBM evaluation outputs
│
├── research_outputs/                # Task-type detection outputs
├── pipeline_figures/                # 7 generated PDF figures
└── data/                            # Dataset files and references
```

---

## Key Abstractions

### 1. `ContinualLearner` (Abstract Base Class)
- **File**: `phd_project/src/incremental_learning/base.py` (line ~80)
- **Responsibility**: Abstract interface for all incremental learning strategies. Provides the training loop (epoch, batch, early stopping) and task evaluation.
- **Interface**: `on_task_start(task_id)`, `compute_loss(obs, act, task_id)`, `on_task_end(task_id)`, `train_task(task_id, loader)`, `evaluate_task(loader, task_id)`
- **Lifecycle**: Instantiated per model; used sequentially across tasks; state saved/loaded via `.save()` / `.load()`
- **Used by**: `NaiveFineTune`, `JointTraining`, and all concrete strategies (EWC, DER++, Adapters, PackNet)

### 2. `NaiveFineTune` (Lower Bound Baseline)
- **File**: `phd_project/src/incremental_learning/behavioral_cloning.py`
- **Responsibility**: Sequential fine-tuning without any anti-forgetting mechanism. The lower bound that demonstrates catastrophic forgetting.
- **Why it exists**: Serves as the "do nothing" baseline for incremental learning experiments.

### 3. `JointTraining` (Upper Bound / Oracle)
- **File**: `phd_project/src/incremental_learning/behavioral_cloning.py`
- **Responsibility**: Retrains from scratch on the union of all task data. Not realistic for streaming data but shows the theoretical best performance.
- **Why it exists**: Defines the performance ceiling for all incremental methods.

### 4. DASIG Data Loader (`load_all_trials`, `TrialData`, `SubjectInfo`)
- **File**: `cobot_safety_model/data_loader.py`
- **Responsibility**: Load DASIG CSV files (semicolon-separated, comma-decimal format), parse 65 MIMU channels + Arduino events, generate safety-state labels (SAFE/WARNING/DANGER) with configurable reaction time and abrupt duration.
- **Key detail**: Handles the European decimal format (`2,5` instead of `2.5`) which is a common parsing trap.
- **Label strategy**: Dynamic motion-onset labeling using peak sternum acceleration.

### 5. Safety Model Pipeline (`features.py` + `models.py`)
- **File**: `cobot_safety_model/features.py`, `cobot_safety_model/models.py`
- **Responsibility**: Sliding window segmentation (1.0s windows, 0.5s step), handcrafted feature extraction (RMS, peak, jerk, energy, quaternion angular displacement), Z-score normalization, RF/GBM training and evaluation.
- **Label strategies**: "majority" (most frequent label), "any_danger" (DANGER if any), "center" (center sample).
- **Used by**: `run_pipeline.py`

### 6. FLAIR Algorithm
- **Not in a single file yet** — documented in `IEEE-conference-template-062824/flair_paper.tex`
- **Responsibility**: Novel hybrid incremental learning algorithm combining: FiLM (Feature-wise Linear Modulation) for task-specific adaptation, experience replay (DER++ style), Fisher importance regularization (EWC style), RetroBoost for boosting retention, warm-start for FiLM layers, and adaptive replay mixing.
- **Results**: Achieves R²=0.6895 on sequential HARMONIC benchmark, outperforming DER++ (0.612), EWC (0.368), and A-GEM (0.420) with only 1.28 MB memory footprint.
- **Innovation**: Combines architectural (FiLM), replay, and regularization approaches in a single framework.

### 7. FLAIR Ablation (`IEEE-conference-template-062824/flairv1.tex`)
- Demonstrates that all components contribute: baseline → +FiLM → +replay → +importance regularization → +RetroBoost → +warm-start → +adaptive replay weighting
- Each component adds measurable improvement on the sequential benchmark.

### 8. Task Type Detection Pipeline (MultiPhysio-HRC)
- **Artifacts in**: `IEEE-conference-template-062824/task_type_detection_paper.tex`
- **Responsibility**: Classify 5 task types (cognitive load, high stress, industrial task, low load, other) from fused physiological features.
- **Best model**: XGBoost (accuracy 0.937, balanced accuracy 0.894, macro-F1 0.914)
- **Pipeline**: Preprocessing → tabular fusion → imbalance-aware training → stratified 80/20 split

---

## Data Flow

### A) Safety Detection Pipeline (DASIG) — Fully Implemented

```
1. Raw IMU (.csv files)  →  cobot_safety_model/data_loader.py
   - 60 subjects × 3 trials each (FR_R, FR_L, LA_L)
   - 65 MIMU channels @ 200 Hz
   - Arduino event timestamps → safety labels (SAFE, WARNING, DANGER)

2. Sliding Window Segmentation  →  cobot_safety_model/features.py
   - Window: 0.5–1.0s, Step: 0.25–0.5s (50% overlap)
   - Per-trial Z-score normalization
   - Output: 69,094 windows × 65 channels × 100 timesteps

3. Dynamic Feature Engineering (on GPU in test_v4.py)
   - Position (65) + Velocity (65) + Acceleration (65) = 195 channels
   - Computed on-the-fly inside DataLoader via torch.diff()

4. Model Training  →  test_v4.py  (or run_pipeline.py for classical ML)
   Models trained: TCN, ConvNeXt 1D, MLP-Mixer 1D, Transformer 1D,
   InceptionTime, LSTM-FCN, CRNN, ResNet 1D, 1D-CNN
   - 5-fold StratifiedKFold cross-validation
   - Focal Loss (γ=2.0), AdamW, OneCycleLR, Mixup (30%)
   - WeightedRandomSampler for class balancing

5. Test-Time Augmentation (TTA)
   - 4 forward passes with Gaussian noise (σ=0.01)
   - Average softmax probabilities

6. Post-Processing (Production)
   - Median filter (kernel=3) for temporal smoothing
   - Fixed threshold (0.77) on smoothed probability
   - Output: SAFE or DANGER decision

7. Result: ConvNeXt 1D best (99.43% acc, 98.68% macro-F1, 97.13% danger recall)
```

### B) Incremental Learning Benchmark — Implemented

```
1. Task sequence defined (e.g., HARMONIC shared autonomy per-subject)
2. For each task t:
   a. on_task_start(t)  →  prepare per-task data
   b. Training loop over task t's data
   c. on_task_end(t)  →  consolidate (DER++ buffer, EWC fisher, etc.)
3. After each task t, evaluate on ALL tasks seen so far
4. Build accuracy matrix R[i,j] = performance on task j after training up to task i
5. Compute metrics: Average Accuracy, Backward Transfer (BWT), Forward Transfer (FWT)
6. Compare: NaiveFineTune (lower bound) < EWC < DER++ < FLAIR < JointTraining (oracle)
```

### C) Planned Full System Flow (from ARCHITECTURE.md)

```
Wearable Sensors (EMG + IMU)  →  Layer 0
  → Human Intent Encoder (LSTM/Transformer)  →  Layer 1
  → Incremental Learning Module (DER++ / Adapters)  →  Layer 2–3
  → Skill Library  →  Layer 3
  → CVAE Style-Conditioned Trajectory Generator  →  Layer 4
  → Motion Planner (MoveIt 2) + Temporal Scheduler  →  Layer 5
  → Impedance/PID Control  →  Layer 5
  → Safety Layer (always active)  →  Layer 6
  → Evaluation Metrics  →  Layer 7
```

---

## Non-Obvious Behaviors & Design Decisions

### 1. European Decimal Format Parser
The DASIG dataset uses `,` as decimal separator and `;` as CSV delimiter. The `data_loader.py` must explicitly handle this with `sep=';'` and `decimal=','`. Most data scientists expect `.` separator — this is a common parsing trap that can silently corrupt numeric data.

### 2. Dynamic On-GPU Derivative Expansion
Instead of pre-computing velocity and acceleration features (which would bloat RAM by 3×), `test_v4.py` computes them **inside the data loader callback** using `torch.diff()`. This avoids OOM crashes while still providing the 195-channel feature space. The padding strategy (`pad = x[:,:,:1]`) ensures dimension alignment after diff.

### 3. 3-Class → 2-Class Label Collapse
The initial safety labeling produces 3 classes (SAFE=0, WARNING=1, DANGER=2). WARNING is a transition state between standard and abrupt motion. Most models collapse this to SAFE(0) vs ABRUPT(1) by treating WARNING as DANGER. The rationale: in a factory safety context, false WARNING is acceptable but missing a DANGER event is critical.

### 4. Testing-Time Augmentation (TTA) for Safety
TTA is extremely rare in real-time
