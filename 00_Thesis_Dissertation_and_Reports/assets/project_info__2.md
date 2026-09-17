# PhD Thesis: Collaboration Humain-Robot — Codebase Analysis

**Author**: Ameur Gargouri  
**Title**: "Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale"  
**Repository**: https://github.com/G0AMER/thesis.git  
**Date of Analysis**: 2026-07-07

---

## Summary

This workspace contains the complete research ecosystem for a PhD thesis on **incremental human-robot collaboration** with behavioral adaptation. The work addresses how a collaborative robot (cobot) can learn operator preferences incrementally during physical collaboration — evolving from standardized interaction to a personalized partnership without explicit reprogramming. The project spans four research axes and delivers validated contributions on three public datasets (HARMONIC, DASIG, MultiPhysio-HRC), including a novel incremental learning algorithm called **FLAIR**, a state-of-the-art IMU-based safety detection system (99.43% accuracy), and reproducible multimodal preprocessing pipelines.

---

## Architecture

### 4 Research Axes

| Axis | Goal | Key Methods |
|------|------|-------------|
| **A1 — Modeling human operational schemas** | Learn operator habits, sequences, timing | LSTM / Transformers; strategy clustering |
| **A2 — Incremental Inverse RL** | Infer reward function online from corrections | Online MaxEnt IRL, Deep MaxEnt, AIRL, T-REX; FLAIR algorithm |
| **A3 — Personalized behavior generation** | Produce style-aligned robot trajectories | CVAE conditioned on style vector; DMP; temporal scheduling |
| **A4 — Alignment & fluidity metrics** | Quantify human-robot coordination quality | Joint-action entropy; workspace overlap; mutual idle time |

### 8-Layer Architecture (from `phd_project/ARCHITECTURE.md`)

```
L0 — Sensor & Data Acquisition (EMG, IMU, RGB-D, Microphone, F/T, Joint encoders)
L1 — Human Perception & State Estimation (intent, gesture, cognitive load, phase seg.)
L2 — Operator Style Modeling (sequence encoder, strategy clustering, profile manager)
L3 — Incremental Inverse RL (MaxEnt, Deep MaxEnt, AIRL, T-REX)
L4 — Personalized Behavior Generation (CVAE, DMP, temporal scheduler)
L5 — Motion Planning & Execution (MoveIt 2, impedance control)
L6 — Safety & Compliance (always active, non-learnable, ISO/TS 15066)
L7 — Collaboration Metrics & Evaluation
```

### Technology Stack

| Component | Choice |
|-----------|--------|
| Framework | **PyTorch** |
| Middleware | **ROS 2** (planned) |
| IL methods | **DER/DER++** + replay buffer; **Adapter modules** |
| Novel algorithm | **FLAIR** (FiLM + replay + Fisher + RetroBoost + warm-start) |
| Safety models | **ConvNeXt 1D, TCN, MLP-Mixer 1D, Transformer 1D, InceptionTime, LSTM-FCN, CRNN** |
| Classical ML | Random Forest, Gradient Boosting, XGBoost |

### Datasets

| Dataset | Modalities | Subjects | Role |
|---------|-----------|----------|------|
| **HARMONIC** | EMG, IMU, gaze, joystick, robot joints, video, body/hand/face pose | 24 | Intent prediction, style modeling, IRL |
| **DASIG** | MIMU (65 channels), Arduino alarms | 60 | Safety-state detection |
| **MultiPhysio-HRC** | EEG, ECG, EDA, EMG, RESP, audio, video | 55 | Task type detection, stress estimation |

---

## Key Abstractions

### 1. `ContinualLearner` (Abstract Base)
- **File**: `phd_project/src/incremental_learning/base.py` (line ~80)
- **Role**: Interface for all incremental learning strategies. Owns model, optimizer, provides `train_task()` loop with early stopping.
- **Key methods**: `on_task_start()`, `compute_loss()`, `on_task_end()`, `evaluate_task()`
- **Used by**: `NaiveFineTune`, `JointTraining`, EWC, DER++, Adapters

### 2. `NaiveFineTune` & `JointTraining`
- **File**: `phd_project/src/incremental_learning/behavioral_cloning.py`
- **Role**: Lower bound (Naive = no anti-forgetting) and upper bound (Joint = all data) baselines for all incremental experiments.

### 3. DASIG Data Pipeline (`cobot_safety_model/`)
- **`data_loader.py`**: Loads 65-channel MIMU CSVs (European decimal format — critical parsing detail), generates SAFE/WARNING/DANGER labels from Arduino alarm events.
- **`features.py`**: Sliding window segmentation (0.5s–1.0s windows, 50% overlap), handcrafted feature extraction (RMS, peak, jerk, quaternion angular displacement), Z-score normalization.
- **`models.py`**: RF and GBM training + evaluation (confusion matrices, feature importance plots).

### 4. Safety Deep Learning (`test_v4.py`)
- End-to-end TCN training with: on-GPU derivative expansion (65→195 channels), Mixup (30% prob), Focal Loss (γ=2), 5-fold GroupKFold, TTA with Gaussian noise.
- Output: best .pth checkpoints for each architecture.

### 5. FLAIR Algorithm
- Documented in `IEEE-conference-template-062824/flair_paper.tex`
- Novel hybrid combining: **FiLM** (task-specific feature modulation) + **experience replay** (DER++) + **Fisher importance regularization** (EWC) + **RetroBoost** + **warm-start** + **adaptive replay mixing**
- **Results**: R²=0.6895 on sequential HARMONIC shared autonomy benchmark (vice DER++ 0.612, EWC 0.368, A-GEM 0.420). Memory: 1.28 MB.
- Ablation shows every component contributes measurably.

### 6. Task Type Detection (MultiPhysio-HRC)
- Artifacts in `IEEE-conference-template-062824/task_type_detection_paper.tex`
- Best model: XGBoost (accuracy 0.937, balanced accuracy 0.894, macro-F1 0.914) on 5-class classification.

---

## Data Flows

### A) Safety Detection Pipeline (Fully Implemented)

```
1. Raw CSV → data_loader.py (65 MIMU channels @ 200 Hz, 60 subjects × ~3 trials)
2. Sliding window → 69,094 windows (0.5–1.0s, 50% overlap)
3. GPU derivative expansion: 65 → 195 channels (pos + vel + acc)
4. Model training (5-fold CV, Focal Loss, AdamW, Mixup, TTA)
5. Post-processing: median filter (k=3) + threshold (0.77)
6. Output: SAFE / DANGER
   Final: ConvNeXt 1D → 99.43% acc, 98.68% macro-F1, 97.13% danger recall
```

### B) Incremental Learning Benchmark (Implemented)

```
For each task t in sequence:
  on_task_start(t) → prepare data
  Train loop on task t
  on_task_end(t) → consolidate (DER++ buffer, EWC Fisher, etc.)
After each task: evaluate on ALL tasks
Build accuracy matrix R[i,j] → compute AA, BWT, FWT
```

### C) Planned Full System

```
Sensors → Human Intent Encoder (LSTM/Transformer)
  → Incremental Learning (DER++ / Adapters) → Style Vector
  → CVAE Trajectory Generator
  → MoveIt 2 Motion Planner + Temporal Scheduler
  → Impedance/PID Control → Robot Arm
  → Safety Layer (always active, non-learnable)
  → Evaluation Metrics
```

---

## Non-Obvious Design Decisions

### 1. European Decimal Format Parser
DASIG CSVs use `,` as decimal separator and `;` as delimiter (`2,5` instead of `2.5`). `data_loader.py` handles this with `sep=';'; decimal=','`. This is a common trap — without explicit handling, all numeric data would be silently corrupted.

### 2. On-GPU Derivative Expansion
Instead of pre-computing velocity+acceleration (3× RAM), `test_v4.py` computes them with `torch.diff()` inside the training loop. Padding with the first value ensures dimension alignment. This avoids OOM on GPU while still providing 195-channel features.

### 3. 3-Class → 2-Class Collapse
Initial labels: SAFE(0), WARNING(1), DANGER(2). In production, WARNING (transition state) is collapsed into DANGER because missing a real danger event is far more costly than a false positive warning.

### 4. TTA for Safety Detection
4 forward passes with Gaussian noise (σ=0.01) are averaged at inference time. This is rare in industrial safety systems but provides calibrated probability scores essential for setting the 0.77 threshold.

### 5. Median Filter + Threshold = 67% FP Reduction
The post-processing pipeline (median filter k=3 + threshold 0.77) reduces false positives from 1,324 to 434 (−67%) while only increasing false negatives by 6%. This directly addresses the productivity-safety tradeoff.

### 6. Subject-Level Splits
All splits are at the **subject** level (not trial level) to prevent data leakage. `splits.py` enforces this via `GroupKFold` and `leave_N_subjects_out`. Stratification by gender is supported.

### 7. `~/.kaggle-outputs/` Directory
Contains Kaggle notebook outputs — these are the actual compute logs from running the experiments on Kaggle GPUs (free T4/P100 access). The notebooks in `phd_project/notebooks/` were executed on Kaggle.

---

## Current State: What's Working vs Not Yet Implemented

### ✅ Fully Working
- **DASIG safety detection pipeline**: End-to-end from raw CSV to deployed model (all 9 architectures trained, evaluated, weights saved)
- **DASIG preprocessing pipeline** (`phd_project/src/style_modeling/dasig_pipeline.py`): CLI tool, outputs Parquet files
- **HARMONIC preprocessing**: Timestamp alignment, EMG filtering, gaze filtering, IMU filtering (`preprocessing.py`)
- **Incremental learning base classes**: `ContinualLearner`, `NaiveFineTune`, `JointTraining` with full metrics tracking (AA, BWT, FWT)
- **FLAIR algorithm**: Implemented in Kaggle notebooks, documented in conference paper
- **Task type detection**: MultiPhysio-HRC, XGBoost best at 0.937 accuracy
- **Reference report builder**: Script that queries OpenAlex, Crossref, Semantic Scholar to build structured reference reports
- **Publication-quality figure generator**: 7 PDF figures for paper inclusion
- **Thesis defense presentation**: Comprehensive Marp slides (`thesis_presentation.md`)

### 🚧 Partially Implemented / Placeholder
- **ROS 2 integration**: The `ros2_interface/` directory exists but contains no nodes or launch files yet
- **Behavior generation**: `cvae/`, `trajectory_planner/`, `temporal_scheduler/` directories exist with `__init__.py` and `.gitkeep` but no implementation code
- **IRL algorithms**: `maxent_irl/`, `deep_maxent_irl/`, `adversarial_irl/`, `online_irl/` directories mostly empty
- **HARMONIC-specific IRL experiments**: The `exp02_irl_comparison/` and `exp04_behavior_gen/` directories are empty
- **Longitudinal study (Exp 6)**: Not yet run (requires human subjects)
- **Perception modules**: `intent_recognition/`, `gesture_detection/`, `human_state/` are mostly stubs
- **Safety watchdog ROS node**: Described in ARCHITECTURE.md but not implemented as ROS 2 code

### 📝 Documented in Papers
- **FLAIR** (`flair_paper.tex`) — complete algorithmic description + results
- **Task type detection** (`task_type_detection_paper.tex`) — system description + results
- **FLAIR ablation study** (`flairv1.tex`) — component contribution analysis
- **Pipeline Architecture** (`Pipeline_Architecture_Workflow.md`) — full safety pipeline

---

## Module Reference

| File | Purpose |
|------|---------|
| `cobot_safety_model/data_loader.py` | DASIG CSV loading + safety label generation |
| `cobot_safety_model/features.py` | Sliding window segmentation + handcrafted features |
| `cobot_safety_model/models.py` | RF/GBM training, evaluation, confusion matrices |
| `test_v4.py` | TCN training with GPU derivatives, 5-fold CV |
| `run_pipeline.py` | CLI: DASIG → features → RF/GBM pipeline |
| `generate_pipeline_pdfs.py` | 7 publication-quality PDF figures |
| `phd_project/src/incremental_learning/base.py` | ContinualLearner abstract class |
| `phd_project/src/incremental_learning/behavioral_cloning.py` | NaiveFineTune + JointTraining baselines |
| `phd_project/src/perception/preprocessing.py` | HARMONIC signal preprocessing (EMG, gaze, IMU) |
| `phd_project/src/style_modeling/dasig_pipeline.py` | DASIG end-to-end preprocessing CLI |
| `phd_project/src/style_modeling/dasig_preprocessing.py` | DASIG filtering, feature extraction, abrupt detection |
| `phd_project/src/utils/splits.py` | Subject-level train/val/test split utilities |
| `phd_project/scripts/build_reference_report.py` | OpenAlex/Crossref/Semantic Scholar metadata collector |
| `phd_project/configs/` | YAML configs for DASIG + HARMONIC preprocessing |
| `get_papers.py` | OpenAlex paper search for SotA literature |
| `IEEE-conference-template-062824/flair_paper.tex` | FLAIR algorithm paper |
| `IEEE-conference-template-062824/task_type_detection_paper.tex` | Task detection paper |
| `Pipeline_Architecture_Workflow.md` | Safety pipeline documentation (with mermaid diagrams) |
| `Final_Models_Summary.md` | Model ranking with confusion matrices |
| `Literature_Contributions.md` | State-of-the-art analysis + 5 contributions |
| `README_RESEARCH.md` | MultiPhysio-HRC research workflow |
| `README.md` | Main thesis overview (4 axes, datasets, architecture) |

---

## Suggested Reading Order

1. **`README.md`** — Start here. Gives the full thesis picture: research questions, 4 axes, datasets, architecture, work plan.

2. **`phd_project/ARCHITECTURE.md`** — The single most important document. 8-layer architecture, 6 experiments, 36-step work plan, publication strategy. Read this carefully.

3. **`Pipeline_Architecture_Workflow.md`** — Deep dive into the fully working safety pipeline with mermaid diagrams. Shows what complete implementation looks like.

4. **`cobot_safety_model/data_loader.py`** + **`test_v4.py`** — The two core implementation files. See the DASIG parser, GPU derivative expansion, TCN training loop, post-processing.

5. **`phd_project/src/incremental_learning/base.py`** — The ContinualLearner abstract class. Understanding this is key to the IL/IRL contribution.

6. **`IEEE-conference-template-062824/flair_paper.tex`** — The FLAIR algorithm paper. The main algorithmic contribution of the thesis.

7. **`thesis_presentation.md`** — The Marp defense slides. Gives the big-picture narrative that ties everything together.

8. **`phd_project/src/perception/preprocessing.py`** — HARMONIC preprocessing details (timestamp alignment, EMG filtering pipeline).

---

## Key Metrics Summary

| Contribution | Metric | Result |
|-------------|--------|--------|
| Safety (ConvNeXt 1D) | Accuracy | **99.43%** |
| Safety (ConvNeXt 1D) | Macro F1 | **98.68%** |
| Safety (ConvNeXt 1D) | Danger Recall | **97.13%** |
| Safety (ConvNeXt 1D) | Danger Precision | **98.25%** |
| Safety (TCN) | Accuracy | **99.41%** |
| FP Reduction (post-processing) | Median filter + 0.77 threshold | **−67%** |
| FLAIR (sequential HRC) | R² | **0.6895** |
| FLAIR (memory) | Model size | **1.28 MB** |
| DER++ (sequential HRC) | R² | 0.6120 |
| EWC (sequential HRC) | R² | 0.3680 |
| Task Type Detection (XGBoost) | Accuracy | **0.9371** |
| Task Type Detection (XGBoost) | Macro F1 | **0.9142** |

---

## Note on File Location

The full report has been saved to `project_info__1.md` in the project root directory for permanent reference.