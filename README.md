# PhD Thesis — Ameur Gargouri

## Title
**Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale**  
*(Human-Robot Collaboration: Incremental Learning and Behavioral Adaptation)*

**Author:** Ameur Gargouri  
**Laboratories:** ATISP Laboratory (ENET'Com, University of Sfax, Tunisia) & ESME Research Lab (Ivry-sur-Seine, France)  
**Supervisors:** Mohamed Karray (ESME) & Mohamed Ksantini (University of Sfax)

---

## Reorganized Workspace Architecture

The workspace is organized into **four self-contained research work folders**, each structured identically with:
- **`src/`** — Code source, Jupyter notebooks, pipelines, and algorithms.
- **`assets/`** — High-resolution architecture diagrams, figures, datasets, and generation scripts.
- **`output/`** — Trained model weights (`.pth`), experimental results, confusion matrices, and reports.
- **`paper/`** — Complete paper redactions, LaTeX templates, bibliographies, and compiled PDFs.

```
thesis/
├── 00_Thesis_Dissertation_and_Reports/
│   ├── paper/               # Thesis dissertation drafts, progress reports (1-6), latex_project
│   ├── assets/              # Defense presentation slides, literature surveys, templates
│   ├── output/              # Compilation logs and synthesis outputs
│   └── src/                 # Literature extraction scripts & side explorations
│
├── 01_Task_Recognition_Multimodal_Fusion/
│   ├── paper/               # CoopIS manuscript (task_type_detection_paper.tex & pdf)
│   ├── src/                 # MultiPhysio-HRC sensor fusion notebooks & LOSO benchmark scripts
│   ├── assets/              # Pipeline diagrams, README_RESEARCH.md, physiological_data.zip
│   └── output/              # Fused datasets, trained models, confusion matrices
│
├── 02_Danger_State_Detection_Deep_Learning/
│   ├── paper/               # Elsevier CAS, Springer, Robotica journal submission packages
│   ├── src/                 # 1D DL pipeline (test_v4.py, ConvNeXt, TCN, run_pipeline.py)
│   ├── assets/              # 7 publication PDF figures, figure generator, DASIG dataset
│   └── output/              # 14 trained PyTorch checkpoints (*.pth), confusion matrices, metrics
│
├── 03_Incremental_Learning_FLAIR/
│   ├── paper/               # KES 2026 conference paper (flair_paper.tex & pdf, KES template)
│   ├── src/                 # 8-layer architecture, continual learning modules (DER++, FLAIR), notebooks
│   ├── assets/              # FLAIR architecture diagrams, conference slide deck
│   └── output/              # Benchmark reports, experimental results (exp01 to exp07)
│
└── data/                    # Backward-compatibility symlinks for raw datasets
```

---

## Research Works Overview

### 1. [01_Task_Recognition_Multimodal_Fusion](./01_Task_Recognition_Multimodal_Fusion/)
- **Paper:** *Task Recognition from Physiological Data Using Multimodal Sensor Fusion For Human-Robot Collaboration* (Submitted to CoopIS).
- **Focus:** Multi-modal sensor fusion (EEG, ECG, EDA, EMG, Respiration) on MultiPhysio-HRC dataset. Achieves 93.71% accuracy and 0.9142 macro-F1 with tuned XGBoost across 5 operational states.

### 2. [02_Danger_State_Detection_Deep_Learning](./02_Danger_State_Detection_Deep_Learning/)
- **Paper:** *Danger-State Detection in Human-Robot Collaboration Using Deep Learning* (Journal article).
- **Focus:** Real-time danger and hazard detection from wearable MIMU inertial sensors (DASIG dataset, 60 subjects). Features on-GPU dynamic kinematic expansion (65→195 features), 5 deep 1D architectures (ConvNeXt 1D, TCN, Transformer 1D, InceptionTime, MLP-Mixer), achieving >98.6% macro-F1 and 87.5% false-alarm reduction.

### 3. [03_Incremental_Learning_FLAIR](./03_Incremental_Learning_FLAIR/)
- **Paper:** *A Combined Incremental Learning Algorithm for Human-Robot Collaboration* (Presented at KES 2026).
- **Focus:** Novel hybrid continual learning algorithm (**FLAIR**: FiLM + Replay + Fisher Regularization + RetroBoost) for multi-operator shared autonomy (HARMONIC dataset). Surpasses joint training ($R^2 = 0.690$ vs $0.644$) with near-zero forgetting ($F = 0.017$) and 1.28 MB memory footprint.

### 4. [00_Thesis_Dissertation_and_Reports](./00_Thesis_Dissertation_and_Reports/)
- **Focus:** Complete thesis synthesis, LaTeX progress reports (`latex_project/`), periodic research evaluations (Reports 1 through 6), and defense presentation slides (`thesis_presentation.html`).

---

## Core Research Axes

| Axis | Goal | Key Methods |
| --- | --- | --- |
| **A1 — Modeling human operational schemas** | Learn each operator's habits, action sequences, timing | LSTM / Transformers on hand trajectories & assembly order; clustering |
| **A2 — Incremental Inverse RL (IRL)** | Infer operator's reward function online from corrections | Online MaxEnt IRL; adaptive forgetting; FLAIR algorithm |
| **A3 — Personalized behavior generation** | Produce robot trajectories and timing aligned with operator style | Conditional VAE (CVAE) conditioned on style vector; temporal scheduling |
| **A4 — Alignment & fluidity metrics** | Objectively quantify human-robot coordination quality | Joint-action entropy; workspace overlap rate; mutual idle time |
