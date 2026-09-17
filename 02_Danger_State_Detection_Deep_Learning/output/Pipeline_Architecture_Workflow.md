# 🏭 Cobot Safety-State Detection — Full Pipeline Architecture

> Deep Learning Pipeline v4 — Factory-Calibrated (Subject-Dependent) Benchmark

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart LR
    A["🦾 Raw IMU\nSensors\n60 Subjects"] --> B["📦 Data Loading\n& Labelling"]
    B --> C["✂️ Windowing\n& Normalisation"]
    C --> D["🔀 StratifiedKFold\n5-Fold CV"]
    D --> E["🧠 GPU Training\nLoop"]
    E --> F["📊 Validation\n& TTA"]
    F --> G["⚙️ Post-Processing\nSmoothing + Threshold"]
    G --> H["✅ Final\nPrediction\nSAFE / DANGER"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style C fill:#16213e,stroke:#0f3460,color:#fff
    style D fill:#0f3460,stroke:#53a8b6,color:#fff
    style E fill:#e94560,stroke:#fff,color:#fff
    style F fill:#0f3460,stroke:#53a8b6,color:#fff
    style G fill:#533483,stroke:#e94560,color:#fff
    style H fill:#2b9348,stroke:#fff,color:#fff
```

---

## 2. Data Ingestion & Labelling

```mermaid
flowchart TD
    subgraph RAW["📁 DASIG Dataset"]
        S1["sub001/"] --> T1["LA_L, LA_R, RA_L, RA_R..."]
        S2["sub002/"]
        SN["sub060/"]
    end

    RAW --> LOAD["load_all_trials\n• Reads CSV files\n• 60 subjects x 3 trials each\n• Generates per-timestep labels"]
    LOAD --> LABELS["Label Generation\n• SAFE = Normal movement\n• DANGER = Abrupt / unexpected"]
    LOAD -->|"Data Cleaning"| WARN["Skips corrupt files\ne.g. sub013/LA_L\ncomma decimal format"]
    LABELS --> OUT["179 Trials Loaded\n65 kinematic channels x variable length"]

    style RAW fill:#1a1a2e,stroke:#e94560,color:#fff
    style LOAD fill:#16213e,stroke:#0f3460,color:#fff
    style LABELS fill:#0f3460,stroke:#53a8b6,color:#fff
    style WARN fill:#e94560,stroke:#fff,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

| Parameter | Value |
|---|---|
| Sensor Channels | **65** (joint angles, positions, orientations) |
| Sampling Rate | **200 Hz** |
| Total Subjects | **60** |
| Total Trials | **179** |

---

## 3. Windowing & Normalisation

```mermaid
flowchart TD
    TRIALS["179 Trials\n65 channels x variable length"] --> SEG["segment_all_trials\n• Window Size: 0.5s = 100 timesteps\n• Step Size: 0.25s = 50% overlap\n• Label Strategy: all"]

    SEG --> NORM["Per-Trial Z-Score Normalisation\n• mu = mean of trial\n• sigma = std of trial\n• x_norm = x - mu / sigma"]

    NORM --> WLABEL["Window-Level Label\n• y_window = max of y_timesteps\n• If ANY timestep is DANGER\n  then entire window = DANGER"]

    WLABEL --> TENSOR["Final Tensors\n• X: 69094 x 65 x 100\n  Batch, Channels, SeqLen\n• y: 69094\n  Binary: 0=SAFE, 1=DANGER"]

    TENSOR --> DIST["Class Distribution\n• SAFE:   60559 = 87.6%\n• DANGER: 8535 = 12.4%"]

    style TRIALS fill:#1a1a2e,stroke:#e94560,color:#fff
    style SEG fill:#16213e,stroke:#0f3460,color:#fff
    style NORM fill:#0f3460,stroke:#53a8b6,color:#fff
    style WLABEL fill:#533483,stroke:#e94560,color:#fff
    style TENSOR fill:#2b9348,stroke:#fff,color:#fff
    style DIST fill:#e94560,stroke:#fff,color:#fff
```

---

## 4. Dynamic Feature Engineering (On-GPU)

To prevent OOM crashes, velocity and acceleration derivatives are computed **dynamically inside the training loop on the GPU** rather than being pre-computed in RAM.

```mermaid
flowchart LR
    X["Input Batch\nB, 65, 100"] --> VEL["Velocity\ndiff along time\nB, 65, 100"]
    X --> ACC["Acceleration\ndiff of diff\nB, 65, 100"]
    X --> CAT["torch.cat\nalong channel dim"]
    VEL --> CAT
    ACC --> CAT
    CAT --> OUT["Expanded Batch\nB, 195, 100\n65 pos + 65 vel + 65 acc"]

    style X fill:#16213e,stroke:#0f3460,color:#fff
    style VEL fill:#0f3460,stroke:#53a8b6,color:#fff
    style ACC fill:#0f3460,stroke:#53a8b6,color:#fff
    style CAT fill:#533483,stroke:#e94560,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

| Feature Group | Channels | Description |
|---|---|---|
| Position (raw) | 65 | Original kinematic channels |
| Velocity (1st derivative) | 65 | Rate of change of position |
| Acceleration (2nd derivative) | 65 | Jerk detection |
| **Total** | **195** | Fed into every model |

---

## 5. Cross-Validation Strategy

```mermaid
flowchart TD
    DATA["Full Dataset\n69094 windows"] --> SKF["StratifiedKFold\nn_splits=5, shuffle=True, seed=42"]

    SKF --> F1["Fold 1\nTrain: ~55275 / Val: ~13819"]
    SKF --> F2["Fold 2\nTrain: ~55275 / Val: ~13819"]
    SKF --> F3["Fold 3\nTrain: ~55275 / Val: ~13819"]
    SKF --> F4["Fold 4\nTrain: ~55275 / Val: ~13819"]
    SKF --> F5["Fold 5\nTrain: ~55275 / Val: ~13819"]

    F1 & F2 & F3 & F4 & F5 --> AVG["Average Metrics\nAcc, F1, Recall, Precision"]

    style DATA fill:#1a1a2e,stroke:#e94560,color:#fff
    style SKF fill:#533483,stroke:#e94560,color:#fff
    style AVG fill:#2b9348,stroke:#fff,color:#fff
```

> [!IMPORTANT]
> **Subject-Dependent (Factory-Calibrated):** StratifiedKFold allows the model to see portions of every subject's data during training. This simulates a real factory where the cobot is calibrated to the specific workers on that floor.

---

## 6. Training Loop Architecture

```mermaid
flowchart TD
    BATCH["Mini-Batch\n128, 65, 100"] --> AUG["Data Augmentation\n• Scale Jitter ±5%\n• Gaussian Noise sigma=0.01"]
    AUG --> DERIV["add_derivatives\n128, 65, 100 -> 128, 195, 100"]

    DERIV --> MIXUP{"Mixup?\n30% chance"}
    MIXUP -->|Yes| MIX["Mixup Augmentation\n• alpha=0.1 Beta distribution\n• Blend two samples\n• Interpolate labels"]
    MIXUP -->|No| DIRECT["Direct Forward Pass"]

    MIX --> MODEL["Neural Network\nTCN / ConvNeXt / etc."]
    DIRECT --> MODEL

    MODEL --> LOGITS["Logits B, 2"]
    LOGITS --> LOSS["Focal Loss\n• gamma=2.0 focus on hard examples\n• Label smoothing=0.01\n• Class weights inverse freq"]

    LOSS --> BACK["Backpropagation\n• Gradient Clipping max=1.0\n• AdamW lr=1e-3, wd=1e-4\n• OneCycleLR Scheduler"]

    BACK --> SAMPLER["WeightedRandomSampler\nBalances SAFE/DANGER per batch"]
    SAMPLER -->|"Next Batch"| BATCH

    style BATCH fill:#1a1a2e,stroke:#e94560,color:#fff
    style AUG fill:#16213e,stroke:#0f3460,color:#fff
    style DERIV fill:#0f3460,stroke:#53a8b6,color:#fff
    style MIXUP fill:#533483,stroke:#e94560,color:#fff
    style MIX fill:#533483,stroke:#e94560,color:#fff
    style MODEL fill:#e94560,stroke:#fff,color:#fff
    style LOSS fill:#16213e,stroke:#0f3460,color:#fff
    style BACK fill:#0f3460,stroke:#53a8b6,color:#fff
    style SAMPLER fill:#533483,stroke:#e94560,color:#fff
```

| Training Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch Size | 128 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=2e-3) |
| Loss Function | Focal Loss (gamma=2.0, smoothing=0.01) |
| Gradient Clipping | max_norm=1.0 |
| Mixup Probability | 30% |
| Mixup Alpha | 0.1 |

---

## 7. Model Architectures

### 7.1 TCN (Temporal Convolutional Network) — Best Overall

```mermaid
flowchart LR
    IN["Input\nB, 195, 100"] --> PROJ["Conv1d 1x1\n195 to 256\n+ BN + GELU"]
    PROJ --> B1["TCNBlock\nd=1"]
    B1 --> B2["TCNBlock\nd=2"]
    B2 --> B3["TCNBlock\nd=4"]
    B3 --> B4["TCNBlock\nd=8"]
    B4 --> B5["TCNBlock\nd=16"]
    B5 --> B6["TCNBlock\nd=32"]
    B6 --> GAP["AdaptiveAvgPool1d\nB, 256, 1"]
    GAP --> DROP["Dropout 0.2"]
    DROP --> FC["Linear\n256 to 2"]
    FC --> OUT["SAFE / DANGER"]

    style IN fill:#1a1a2e,stroke:#e94560,color:#fff
    style PROJ fill:#16213e,stroke:#0f3460,color:#fff
    style B6 fill:#0f3460,stroke:#53a8b6,color:#fff
    style GAP fill:#533483,stroke:#e94560,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

**TCNBlock (Residual):**
```
Input ---+--- Conv1d(k=3, dilation=d) -> BN -> GELU -> Conv1d(k=3, dilation=d) -> BN -> GELU ---+--- GELU -> Output
         |                                                                                       |
         +--------------------------------------- Residual Skip --------------------------------+
```

### 7.2 ConvNeXt 1D

```mermaid
flowchart LR
    IN["Input\nB, 195, 100"] --> STEM["Conv1d 4x1\nstride=2\n195 to 128\n+ BN"]
    STEM --> B1["ConvNeXt\nBlock x4"]
    B1 --> GAP["AdaptiveAvgPool1d"]
    GAP --> LN["LayerNorm 128"]
    LN --> FC["Linear\n128 to 2"]
    FC --> OUT["SAFE / DANGER"]

    style IN fill:#1a1a2e,stroke:#e94560,color:#fff
    style STEM fill:#16213e,stroke:#0f3460,color:#fff
    style B1 fill:#0f3460,stroke:#53a8b6,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

**ConvNeXtBlock (Inverted Bottleneck):**
```
Input ---+--- DWConv1d(k=7, groups=dim) -> LayerNorm -> Linear(dim to 4*dim) -> GELU -> Linear(4*dim to dim) ---+--- Output
         |                                                                                                       |
         +------------------------------------------ Residual Skip ---------------------------------------------+
```

### 7.3 MLP-Mixer 1D

```mermaid
flowchart LR
    IN["Input\nB, 195, 100"] --> PROJ["Conv1d\nk=5, stride=5\n195 to 128\nSeqLen: 100 to 20"]
    PROJ --> TR["Transpose\nB, 20, 128"]
    TR --> M1["MixerBlock x4"]
    M1 --> MEAN["Mean Pooling\nB, 128"]
    MEAN --> LN["LayerNorm"]
    LN --> FC["Linear\n128 to 2"]
    FC --> OUT["SAFE / DANGER"]

    style IN fill:#1a1a2e,stroke:#e94560,color:#fff
    style PROJ fill:#16213e,stroke:#0f3460,color:#fff
    style M1 fill:#0f3460,stroke:#53a8b6,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

**MixerBlock:**
```
Input ---+--- LayerNorm -> Transpose -> Linear(20 to 40) -> GELU -> Linear(40 to 20) -> Transpose ---+--- (Token Mixing)
         |                                                                                            |
         +-------------------------------------- Residual -------------------------------------------+
      ---+--- LayerNorm -> Linear(128 to 256) -> GELU -> Linear(256 to 128) ---+--- (Channel Mixing) -> Output
         |                                                                      |
         +--------------------------- Residual --------------------------------+
```

### 7.4 Transformer 1D

```mermaid
flowchart LR
    IN["Input\nB, 195, 100"] --> TR["Transpose\nB, 100, 195"]
    TR --> PROJ["Linear\n195 to 128"]
    PROJ --> PE["Positional\nEncoding\nSinusoidal"]
    PE --> ENC["Transformer\nEncoder\n4 layers\n4 heads\nFFN=256\ndrop=0.2"]
    ENC --> MEAN["Mean Pooling\nB, 128"]
    MEAN --> LN["LayerNorm"]
    LN --> FC["Linear\n128 to 2"]
    FC --> OUT["SAFE / DANGER"]

    style IN fill:#1a1a2e,stroke:#e94560,color:#fff
    style PE fill:#533483,stroke:#e94560,color:#fff
    style ENC fill:#e94560,stroke:#fff,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

### 7.5 InceptionTime

```mermaid
flowchart LR
    IN["Input\nB, 195, 100"] --> INC1["InceptionModule 1\n195 to 4x64 = 256"]
    INC1 --> INC2["InceptionModule 2\n256 to 4x64 = 256"]
    INC2 --> GAP["AdaptiveAvgPool1d"]
    GAP --> FC["Linear\n256 to 2"]
    FC --> OUT["SAFE / DANGER"]

    style IN fill:#1a1a2e,stroke:#e94560,color:#fff
    style INC1 fill:#0f3460,stroke:#53a8b6,color:#fff
    style INC2 fill:#0f3460,stroke:#53a8b6,color:#fff
    style OUT fill:#2b9348,stroke:#fff,color:#fff
```

**InceptionModule (Multi-Scale):**
```
Input ---+--- Conv1d(k=1) ----------------------------------+
         |                                                   |
         +--- Conv1d(k=3, pad=1) ----------------------------+--- Concatenate -> BatchNorm -> GELU -> Output
         |                                                   |
         +--- Conv1d(k=5, pad=2) ----------------------------+
         |                                                   |
         +--- MaxPool1d(k=3, pad=1) -> Conv1d(k=1) ---------+
```

---

## 8. Validation & Test-Time Augmentation (TTA)

```mermaid
flowchart TD
    BEST["Best Model\nby val Macro F1"] --> ORIG["Forward Pass 1\nClean input"]
    BEST --> TTA1["Forward Pass 2\n+ Gaussian noise sigma=0.01"]
    BEST --> TTA2["Forward Pass 3\n+ Gaussian noise sigma=0.01"]
    BEST --> TTA3["Forward Pass 4\n+ Gaussian noise sigma=0.01"]

    ORIG & TTA1 & TTA2 & TTA3 --> AVG["Average Softmax\nProbabilities\nEnsemble of 4 passes"]

    AVG --> THRESH["Threshold Sweep\n0.30 to 0.70, step 0.01\nMaximize Macro F1"]

    THRESH --> METRICS["Per-Fold Metrics\nAcc, F1, Recall, Precision"]

    style BEST fill:#e94560,stroke:#fff,color:#fff
    style AVG fill:#533483,stroke:#e94560,color:#fff
    style THRESH fill:#0f3460,stroke:#53a8b6,color:#fff
    style METRICS fill:#2b9348,stroke:#fff,color:#fff
```

---

## 9. Post-Processing Pipeline (Production Deployment)

```mermaid
flowchart LR
    RAW["Raw Softmax\nProbability\np of DANGER"] --> MED["Temporal Smoothing\nMedian Filter\nkernel=3 windows"]
    MED --> THRESH["Fixed Threshold\np greater than 0.77"]
    THRESH --> DEC{"DANGER?"}
    DEC -->|"Yes"| STOP["Emergency Stop\nHalt Cobot Arm"]
    DEC -->|"No"| CONT["Continue\nNormal Operation"]

    style RAW fill:#1a1a2e,stroke:#e94560,color:#fff
    style MED fill:#533483,stroke:#e94560,color:#fff
    style THRESH fill:#0f3460,stroke:#53a8b6,color:#fff
    style STOP fill:#e94560,stroke:#fff,color:#fff
    style CONT fill:#2b9348,stroke:#fff,color:#fff
```

**Impact of Post-Processing:**

| Metric | Raw Model | + Smoothing + Threshold 0.77 |
|---|---|---|
| Danger Precision | 83.92% | **95.09%** |
| Danger Recall | 82.84% | **98.44%** |
| False Positives | ~1,324 | **434** (-67%) |
| False Negatives | ~142 | **133** (-6%) |

---

## 10. Complete Data Flow Summary

```mermaid
flowchart TD
    subgraph DATA["Phase 1: Data Pipeline"]
        A["60 Subjects\n65 IMU Channels\n200 Hz"] --> B["179 Trials\nLabelled: SAFE / DANGER"]
        B --> C["69094 Windows\n0.5s, 50% overlap\nZ-Score Normalised"]
    end

    subgraph TRAIN["Phase 2: Training x5 Folds x 30 Epochs"]
        C --> D["WeightedRandomSampler\nBalance SAFE/DANGER"]
        D --> E["GPU: add_derivatives\n65 to 195 channels"]
        E --> F["Augment: Scale Jitter\n+ Noise + Mixup 30%"]
        F --> G["Model Forward Pass"]
        G --> H["Focal Loss\ngamma=2.0, weighted"]
        H --> I["AdamW + OneCycleLR\n+ Grad Clip 1.0"]
    end

    subgraph EVAL["Phase 3: Evaluation"]
        I --> J["Best Checkpoint\nmax val Macro F1"]
        J --> K["Test-Time Augmentation\n4x forward passes"]
        K --> L["Threshold Optimisation"]
    end

    subgraph POST["Phase 4: Production Post-Processing"]
        L --> M["Median Filter\nkernel=3"]
        M --> N["Fixed Threshold\np greater than 0.77"]
        N --> O["SAFE or DANGER"]
    end

    style DATA fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TRAIN fill:#16213e,stroke:#e94560,color:#fff
    style EVAL fill:#0f3460,stroke:#53a8b6,color:#fff
    style POST fill:#533483,stroke:#2b9348,color:#fff
```

---

## 11. Final Model Rankings

| Rank | Model | Accuracy | Macro F1 | Danger Recall | Danger Precision |
|---|---|---|---|---|---|
| 1 | **TCN** | 99.22% | 97.88% | 98.34% | 95.09% |
| 2 | **ConvNeXt 1D** | 99.17% | 97.76% | 97.84% | 95.62% |
| 3 | **MLP-Mixer 1D** | 99.28% | 97.34% | 95.87% | 95.49% |
| 4 | **InceptionTime** | 98.30% | 96.22% | 97.53% | 89.62% |
| 5 | **Transformer 1D** | 96.47% | 92.50% | 95.85% | 79.71% |

> [!NOTE]
> Rankings are based on the **0.77 threshold + Median Filter** post-processing configuration. All metrics are computed over the full 69,094-window dataset using saved best weights from 5-fold cross-validation.
