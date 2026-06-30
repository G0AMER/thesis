# PhD Project Architecture — Incremental HRC with Behavioral Adaptation

## Table of Contents
1. [Global Architecture Overview](#1-global-architecture-overview)
2. [Layer 0 — Sensor & Data Acquisition](#2-layer-0--sensor--data-acquisition)
3. [Layer 1 — Human Perception & State Estimation](#3-layer-1--human-perception--state-estimation)
4. [Layer 2 — Operator Style Modeling](#4-layer-2--operator-style-modeling)
5. [Layer 3 — Incremental Inverse Reinforcement Learning](#5-layer-3--incremental-inverse-reinforcement-learning)
6. [Layer 4 — Personalized Behavior Generation](#6-layer-4--personalized-behavior-generation)
7. [Layer 5 — Motion Planning & Execution](#7-layer-5--motion-planning--execution)
8. [Layer 6 — Safety & Compliance](#8-layer-6--safety--compliance)
9. [Layer 7 — Collaboration Metrics & Evaluation](#9-layer-7--collaboration-metrics--evaluation)
10. [Algorithm Comparison Plan (6 Experiments)](#10-algorithm-comparison-plan-6-experiments)
11. [Detailed Step-by-Step Work Plan (36 Steps)](#11-detailed-step-by-step-work-plan-36-steps)
12. [Project Directory Map](#12-project-directory-map)
13. [Publication Strategy](#13-publication-strategy)

---

## 1. Global Architecture Overview

The system is organized in **8 layers**, from raw sensors to high-level evaluation. Each layer is independently testable, which is critical for PhD ablation studies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 7 — METRICS & EVALUATION                   │
│  Joint-action entropy · Workspace overlap · Mutual idle time        │
│  Subjective questionnaires (NASA-TLX, SUS, fluency scales)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ measures
┌────────────────────────────▼────────────────────────────────────────┐
│                LAYER 6 — SAFETY & COMPLIANCE (always active)        │
│  Impedance control · Joint/velocity limits · Collision avoidance    │
│  Human override · ISO 10218 / ISO/TS 15066 compliance               │
└────────────────────────────┬────────────────────────────────────────┘
                             │ constrains
┌────────────────────────────▼────────────────────────────────────────┐
│              LAYER 5 — MOTION PLANNING & EXECUTION                  │
│  MoveIt 2 · Trajectory interpolation · Impedance/PID control        │
│  DMP execution · Real-time joint commands                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ receives trajectory
┌────────────────────────────▼────────────────────────────────────────┐
│           LAYER 4 — PERSONALIZED BEHAVIOR GENERATION                │
│  CVAE trajectory generator · Temporal scheduler · Style-conditioned │
│  DMP parameterizer · Safety/efficiency/comfort optimizer            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ receives reward + style vector
┌────────────────────────────▼────────────────────────────────────────┐
│         LAYER 3 — INCREMENTAL INVERSE REINFORCEMENT LEARNING        │
│  Online MaxEnt IRL · Deep MaxEnt IRL · Adversarial IRL (AIRL)      │
│  Feedback integration · Adaptive forgetting · Reward model          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ receives style embedding + corrections
┌────────────────────────────▼────────────────────────────────────────┐
│              LAYER 2 — OPERATOR STYLE MODELING                      │
│  LSTM / Transformer sequence encoder · Strategy clustering          │
│  Style vector extraction · Operator profile management              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ receives features
┌────────────────────────────▼────────────────────────────────────────┐
│          LAYER 1 — HUMAN PERCEPTION & STATE ESTIMATION              │
│  Intent recognition · Gesture detection · Cognitive load est.       │
│  Emotion/stress detection · Action phase segmentation               │
└────────────────────────────┬────────────────────────────────────────┘
                             │ receives raw signals
┌────────────────────────────▼────────────────────────────────────────┐
│            LAYER 0 — SENSOR & DATA ACQUISITION                      │
│  EMG · IMU · RGB-D camera · Microphone · Force/Torque sensor        │
│  Robot joint encoders · ROS 2 message synchronization               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles
1. **Modularity** — each layer can be swapped/ablated independently (critical for PhD experiments)
2. **Safety-first** — Layer 6 wraps ALL motor commands; the learned model NEVER bypasses it
3. **Incremental by design** — no module assumes a fixed dataset; all accept streaming data
4. **Multi-algorithm** — each layer implements multiple algorithms for comparative study
5. **ROS 2 native** — all inter-layer communication uses ROS 2 topics/services/actions

---

## 2. Layer 0 — Sensor & Data Acquisition

### Purpose
Collect, synchronize, and publish all raw signals as ROS 2 messages with aligned timestamps.

### Sensors

| Sensor | Signal | ROS 2 Topic | Rate | Use |
|--------|--------|-------------|------|-----|
| EMG armband (e.g., Delsys Trigno) | 8-ch muscle activation | `/human/emg` | 2000 Hz | Intent, fatigue, force estimation |
| IMU wristband | 3-axis accel + gyro + mag | `/human/imu` | 200 Hz | Hand trajectory, gesture |
| RGB-D camera (RealSense D435i) | Color + depth + built-in IMU | `/camera/color`, `/camera/depth` | 30 Hz | Body/hand pose, scene understanding |
| Microphone | Audio waveform | `/human/audio` | 16 kHz | Voice commands (stop, non, ok) |
| F/T sensor (wrist-mounted) | 6-axis force/torque | `/robot/ft_sensor` | 500 Hz | Physical corrections, contact detection |
| Robot joint encoders | Joint positions, velocities, torques | `/robot/joint_states` | 1000 Hz | Robot state, compliance |

### Synchronization Strategy
- Use ROS 2 `message_filters::ApproximateTimeSynchronizer`
- All messages stamped with hardware-triggered timestamps
- Record to **rosbag2** for offline replay and dataset creation

### Data Recording Format
```
data/raw/
  session_YYYYMMDD_HHMMSS/
    rosbag2/               # Full ROS 2 bag
    metadata.yaml          # Operator ID, task, session number
    annotations.csv        # Manual labels (corrections, phases)
```

---

## 3. Layer 1 — Human Perception & State Estimation

### Purpose
Extract meaningful features from raw signals: what is the human doing, intending, and feeling?

### Module 1.1 — Intent Recognition

**Input**: EMG + IMU + RGB-D (hand pose)
**Output**: Predicted next action (discrete) + confidence

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **LSTM classifier** | Sequence model | Baseline; proven on EMG time series |
| **1D-CNN + BiLSTM** | Hybrid | Better temporal feature extraction |
| **Temporal Transformer** | Attention-based | State-of-art on multimodal sequences; captures long-range dependencies |

**Incremental aspect**: New intents (new assembly steps) added over time → use adapter heads or expandable output layer.

### Module 1.2 — Gesture Detection

**Input**: IMU + RGB-D (body/hand keypoints via MediaPipe)
**Output**: Gesture class (standard industrial / abrupt / corrective)

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **DTW + kNN** | Template matching | Simple baseline; works with few samples |
| **1D-CNN on IMU windows** | Deep learning | Fast, proven on DASIG-like data |
| **ST-GCN (Spatial-Temporal Graph ConvNet)** | Graph neural network | Models skeleton topology; state-of-art for action recognition |

**Safety-critical**: abrupt gesture detection must run at ≤50ms latency.

### Module 1.3 — Human State Estimation (Cognitive Load / Stress)

**Input**: EMG + audio + RGB-D (facial AUs) + optionally EEG/ECG/EDA
**Output**: Stress level (low/medium/high), cognitive load estimate

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **Random Forest on handcrafted features** | Classical ML | Interpretable baseline; proven on MultiPhysio-HRC |
| **Multimodal late fusion (CNN per modality + MLP)** | Deep fusion | Standard multimodal approach |
| **Cross-modal Transformer** | Attention fusion | Learns inter-modality relationships; best for heterogeneous signals |

### Module 1.4 — Action Phase Segmentation

**Input**: All sensor streams
**Output**: Current phase of the collaborative task (approach / manipulate / handover / wait / correct)

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **HMM (Hidden Markov Model)** | Probabilistic | Classical; interpretable transitions |
| **TCN (Temporal Convolutional Network)** | Deep learning | Causal convolutions; good for online segmentation |
| **CTC-based segmentation** | Sequence-to-sequence | Handles variable-length phases without explicit boundaries |

---

## 4. Layer 2 — Operator Style Modeling

### Purpose
Encode each operator's unique behavioral signature into a compact **style vector** that conditions all downstream modules.

### Module 2.1 — Sequence Encoder (Trajectory → Embedding)

**Input**: Sequence of (hand position, action choice, timing, force) over a task episode
**Output**: Fixed-size style embedding vector z_style ∈ ℝ^d (d = 64 or 128)

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **LSTM autoencoder** | Seq2seq | Baseline; compresses trajectory into bottleneck |
| **Temporal Transformer encoder** | Self-attention | Captures non-local dependencies in long action sequences |
| **Variational Recurrent Neural Network (VRNN)** | Probabilistic | Provides uncertainty over style; supports generative sampling |

The encoder is trained on full task episodes. The bottleneck representation IS the style vector.

### Module 2.2 — Strategy Clustering

**Input**: Style vectors from multiple episodes/operators
**Output**: K strategy clusters + cluster assignments

| Algorithm to Compare | Type | Why Test It |
|---------------------|------|-------------|
| **K-Means on style vectors** | Centroid-based | Simple baseline |
| **HDBSCAN** | Density-based | Discovers clusters without specifying K; handles outliers |
| **Gaussian Mixture Model (GMM)** | Probabilistic | Soft assignments; provides cluster likelihood |

**PhD contribution angle**: Show that operators naturally cluster into distinct strategy types, and that the robot can identify which cluster a new operator belongs to within N episodes.

### Module 2.3 — Operator Profile Manager

Maintains a persistent profile per operator:
```python
class OperatorProfile:
    operator_id: str
    style_vector: np.ndarray          # Current style embedding (updated incrementally)
    strategy_cluster: int             # Assigned cluster
    reward_model: nn.Module           # Personal IRL reward network
    adapter_weights: dict             # Personal adapter parameters
    interaction_history: deque        # Recent episodes (bounded buffer)
    session_count: int
    last_updated: datetime
```

Profiles are saved to disk and loaded when the operator is identified (e.g., via badge scan or face recognition).

---

## 5. Layer 3 — Incremental Inverse Reinforcement Learning

### Purpose
Infer what the operator wants (reward function) from observed behavior and corrections, and **update this model incrementally** across sessions without catastrophic forgetting.

### Formal Problem

Given:
- Demonstration trajectories τ = {(s_t, a_t)}_{t=0}^{T} from operator
- Corrections c_t (physical, verbal, facial) as reward signals
- Prior reward model R_θ from previous sessions

Learn: Updated R_θ' that explains current operator preferences while retaining knowledge from past sessions.

### Module 3.1 — IRL Algorithm Candidates

| Algorithm | Formulation | Why Test It | Incremental Adaptation |
|-----------|-------------|-------------|----------------------|
| **MaxEnt IRL** | P(τ) ∝ exp(R_θ(τ)); maximize likelihood of demos | Classical; well-understood theory | Re-run on buffer of old + new demos |
| **Deep MaxEnt IRL** | Neural network reward R_θ(s,a); same MaxEnt objective | Scales to high-dim state spaces | Fine-tune with replay (DER++) |
| **AIRL (Adversarial IRL)** | GAN-style: discriminator recovers reward, generator is policy | Disentangles reward from dynamics; transferable | Fine-tune discriminator incrementally |
| **T-REX / D-REX** | Learn reward from ranked trajectory pairs | No need for optimal demos; handles suboptimal behavior | Add new ranked pairs incrementally |
| **Bayesian IRL** | Posterior over reward functions P(R|τ) | Principled uncertainty; naturally incremental via posterior update | True Bayesian update (computationally heavy) |

### Module 3.2 — Making IRL Incremental (Core PhD Contribution)

For each IRL algorithm above, we implement **three incremental strategies** and compare:

| Strategy | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **A) Replay buffer (DER++)** | Store subset of old (state, action, reward_logit) tuples; replay during new updates | Strong anti-forgetting; proven | Requires memory |
| **B) EWC regularization** | Penalize changes to important reward-network parameters | No memory needed | Weaker on long sequences |
| **C) Adapter modules** | Freeze shared reward backbone; per-operator adapter layers | Scales to many operators; no forgetting by design | Less flexible |

This gives us **5 IRL algorithms × 3 incremental strategies = 15 combinations** to benchmark.

### Module 3.3 — Multi-Modal Feedback Integration

Human corrections arrive through multiple channels. Each is converted to a reward signal:

| Feedback Channel | Sensor | Reward Signal Mapping |
|-----------------|--------|-----------------------|
| Physical correction (guiding robot arm) | F/T sensor | Direction & magnitude → gradient on reward near current state |
| Voice command ("stop", "non", "bien") | Microphone → speech recognition | Negative/positive scalar reward at current timestep |
| Facial expression (frown, smile) | RGB-D → facial AU detection | Continuous valence → soft reward modifier |
| Pause / hesitation | IMU + action timer | Longer-than-usual pause → uncertainty signal → reduce reward confidence |
| Explicit demonstration | All sensors | Full trajectory → standard IRL demonstration |

**Fusion**: Weighted sum with learned (or hand-tuned) channel reliability weights:
$$R_{feedback}(s_t) = \sum_{c \in \text{channels}} w_c \cdot r_c(s_t)$$

### Module 3.4 — Adaptive Forgetting Mechanism

The operator may change strategy. The system must forget outdated preferences gracefully.

| Mechanism to Compare | How It Works |
|---------------------|-------------|
| **Exponential decay on replay buffer** | Older samples have decaying probability of being replayed |
| **Sliding window** | Only keep last W episodes in buffer |
| **Change-point detection + reset** | Monitor reward model loss; if spike detected → partial reset of recent adapter weights |
| **Soft attention over history** | Transformer-based attention over stored episodes; naturally down-weights irrelevant past |

---

## 6. Layer 4 — Personalized Behavior Generation

### Purpose
Given a task goal and the operator's style vector + inferred reward, generate a robot trajectory that feels natural and aligned to that specific operator.

### Module 4.1 — CVAE Trajectory Generator (Primary)

**Architecture**:
```
Encoder:  q(z | τ_robot, task, z_style) → latent code z
Decoder:  p(τ_robot | z, task, z_style) → trajectory
Prior:    p(z | task, z_style)
```

**Input**: Task descriptor (target positions, sequence) + style vector z_style
**Output**: Full robot trajectory τ = {(q_t, q̇_t)}_{t=0}^{T} (joint positions + velocities)

**Training**: On recorded demonstrations from Phase 1, conditioned on operator style.

| Variant to Compare | Architecture Detail | Why Test It |
|-------------------|-------------------|-------------|
| **Vanilla CVAE** | MLP encoder/decoder | Baseline |
| **Recurrent CVAE** | LSTM encoder/decoder | Better for variable-length trajectories |
| **Transformer CVAE** | Transformer encoder/decoder | Captures long-range trajectory dependencies |
| **Conditional Diffusion Model** | Denoising diffusion conditioned on style | State-of-art generative quality (2024-2025 trend); higher compute |

### Module 4.2 — DMP Parameterizer (Alternative / Complementary)

Instead of generating raw trajectories, generate **DMP parameters** conditioned on style:

```
Style vector z_style → MLP → DMP weights (w_1, ..., w_K) + timing τ + goal g
```

DMPs provide built-in smoothness and perturbation recovery. The style only modifies the shape and timing.

| Variant to Compare | Description |
|-------------------|-------------|
| **Standard DMP** | Fixed basis functions, learned weights |
| **ProMP (Probabilistic Movement Primitives)** | Gaussian distribution over trajectories; blends via conditioning |
| **KMP (Kernelized Movement Primitives)** | Non-parametric; adapts from few demos |

### Module 4.3 — Temporal Scheduler

Adjusts WHEN the robot acts to match operator rhythm.

**Input**: Operator's observed action timing (from Layer 2) + current task state
**Output**: Robot action timestamps (when to start each subtask)

| Approach | Description |
|----------|-------------|
| **Phase-coupled oscillator** | Robot's action clock synchronizes to human's rhythm like coupled pendulums |
| **Predictive timing model (LSTM)** | Predict operator's next action time → schedule robot to finish just before |
| **Reactive timing** | Simple: wait for human phase completion signal, then act with learned offset |

### Module 4.4 — Joint Optimization

The generated trajectory must satisfy multiple objectives simultaneously:

$$\min_{\tau} \underbrace{\lambda_1 \cdot \mathcal{L}_{style}(\tau, z_{style})}_{\text{style alignment}} + \underbrace{\lambda_2 \cdot \mathcal{L}_{safety}(\tau)}_{\text{min. distance to human}} + \underbrace{\lambda_3 \cdot \mathcal{L}_{energy}(\tau)}_{\text{joint torque cost}} + \underbrace{\lambda_4 \cdot \mathcal{L}_{time}(\tau)}_{\text{task completion time}}$$

Solved via gradient-based optimization on the CVAE latent space, or as a constrained optimization on DMP parameters.

---

## 7. Layer 5 — Motion Planning & Execution

### Purpose
Convert the generated trajectory into safe, executable robot commands.

### Components

| Component | Tool | Role |
|-----------|------|------|
| **Trajectory validation** | MoveIt 2 | Check joint limits, self-collision, workspace bounds |
| **Collision checking** | MoveIt 2 + Octomap | Avoid human body (from depth camera) |
| **Trajectory interpolation** | MoveIt 2 Servo | Smooth real-time interpolation at 1kHz |
| **Controller** | ros2_control | Joint position / velocity / impedance control |

### Controller Candidates

| Controller | Properties | Use Case |
|------------|-----------|----------|
| **Position PID** | Stiff tracking | Non-contact phases |
| **Impedance control** | Compliant; safe for contact | Handover, co-manipulation |
| **Admittance control** | Force input → position output | Physical corrections by human |
| **Variable impedance** | Stiffness adapts to context | Learned from IRL reward (PhD contribution) |

**PhD angle**: The inferred reward function from Layer 3 modulates impedance — higher uncertainty → lower stiffness → more compliant.

---

## 8. Layer 6 — Safety & Compliance

### Purpose
Guarantee human safety regardless of what the learning system produces. This layer is **non-learnable** and **always active**.

### Safety Mechanisms

| Mechanism | Implementation | Standard |
|-----------|---------------|----------|
| **Speed & force limits** | Hard caps in ros2_control | ISO/TS 15066 (PFL mode) |
| **Safety-rated monitored stop** | Laser scanner zones → stop if human enters | ISO 10218-2 |
| **Power & force limiting** | F/T sensor threshold → compliant mode | ISO/TS 15066 Table A.2 |
| **Collision detection** | Torque observer + depth camera | — |
| **Abrupt motion detector** | DASIG-trained model from Layer 1.2 | ISO 10218-1 |
| **Emergency stop** | Physical button + voice "STOP" | IEC 60204-1 |
| **Workspace separation** | Octomap zones; robot reduces speed near human | ISO/TS 15066 SSM |

### Safety Watchdog Node (ROS 2)
Runs at highest priority. Subscribes to ALL sensor topics. Can override any motor command within 10ms.

---

## 9. Layer 7 — Collaboration Metrics & Evaluation

### Purpose
Quantify the quality of human-robot collaboration objectively and subjectively.

### Objective Metrics

| Metric | Formula / Description | What It Measures |
|--------|----------------------|-----------------|
| **Joint-action entropy** | $H = -\sum p(a_h, a_r) \log p(a_h, a_r)$ | Predictability of joint actions; lower = more fluent |
| **Workspace overlap rate** | $\frac{|W_h \cap W_r|}{|W_h \cup W_r|}$ over time | Spatial interference; lower = better coordination |
| **Mutual idle time** | $T_{idle} = \sum \max(0, t_{wait}^h) + \max(0, t_{wait}^r)$ | Time wasted waiting; lower = better temporal alignment |
| **Task completion time** | Wall-clock time per assembly cycle | Efficiency |
| **Human intervention rate** | # corrections / # episodes | How much the human must fix the robot |
| **Reward model convergence** | Loss of IRL reward model over sessions | Learning speed |
| **Style alignment score** | $\cos(z_{style}^{predicted}, z_{style}^{actual})$ | How well the robot matches operator style |
| **Forgetting metric** | Performance on old preferences after learning new ones | Catastrophic forgetting resistance |

### Subjective Metrics

| Questionnaire | What It Measures |
|---------------|-----------------|
| **NASA-TLX** | Perceived workload |
| **System Usability Scale (SUS)** | Usability |
| **Robotic Social Attributes Scale (RoSAS)** | Perceived robot warmth/competence |
| **Custom fluency questionnaire** | Perceived collaboration smoothness |
| **Trust in Automation (Jian et al.)** | Trust level |

---

## 10. Algorithm Comparison Plan (6 Experiments)

This is the experimental backbone of the thesis. Each experiment produces a paper.

### Experiment 1 — Incremental Learning Method Comparison
**Question**: Which IL method best prevents forgetting of operator preferences in sequential HRC sessions?

| Algorithm | Family | Implementation |
|-----------|--------|---------------|
| DER++ | Replay | `src/incremental_learning/replay_based/der.py` |
| EWC | Regularization | `src/incremental_learning/regularization_based/ewc.py` |
| SI | Regularization | `src/incremental_learning/regularization_based/si.py` |
| LwF | Distillation | `src/incremental_learning/regularization_based/lwf.py` |
| Adapter Modules | Parameter isolation | `src/incremental_learning/adapter_based/adapters.py` |
| PackNet | Parameter isolation | `src/incremental_learning/adapter_based/packnet.py` |

**Dataset**: HARMONIC (24 subjects, sequential sessions) + LIBERO (manipulation tasks)
**Metrics**: Average accuracy after N tasks, backward transfer, forward transfer, memory cost
**Output**: `experiments/exp01_il_comparison/`

---

### Experiment 2 — IRL Algorithm Comparison for Online Preference Learning
**Question**: Which IRL algorithm best infers operator preferences incrementally from demonstrations + corrections?

| Algorithm | Incremental Strategy | Implementation |
|-----------|---------------------|---------------|
| MaxEnt IRL + DER++ replay | Replay | `src/irl/maxent_irl/` |
| MaxEnt IRL + EWC | Regularization | `src/irl/maxent_irl/` |
| Deep MaxEnt IRL + DER++ | Replay | `src/irl/deep_maxent_irl/` |
| Deep MaxEnt IRL + Adapters | Isolation | `src/irl/deep_maxent_irl/` |
| AIRL + DER++ | Replay | `src/irl/adversarial_irl/` |
| T-REX + sliding window | Window | `src/irl/online_irl/` |

**Dataset**: HARMONIC (joystick + corrections as demonstrations) + custom simulated HRC environment
**Metrics**: Reward prediction accuracy, policy quality (cumulative reward), adaptation speed, forgetting
**Output**: `experiments/exp02_irl_comparison/`

---

### Experiment 3 — Operator Style Modeling
**Question**: Which sequence model best encodes operator-specific behavioral signatures?

| Encoder | Clustering | Combination |
|---------|-----------|-------------|
| LSTM autoencoder | K-Means | Baseline |
| LSTM autoencoder | HDBSCAN | Better outlier handling |
| Transformer encoder | K-Means | Stronger encoding |
| Transformer encoder | GMM | Probabilistic clusters |
| VRNN | HDBSCAN | Uncertainty-aware |

**Dataset**: HARMONIC + DASIG (60 subjects; natural inter-subject variability)
**Metrics**: Cluster purity, silhouette score, style vector discriminability (operator classification accuracy), few-shot operator identification
**Output**: `experiments/exp03_style_modeling/`

---

### Experiment 4 — Behavior Generation Quality
**Question**: Which generative model produces the most natural and style-aligned trajectories?

| Generator | Trajectory Repr. | Combination |
|-----------|-----------------|-------------|
| Vanilla CVAE | Raw joint trajectory | Baseline |
| Recurrent CVAE | Raw joint trajectory | Variable-length |
| Transformer CVAE | Raw joint trajectory | Long-range |
| Conditional Diffusion | Raw joint trajectory | SOTA generative |
| CVAE → DMP params | DMP | Smooth + perturbation robust |
| CVAE → ProMP params | ProMP | Probabilistic blending |

**Dataset**: Recorded operator demonstrations (Phase 1 data collection)
**Metrics**: Trajectory smoothness (jerk), style alignment score, task success rate, human preference rating (A/B test)
**Output**: `experiments/exp04_behavior_gen/`

---

### Experiment 5 — Full System Integration
**Question**: Does the complete pipeline (perception → style → IRL → generation → execution) improve collaboration versus a non-adaptive baseline?

| Condition | Description |
|-----------|-------------|
| **Baseline** | Fixed pre-programmed robot behavior |
| **Style-only** | Robot adapts trajectory style but no reward learning |
| **IRL-only** | Robot learns rewards but no style conditioning |
| **Full system** | All layers active |

**Protocol**: Within-subjects design, counterbalanced order, N ≥ 12 participants
**Metrics**: All objective + subjective metrics from Layer 7
**Output**: `experiments/exp05_full_system/`

---

### Experiment 6 — Longitudinal Study
**Question**: Does collaboration fluency improve over multiple sessions (days/weeks)?

| Group | Description |
|-------|-------------|
| **Control** | Fixed robot, 10 sessions over 2 weeks |
| **Experimental** | Adaptive robot (full system), 10 sessions over 2 weeks |

**Protocol**: Between-subjects, N ≥ 8 per group, track all metrics per session
**Metrics**: Learning curves, session-over-session improvement, final fluency comparison
**Output**: `experiments/exp06_longitudinal/`

---

## 11. Detailed Step-by-Step Work Plan (36 Steps)

### PHASE 1 — Foundation & Data (Months 1–6)

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 1 | Set up ROS 2 workspace (Humble/Iron), MoveIt 2, ros2_control | Working robot simulation (Gazebo + RVIZ) | 2 weeks |
| 2 | Interface wearable sensors (EMG + IMU) with ROS 2 | ROS 2 driver nodes publishing synchronized data | 2 weeks |
| 3 | Interface RGB-D camera + microphone with ROS 2 | Full sensor suite publishing on topics | 1 week |
| 4 | Implement rosbag2 recording pipeline + metadata logger | Automated data acquisition scripts | 1 week |
| 5 | Download & explore existing datasets (HARMONIC, MultiPhysio-HRC, DASIG) | Data loaders, EDA notebooks, baseline statistics | 2 weeks |
| 6 | Design collaborative assembly task (e.g., bolt-tightening sequence) | Task protocol document, physical workspace layout | 2 weeks |
| 7 | Recruit participants (≥ 10 operators) & obtain ethics approval | IRB/ethics approval, consent forms | 4 weeks (parallel) |
| 8 | Conduct data collection sessions (≥ 20 episodes per operator) | Raw dataset: `data/raw/` | 4 weeks |
| 9 | Annotate data: action phases, corrections, strategy labels | Annotated dataset: `data/processed/` | 2 weeks |
| 10 | Implement data preprocessing pipeline (filtering, segmentation, normalization) | Reproducible preprocessing scripts | 2 weeks |

### PHASE 2 — Core Algorithm Development (Months 7–18)

#### Step Group A — Human Perception (Months 7–9)

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 11 | Implement intent recognition (LSTM, CNN+BiLSTM, Transformer) | 3 trained models + comparison table | 3 weeks |
| 12 | Implement gesture detection (DTW+kNN, 1D-CNN, ST-GCN) | 3 trained models + comparison on DASIG | 3 weeks |
| 13 | Implement human state estimation (RF, late fusion, cross-modal Transformer) | 3 models + comparison on MultiPhysio-HRC | 3 weeks |
| 14 | Implement action phase segmentation (HMM, TCN, CTC) | 3 models + comparison | 2 weeks |
| 15 | Write **Paper 1**: "Multimodal Human Perception for Adaptive HRC" | Submitted to conference / journal | 2 weeks |

#### Step Group B — Operator Style Modeling (Months 10–12)

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 16 | Implement style encoders (LSTM-AE, Transformer, VRNN) | 3 encoder architectures | 3 weeks |
| 17 | Implement clustering (K-Means, HDBSCAN, GMM) on style vectors | Cluster analysis + visualization | 2 weeks |
| 18 | Run Experiment 3 (style modeling comparison) | Full results table + ablation | 2 weeks |
| 19 | Implement operator profile manager with persistent storage | Profile CRUD + incremental update | 2 weeks |
| 20 | Write **Paper 2**: "Operator Style Modeling for Personalized HRC" | Submitted | 2 weeks |

#### Step Group C — Incremental IRL (Months 13–18) ← Core contribution

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 21 | Implement MaxEnt IRL (batch baseline) | Working reward inference | 2 weeks |
| 22 | Implement Deep MaxEnt IRL | Neural reward network | 2 weeks |
| 23 | Implement AIRL | Adversarial reward recovery | 3 weeks |
| 24 | Implement T-REX / D-REX | Ranked trajectory reward learning | 2 weeks |
| 25 | Add incremental wrappers: DER++ replay, EWC, Adapter modules | 3 strategies × 4-5 algorithms | 4 weeks |
| 26 | Implement multi-modal feedback integration (force, voice, face) | Feedback-to-reward pipeline | 3 weeks |
| 27 | Implement adaptive forgetting (decay, window, change-point, attention) | 4 forgetting mechanisms | 3 weeks |
| 28 | Run Experiment 1 (IL comparison) + Experiment 2 (IRL comparison) | Full benchmark results | 3 weeks |
| 29 | Write **Paper 3**: "Incremental IRL for Online Preference Learning in HRC" ← **Main contribution paper** | Submitted to top venue (ICRA, RSS, CoRL, RA-L) | 3 weeks |

### PHASE 3 — Behavior Generation & Integration (Months 19–27)

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 30 | Implement CVAE variants (vanilla, recurrent, transformer, diffusion) | 4 generative models | 4 weeks |
| 31 | Implement DMP/ProMP parameterizers | 2 movement primitive approaches | 3 weeks |
| 32 | Implement temporal scheduler (oscillator, LSTM predictive, reactive) | 3 timing approaches | 2 weeks |
| 33 | Implement joint trajectory optimization (style + safety + energy + time) | Multi-objective optimizer | 3 weeks |
| 34 | Run Experiment 4 (behavior generation comparison) | Results + human A/B preference test | 3 weeks |
| 35 | Integrate full pipeline (L0→L7) on real robot | End-to-end working system | 6 weeks |
| 36 | Write **Paper 4**: "Style-Conditioned Trajectory Generation for Personalized Cobots" | Submitted | 3 weeks |

### PHASE 4 — Validation & Thesis Writing (Months 28–36)

| Step | Task | Deliverable | Duration |
|------|------|-------------|----------|
| 37 | Run Experiment 5 (full system ablation, N ≥ 12) | Statistical analysis of all metrics | 4 weeks |
| 38 | Run Experiment 6 (longitudinal, 10 sessions × 2 groups, 2 weeks) | Learning curves + questionnaire analysis | 6 weeks |
| 39 | Write **Paper 5**: "Longitudinal Evaluation of Incremental HRC" | Submitted | 3 weeks |
| 40 | Write thesis chapters (contexte, état de l'art, contributions ×3, validation, conclusion) | Complete manuscript | 12 weeks |
| 41 | Internal review, revision, defense preparation | Final manuscript + defense slides | 4 weeks |

---

## 12. Project Directory Map

```
phd_project/
├── ARCHITECTURE.md              ← This document
├── configs/                     ← YAML configs for all experiments
├── data/
│   ├── raw/                     ← Rosbag2 recordings
│   ├── processed/               ← Preprocessed & annotated
│   ├── replay_buffer/           ← DER++ stored exemplars
│   └── models/                  ← Saved model checkpoints
├── experiments/
│   ├── exp01_il_comparison/     ← IL algorithm benchmark
│   ├── exp02_irl_comparison/    ← IRL algorithm benchmark
│   ├── exp03_style_modeling/    ← Style encoder + clustering
│   ├── exp04_behavior_gen/      ← Trajectory generation quality
│   ├── exp05_full_system/       ← Integrated system ablation
│   └── exp06_longitudinal/      ← Multi-session study
├── notebooks/                   ← EDA, visualization, analysis
├── scripts/                     ← Data collection, training, evaluation
├── src/
│   ├── perception/
│   │   ├── intent_recognition/  ← LSTM, CNN+BiLSTM, Transformer
│   │   ├── gesture_detection/   ← DTW+kNN, 1D-CNN, ST-GCN
│   │   └── human_state/         ← RF, late fusion, cross-modal Transformer
│   ├── style_modeling/
│   │   ├── sequence_models/     ← LSTM-AE, Transformer encoder, VRNN
│   │   ├── clustering/          ← K-Means, HDBSCAN, GMM
│   │   └── style_encoder/       ← Unified style vector API
│   ├── incremental_learning/
│   │   ├── replay_based/        ← DER, DER++, iCaRL
│   │   ├── regularization_based/← EWC, SI, LwF, MAS
│   │   ├── adapter_based/       ← Adapter modules, PackNet
│   │   └── benchmarks/          ← Unified evaluation harness
│   ├── irl/
│   │   ├── maxent_irl/          ← Standard + Deep MaxEnt
│   │   ├── deep_maxent_irl/     ← Neural reward network
│   │   ├── adversarial_irl/     ← AIRL
│   │   ├── online_irl/          ← T-REX, D-REX, Bayesian
│   │   └── benchmarks/          ← IRL evaluation harness
│   ├── behavior_generation/
│   │   ├── cvae/                ← Vanilla, Recurrent, Transformer, Diffusion
│   │   ├── trajectory_planner/  ← DMP, ProMP, KMP
│   │   └── temporal_scheduler/  ← Oscillator, LSTM predictor, reactive
│   ├── metrics/
│   │   ├── alignment/           ← Style alignment score
│   │   ├── fluidity/            ← Joint entropy, idle time, overlap
│   │   └── safety/              ← Distance, speed, force metrics
│   ├── safety/
│   │   ├── collision_avoidance/ ← Octomap integration
│   │   ├── speed_limiter/       ← ISO/TS 15066 limits
│   │   └── impedance_controller/← Variable impedance
│   ├── ros2_interface/
│   │   ├── nodes/               ← All ROS 2 node implementations
│   │   ├── launch/              ← Launch files
│   │   ├── config/              ← ROS 2 parameter files
│   │   └── msgs/                ← Custom message definitions
│   └── utils/
│       ├── data_loading/        ← Dataset loaders (HARMONIC, DASIG, etc.)
│       ├── visualization/       ← Plotting, trajectory viz
│       └── logging/             ← Experiment tracking (W&B / MLflow)
└── tests/                       ← Unit + integration tests
```

---

## 13. Publication Strategy

| Paper | Target Venue | Tier | Timeline |
|-------|-------------|------|----------|
| **P1**: Multimodal Human Perception for Adaptive HRC | ROMAN / HRI Conference | B/A | Month 9 |
| **P2**: Operator Style Modeling for Personalized HRC | IROS / Frontiers in Robotics | A | Month 12 |
| **P3**: Incremental IRL for Online Preference Learning ← **Main** | ICRA / RSS / CoRL / RA-L | A* | Month 18 |
| **P4**: Style-Conditioned Trajectory Generation | RA-L / Autonomous Robots | A | Month 27 |
| **P5**: Longitudinal Evaluation of Adaptive HRC | HRI / THRI Journal | A | Month 33 |

Total: **5 publications**, covering all 4 research axes + validation.

---

## Summary of Algorithm Choices to Benchmark

| Layer | Algorithms to Compare | Total |
|-------|----------------------|-------|
| Intent recognition | LSTM, CNN+BiLSTM, Transformer | 3 |
| Gesture detection | DTW+kNN, 1D-CNN, ST-GCN | 3 |
| Human state | RF, Late Fusion, Cross-modal Transformer | 3 |
| Phase segmentation | HMM, TCN, CTC | 3 |
| Style encoder | LSTM-AE, Transformer, VRNN | 3 |
| Clustering | K-Means, HDBSCAN, GMM | 3 |
| IRL algorithm | MaxEnt, Deep MaxEnt, AIRL, T-REX, Bayesian | 5 |
| IL strategy | DER++, EWC, Adapters | 3 |
| Forgetting | Decay, Window, Change-point, Attention | 4 |
| Trajectory gen. | CVAE (×3 variants), Diffusion, DMP, ProMP | 6 |
| Temporal sched. | Oscillator, LSTM, Reactive | 3 |
| Controller | Position PID, Impedance, Admittance, Variable | 4 |

**Total unique algorithm implementations: ~43**
**Total experimental comparisons across 6 experiments: sufficient for a strong PhD thesis.**
