# Novel Research Ideas & Open Datasets for HRC

**Author:** Ameur Gargouri  
**Thesis:** *Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale*  
**Date:** September 15, 2026  
**Scope:** 100% independent research ideas based on **new, previously unused datasets** discovered via live web search.

---

## 1. Executive Summary: What Sets These Apart?

Your previous works investigated:
- **MultiPhysio-HRC** (Wearable physiological signals: EEG, ECG, EDA, EMG, Respiration for 5 cognitive/task states).
- **DASIG** (Wearable MIMU inertial body sensors for 1D danger state classification).
- **HARMONIC** (Assistive 6-DOF robotic arm trajectory & joystick data for continual learning with FLAIR).

To ensure complete **scientific independence** and open new high-impact avenues, the proposed ideas below pivot to:
1. **Camera-less / Non-wearable Spatial Safety** (3D LiDAR point clouds & robot kinematics).
2. **Dynamic Dexterous Handovers** (Multi-view 3D vision + high-DOF robotic hand grasping).
3. **Ergonomic-Driven Behavioral Adaptation** (Whole-body biomechanics & ground-reaction forces).
4. **Hierarchical Industrial Assembly & Digital Twins** (Multi-camera RGB + skeleton sequences).
5. **Multimodal Industrial Workflow Monitoring** (Multi-view RGB-D + wrist IMU with fine-grained temporal boundaries).

---

## 2. Dataset Benchmarking Matrix

| Dataset Name | Primary Modalities | Year / Source | Key Advantage | Accessibility |
| :--- | :--- | :--- | :--- | :--- |
| **LiHRA** | 3D LiDAR point clouds, 3D keypoints, robot joint states | 2025 (IROS / Zenodo) | Captures intentional contact vs. unintentional collision in 6 industrial scenarios | Open Access (Zenodo) |
| **DexH2R** | 4.2K trials, 456K frames, RGB-D, 3D hand poses, ShadowHand teleop | 2024/2025 (ArXiv / GitHub) | Benchmark for dynamic, flowing human-to-robot handover with dexterous hands | Open Access (GitHub/Project Page) |
| **COMFI** | 4.5h sync RGB, markerless MoCap, joint angles, ground reaction force, robot telemetry | 2024/2025 (GitHub / Zenodo) | Real industrial collaborative tasks with ground-truth biomechanical strain | Open Access (GitHub: `MaximeSabbah/comfi-usage`) |
| **InHARD (+ InHARD-DT)** | >2M frames, 16 subjects, 13 industrial classes, 3 RGB views, skeleton | 2022–2024 (Zenodo / GitHub) | Industrial assembly actions + Digital Twin (VR) synthetic counterpart | Open Access (Zenodo / GitHub) |
| **MIAM** | 290 mins untrimmed video, multi-view RGB, depth, 9-axis hand IMU | 2024 (HuggingFace / ArXiv) | Realistic assembly/disassembly with natural pauses, hesitation, and engagement shifts | HuggingFace Datasets |

---

## 3. Detailed Novel Research Proposals

---

### Proposal 1: Non-Invasive Spatial Risk Forecasting via 4D Spatio-Temporal LiDAR Point Clouds

* **Target Dataset:** **LiHRA** (LiDAR-based Risk Assessment in HRI, IROS 2025)
* **Core Problem:** Wearable sensors (like MIMUs in DASIG) are cumbersome, unergonomic, and prone to compliance issues in real factories. Standard 2D RGB cameras suffer from severe occlusion, varying illumination, and privacy regulations (GDPR).
* **Novel Idea:** Formulate proactive safety not as binary danger classification from wearables, but as **continuous 4D spatial-temporal Occupancy & Trajectory Forecasting** using raw LiDAR point clouds and robot forward kinematics.
* **Proposed Methodology:**
  1. **Architecture:** A 4D Point-Voxel Spatio-Temporal Transformer (PV-ST-Trans) that consumes sequential LiDAR point sweeps $\mathcal{P}_{t-\Delta t : t}$ along with the robot's planned trajectory $\mathcal{T}_{\text{robot}}$.
  2. **Predictive Horizon:** Instead of detecting a collision when it happens ($t$), predict the **Probabilistic Minimum Separation Distance (PMSD)** and voxel collision risk over the next $0.5\text{s} - 2.0\text{s}$.
  3. **Control Integration:** Map predicted PMSD to ISO/TS 15066 Speed and Separation Monitoring (SSM) dynamic protective stop envelopes.
* **Key Innovations:**
  - Zero wearables required for the human operator.
  - Predicts *where* and *when* risk occurs in 3D space, not just a global danger flag.
* **Target Venues:** IEEE Transactions on Robotics (T-RO), IEEE RA-L, or IROS.

---

### Proposal 2: Flow-Matching Dexterous Handover Policy Conditioned on Human Reach Velocity

* **Target Dataset:** **DexH2R** (Dynamic Dexterous Grasping in Human-to-Robot Handover)
* **Core Problem:** Most robotic handover studies assume the human stands still holding an object while the robot approaches (static handover), or rely on simple parallel-jaw grippers. Real human collaboration involves fluid, simultaneous, moving handovers with complex multi-finger grasps.
* **Novel Idea:** An adaptive **Conditional Flow-Matching (CFM) / Diffusion Policy** that predicts 3D multi-finger pre-grasp poses and approach velocity synchronized with the human's dynamic motion profile.
* **Proposed Methodology:**
  1. **Conditioning Signal:** Encode the human hand 3D trajectory $\mathbf{H}_{t}$ and object velocity $\mathbf{v}_{\text{obj}}$ using a lightweight temporal causal attention encoder.
  2. **Generative Pre-Grasp & Velocity Matching:** Train a Continuous Normalizing Flow (CNF) via Optimal Transport Flow Matching (OT-CFM) to generate the 24-DOF ShadowHand joint positions and wrist trajectory that intercept the object at the optimal meeting point.
  3. **Fluidity Evaluation:** Measure interaction fluency using the time-to-handover, human hesitation index (sudden deceleration), and grasp success rate.
* **Key Innovations:**
  - Upgrades HRC from static handoffs to dynamic, flowing peer-to-peer object transfers.
  - Benchmarked against DexH2R's baseline models (DynamicGrasp, DDPM).
* **Target Venues:** IEEE/RSJ IROS, CoRL (Conference on Robot Learning), or IEEE ICRA.

---

### Proposal 3: Ergonomic Co-Adaptation: Robot Trajectory Optimization Driven by Biomechanical Fatigue Forecasting

* **Target Dataset:** **COMFI** (Collaboration Oriented Markerless For Industry)
* **Core Problem:** Collaborative robots typically optimize for trajectory length or cycle time, completely ignoring the ergonomic fatigue and joint overload of the human collaborator. Over hours of work, this causes musculoskeletal disorders (MSD).
* **Novel Idea:** An **Online Ergonomic Digital Twin** that estimates real-time RULA/REBA scores and predicts muscular fatigue from markerless motion, adjusting the robot's delivery position, orientation, and pacing to actively minimize human physical strain.
* **Proposed Methodology:**
  1. **Biomechanical State Estimator:** Train a temporal graph neural network (ST-GCN) on COMFI's markerless keypoints and ground reaction forces to estimate instantaneous joint torque and cumulative ergonomic strain index $\mathcal{E}_t$.
  2. **Adaptive Motion Planning:** Formulate the robot's operational goal using Bi-Level Optimization / Model Predictive Path Integral (MPPI):
     $$\min_{\mathbf{u}_r} \mathcal{J}_{\text{task}}(\mathbf{x}_r) + \lambda \cdot \mathcal{J}_{\text{ergo}}(\mathbf{x}_h(\mathbf{u}_r))$$
     The robot presents parts closer to the operator's neutral working zone when fatigue accumulates.
* **Key Innovations:**
  - Direct connection to **Industry 5.0** (human-centric sustainable manufacturing).
  - Uses COMFI's synchronized ground-reaction forces and joint angles as objective physiological ground truth.
* **Target Venues:** IEEE Transactions on Human-Machine Systems (THMS), Robotics and Computer-Integrated Manufacturing (RCIM).

---

### Proposal 4: Cross-Modal Digital Twin Adaptation for Industrial Action Recognition & Anticipation

* **Target Dataset:** **InHARD + InHARD-DT** (Industrial Human Action Recognition Dataset + Digital Twin VR)
* **Core Problem:** Collecting annotated training data in industrial HRC lines is hazardous and expensive. Digital Twins (VR simulations) can generate infinite synthetic training samples, but models suffer severe domain shift when deployed in real factories.
* **Novel Idea:** A **Physics-Informed Cross-Modal Domain Adaptation (PI-CDA)** framework that trains action anticipation models on synthetic Digital Twin data (InHARD-DT) and adapts online to real industrial camera/skeleton feeds (InHARD) without manual real-world labels.
* **Proposed Methodology:**
  1. **Hierarchical Action Graph:** Industrial assembly actions follow strict precedence constraints (e.g., *aligning part* $\to$ *screwing* $\to$ *quality check*). Represent operational schemas as an Action Precedence Automaton.
  2. **Contrastive Sim2Real Alignment:** Contrastive feature alignment on skeleton topology and motion velocity distributions between InHARD-DT and InHARD.
  3. **Anticipation Horizon:** Predict the operator's next action $k$ seconds in advance so the cobot can pre-fetch tools or fixture parts before the human asks.
* **Key Innovations:**
  - Directly fulfills **Axis A1 of your thesis** (Modeling human operational schemas and action sequences) using a recognized industrial benchmark.
  - Evaluates zero-shot and few-shot Sim-to-Real transfer.
* **Target Venues:** IEEE Transactions on Industrial Informatics (TII), Advanced Engineering Informatics.

---

### Proposal 5: Continual Multimodal Workflow Parsing Under Temporal Shift and Operator Variability

* **Target Dataset:** **MIAM** (Multimodal Industrial Activity Monitoring, HuggingFace)
* **Core Problem:** Workers perform identical assembly sequences with vastly different styles, speeds, and intermediate hesitations. Static models overfit to specific workers or specific shifts, failing when an unfamiliar operator steps onto the line.
* **Novel Idea:** A **Multimodal Test-Time Adaptation (M-TTA)** framework for untrimmed industrial workflow videos (RGB, Depth, and wrist IMU), updating representation features on-the-fly without catastrophic forgetting of earlier assembly phases.
* **Proposed Methodology:**
  1. **Multimodal Early Fusion Backbone:** Unified temporal encoder bridging video tokens (RGB+Depth) and high-frequency wrist IMU accelerations.
  2. **Self-Supervised Test-Time Alignment:** Use cross-modal prediction (predicting IMU kinetics from video representations and vice versa) as an online proxy task to adapt to a new operator's speed and movement style without ground-truth labels.
  3. **Metric Evaluation:** Segmental F1@k, Edit Distance, and Mean Overlap on untrimmed industrial assembly recordings.
* **Key Innovations:**
  - Bridges your continual learning expertise with realistic untrimmed video/IMU streams.
  - Solves the open problem of operator-to-operator domain adaptation in industrial assembly.
* **Target Venues:** International Journal of Computer Vision (IJCV) / IEEE CVPR Workshop on Egocentric & Industrial Perception / Robotics and Autonomous Systems (RAS).

---

## 4. Strategic Comparison & Recommendation

| Idea | Core Theme | Dataset | Implementation Effort | Expected Impact | Primary Venue |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Idea 1** | 4D LiDAR Spatio-Temporal Safety | **LiHRA** (2025) | 6–8 weeks | Very High | IEEE T-RO / RA-L |
| **Idea 2** | Dexterous Flow-Matching Handover | **DexH2R** (2024) | 7–9 weeks | Extremely High | IROS / CoRL |
| **Idea 3** | Ergonomic Biomechanical Adaptation | **COMFI** (2025) | 5–7 weeks | High (Industry 5.0) | IEEE THMS / RCIM |
| **Idea 4** | Sim2Real Action Anticipation | **InHARD-DT** | 5–6 weeks | High (Thesis A1) | IEEE TII |
| **Idea 5** | Multimodal Test-Time Adaptation | **MIAM** (2024) | 6–8 weeks | High | RAS / IEEE T-ASE |

### Top Recommendation for Ameur's Next Publication:
- If you want the **fastest alignment with your Thesis core axes (Axis A1/A3)** while using brand-new data: **Idea 4 (InHARD / InHARD-DT)** or **Idea 3 (COMFI Ergonomic Adaptation)**.
- If you want the **highest technological prestige / robotics flagship venue (CoRL / IROS / T-RO)**: **Idea 2 (DexH2R Flow-Matching Handover)** or **Idea 1 (LiHRA 4D LiDAR Safety)**.
