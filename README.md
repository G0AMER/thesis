# PhD Thesis — Ameur Gargouri

## Title

**Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale**

## Core Research Question

How can a cobot **incrementally learn** an operator's preferences, strategies, and operational style during physical collaboration — evolving from standardized interaction to a **personalized, highly efficient partnership** — without explicit reprogramming?

---

## 1. Four Research Axes

| Axis                                               | Goal                                                                                                     | Key Methods                                                                                                                         |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **A1 — Modeling human operational schemas** | Learn each operator's habits, action sequences, timing                                                   | LSTM / Transformers on hand trajectories & assembly order; clustering of strategies                                                 |
| **A2 — Incremental Inverse RL (IRL)**       | Infer the operator's reward function online, integrating physical/verbal/facial corrections in real time | Online MaxEnt IRL / Deep MaxEnt IRL; adaptive forgetting mechanism; multi-signal feedback integration                               |
| **A3 — Personalized behavior generation**   | Produce robot trajectories, speeds, and timing aligned with the learned operator style                   | Conditional VAE (CVAE) conditioned on style vector; collaborative temporal scheduling; joint safety/efficiency/comfort optimization |
| **A4 — Alignment & fluidity metrics**       | Objectively quantify how well the robot's behavior matches the human's                                   | Joint-action entropy; workspace overlap rate; mutual idle time                                                                      |

---

## 2. Incremental Learning Landscape

### Three Scenarios (van de Ven & Tolias 2022)

- **Task-IL** — separate tasks, task ID given at inference
- **Domain-IL** — same task, shifting data distribution (concept drift)
- **Class-IL** — growing class set, no task ID (hardest, most realistic)

**Central challenge**: catastrophic forgetting & the stability–plasticity dilemma.

### Algorithm Families

| Family                     | Representative Methods             | Relevance to Thesis                                                                                    |
| -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Replay / Experience Replay | iCaRL, GDumb,**DER / DER++** | **Primary choice** — stores input+logit pairs; stable with many tasks; fits noisy wearable data |
| Regularization             | EWC, SI, LwF, MAS                  | Lighter but weaker for complex perception; useful as baselines                                         |
| Parameter Isolation        | PackNet, Progressive NNs, DEN      | Zero forgetting by design but heavy; skill-library use case                                            |
| Prompt / Adapter Tuning    | Prompt vectors, Adapter modules    | **Highly scalable** for multi-user / multi-skill; shared backbone + per-user adapters            |
| Generative Replay          | GAN / VAE / Diffusion synthesis    | Synthetic rare-event replay; simulation-to-real transfer                                               |

**Best choices for this project**: **DER/DER++** (overall best for drifting wearable data) and **Adapter Modules** (scalable multi-user deployment).

---

## 3. RL vs. IRL

| Aspect      | RL                                         | IRL                                                          |
| ----------- | ------------------------------------------ | ------------------------------------------------------------ |
| Direction   | Reward → Policy                           | Behavior → Reward                                           |
| Input       | Environment + explicit rewards             | Expert demonstrations (no explicit rewards)                  |
| Output      | Optimal policy                             | Inferred reward function                                     |
| Thesis role | Derives robot policy from inferred rewards | **Core axis A2** — infers operator preferences online |

**Key IRL algorithms**: MaxEnt IRL, Adversarial IRL (AIRL), Apprenticeship Learning.

**Thesis innovation**: Online/incremental IRL with multi-modal feedback (physical corrections, voice, facial expressions) and adaptive forgetting.

---

## 4. Datasets

### Standard IL Benchmarks

CIFAR-100, ImageNet, CORe50, PermutedMNIST, Cityscapes, KITTI, BDD100K, nuScenes, CARLA, OpenAI Gym, RLBench, RoboNet

### Real-World Robotics / Lifelong Learning

| Dataset                     | Type                                           | Key Value                                                    |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **OpenLORIS-Object**  | RGB-D, 69 objects, 19 categories, ~215K images | Incremental robotic object recognition under real conditions |
| **OpenLORIS-Scene**   | Multi-sensor SLAM (color+depth+IMU+odometry)   | Lifelong SLAM with environment changes                       |
| **LIBERO**            | 100 manipulation tasks in 4 splits             | Lifelong robot skill learning benchmark                      |
| **Open X-Embodiment** | Multi-robot, multi-skill, RLDS format          | Cross-embodiment transfer                                    |

### Human–Robot Collaboration (Most Relevant)

| Dataset                   | Modalities                                                                                          | Thesis Use                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **HARMONIC**        | EMG, gaze, egocentric/third-person video, joystick, robot joints, body/hand/face pose (24 subjects) | Intent prediction, shared autonomy, multimodal fusion for adaptive cobots       |
| **MultiPhysio-HRC** | EEG, ECG, EDA, respiration, EMG, voice, facial AUs                                                  | Cognitive load / stress estimation → adaptive robot behavior                   |
| **DASIG**           | MIMU (accel+gyro+magneto), 60 subjects, standard + abrupt industrial gestures                       | Safety — detect abrupt/panic movements, trigger robot slowdown; ISO compliance |
| **RoboMNIST**       | WiFi CSI, video, audio (2 Franka arms)                                                              | Multi-modal robot activity recognition                                          |

### Dataset Links

| Dataset            | URL                                                                            |
| ------------------ | ------------------------------------------------------------------------------ |
| OpenLORIS-Object   | https://lifelong-robotic-vision.github.io/dataset/Data_Object-Recognition.html |
| OpenLORIS-Scene    | https://lifelong-robotic-vision.github.io/dataset/scene.html                   |
| OpenLORIS-Location | https://lifelong-robotic-vision.github.io/dataset/location.html                |
| LIBERO             | https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html  |
| Open X-Embodiment  | https://github.com/google-deepmind/open_x_embodiment                           |
| HARMONIC           | https://arxiv.org/abs/1807.11154                                               |
| MultiPhysio-HRC    | https://www.mdpi.com/2218-6581/14/12/184                                       |
| DASIG              | https://zenodo.org/records/17660014                                            |
| RoboMNIST          | https://github.com/SiamiLab/RoboMNIST                                          |

---

## 5. System Architecture

```
Wearable Sensors (EMG + IMU)
        ↓
Human Intent Encoder (Neural Net)
        ↓
Incremental Learning Module (DER++ / Adapters)
  (Intent → Skill / Parameters)
        ↓
Skill Library (Frozen base + expandable)
        ↓
CVAE Style-Conditioned Trajectory Generator
        ↓
Motion Planner (MoveIt) + Temporal Scheduler
        ↓
Low-Level Control (Impedance / PID)
        ↓
Robotic Arm
```

**Safety layer** (mandatory, always active): impedance control, joint/speed limits, human override. The learned model never directly controls motors.

---

## 6. Technology Stack

| Component            | Choice                                                              |
| -------------------- | ------------------------------------------------------------------- |
| Middleware           | **ROS 2**                                                     |
| Wearable             | EMG + IMU                                                           |
| Learning framework   | **PyTorch**                                                   |
| IL method            | **DER/DER++** + replay buffer; Adapter modules for multi-user |
| IRL                  | Online MaxEnt IRL / Deep MaxEnt IRL                                 |
| Behavior generator   | **CVAE** conditioned on style vector                          |
| Skill representation | DMPs / motion primitives                                            |
| Motion planner       | **MoveIt**                                                    |
| Controller           | Impedance control                                                   |

---

## 7. Four-Phase Work Plan

| Phase             | Description                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase 1** | Data collection (actions, gaze, decisions of multiple operators on same task) → base model of style variations                                              |
| **Phase 2** | Implement online IRL algorithm; design feedback integration (voice "stop" → negative reward, physical correction → model update)                           |
| **Phase 3** | Train CVAE on Phase 1 data for style-conditioned trajectory generation; integrate into robot planner                                                         |
| **Phase 4** | Longitudinal validation — control group (fixed robot) vs. experimental group (incremental+aligned); measure learning curves, fluidity metrics, satisfaction |

---

## 8. Expected Contributions & Impact

### Contributions

- **New formal framework** for incremental preference learning in physical HRC
- **Interactive, robust IRL algorithm** (online, multi-modal feedback, adaptive forgetting)
- **Generative model** for personalized robot behavior synthesis (CVAE)

### Impact

- Reduced setup time for new operators/products
- Increased comfort & acceptance (robot adapts to human, not the inverse)
- Robustness to individual variability, making cobotique accessible to more profiles

---

## 9. Key References

- Avaei, A.; van der Spaa, L.; Peternel, L.; Kober, J. *An Incremental Inverse Reinforcement Learning Approach for Motion Planning with Separated Path and Velocity Preferences.* Robotics 2023, 12, 61. https://doi.org/10.3390/robotics12020061
- Deshpande, S., Walambe, R., Kotecha, K. et al. *Advances and applications in inverse reinforcement learning: a comprehensive review.* Neural Comput & Applic 37, 11071–11123 (2025). https://doi.org/10.1007/s00521-025-11100-0
- Urrea, C. *Hybrid Deep Learning-Reinforcement Learning for Adaptive Human-Robot Task Allocation in Industry 5.0.* Systems 2025, 13, 631. https://doi.org/10.3390/systems13080631
- Ahmad Farooq and Kamran Iqbal. *A Survey of Reinforcement Learning for Optimization in Automation.* arXiv:2502.09417v1, 13 Feb 2025
- https://doi.org/10.1016/j.chbah.2023.100018
- https://doi.org/10.1146/annurev-control-042920-093225
- https://doi.org/10.1177/0278364917690593
- https://doi.org/10.3389/frobt.2024.1455375
- https://doi.org/10.1109/TCDS.2024.3454779
- https://doi.org/10.1109/ACCESS.2024.3451663
- https://arxiv.org/pdf/2404.18713v1
- https://doi.org/10.48550/arXiv.2209.11908
- https://doi.org/10.1038/s42003-021-02891-8

---

## Keywords

Intelligence artificielle, robotique, cobotique, apprentissage incrémental, apprentissage par renforcement inverse, interaction homme-machine, adaptation comportementale, apprentissage à long terme.
