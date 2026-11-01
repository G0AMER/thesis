<!-- converted from 3rd_research_thesis_18Fév2026.docx -->

# Real-World Robotics Datasets for Incremental Learning and Lifelong Learning
## 1. OpenLORIS-Object (Lifelong Robotic Vision Object Recognition)
Overview
OpenLORIS-Object is a robotic vision dataset designed to evaluate lifelong and incremental learning algorithms in real robotic settings. It was collected with RGB-D sensors mounted on mobile robots operating in indoor environments (e.g., homes and offices), capturing variations in illumination, occlusion, object scale, and clutter. (Lifelong Robotic Vision)
Contents
RGB and Depth images of everyday objects
~69 object instances across 19 categories
Multiple environmental factors (illumination, occlusion, pixel size, clutter)
Temporal image sequences with environmental variance
~215,000 total images for benchmarking incremental tasks (Lifelong Robotic Vision)
Why It’s Useful
Real robot perception challenges (not synthetic)
Designed specifically for incremental/lifelong object recognition
Includes task sequences for incremental learning benchmarks (Lifelong Robotic Vision)
Links
Dataset page & downloads: https://lifelong-robotic-vision.github.io/dataset/Data_Object-Recognition.html (Lifelong Robotic Vision) https://www.kaggle.com/datasets/zhedamai/openlorisobject/discussion?sort=undefined
Code repository & scripts: https://github.com/lifelong-robotic-vision/OpenLORIS-Object (GitHub)

## 2. OpenLORIS-Scene (Lifelong SLAM and Scene Understanding)
Overview
OpenLORIS-Scene provides multi-sensor sequences captured by a wheeled robot in real indoor scenes. It includes visual and inertial sensor data collected at different times to reflect environment changes caused by human activity, illumination, and object movement. (Lifelong Robotic Vision)
Contents
Color + depth images (RealSense D435i)
Stereo fisheye images (RealSense T265)
IMU measurements (accelerometer + gyro)
Wheel odometry data
Ground-truth trajectories via motion capture or LiDAR SLAM
Multiple sequences per scene for long-term change evaluation (Lifelong Robotic Vision)
Why It’s Useful
Real robot dataset for lifelong SLAM
Supports incremental evaluation (sequence order matters)
Includes sensor metadata and calibration information (Lifelong Robotic Vision)
Links
Dataset page & metadata: https://lifelong-robotic-vision.github.io/dataset/scene.html (Lifelong Robotic Vision) https://huggingface.co/datasets/shixuesong/openloris-scene/tree/main/package

## 3. OpenLORIS-Location (Indoor Visual Localization)
Overview
OpenLORIS-Location focuses on visual localization within indoor spaces, captured by robots over different times and conditions. Images describe the same real world location under varying lighting and dynamics, useful for incremental/long-term localization tasks. (Lifelong Robotic Vision)
Contents
~1553 RGB images across 93 locations
Multiple images per location with appearance changes
Indoor scenes: office, home, corridor, market, restaurant, station
Conditions include illumination changes and dynamic object presence (Lifelong Robotic Vision)
Why It’s Useful
Real world location changes over time
Useful for incremental visual localization / mapping research (Lifelong Robotic Vision)
Links
Dataset page & metadata: https://lifelong-robotic-vision.github.io/dataset/location.html (Lifelong Robotic Vision) https://github.com/lifelong-robotic-vision/OpenLORIS-Location/releases

## 4. LIBERO (Lifelong Robot Learning Benchmarks)
Overview
LIBERO is a robot manipulation benchmark suite designed explicitly for lifelong and incremental skill learning research. It includes multiple versions with varying numbers of tasks, emphasizing generalization and skill transfer across tasks. (lifelong-robot-learning.github.io)
Contents
libero_spatial: spatial reasoning tasks
libero_object: object manipulation tasks
libero_goal: goal-oriented tasks
libero_100: 100 diverse manipulation tasks
Tasks represent incremental sequences of manipulation challenges (lifelong-robot-learning.github.io)
Why It’s Useful
Structured for incremental skill learning evaluation
Growing number of tasks allows testing scalability and transfer (lifelong-robot-learning.github.io)
Links
Dataset documentation & downloads: https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html (lifelong-robot-learning.github.io)https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets

## 5. Open X-Embodiment (Cross-Robot Learning Dataset)
Overview
Open X-Embodiment is a large unified robotics dataset designed to support multi-embodiment learning and skill transfer across a variety of robot types (arms, humanoids, quadrupeds) with real data. (GitHub)
Contents & Metadata
Robot episodes captured in RLDS (Robotics Learning Data Storage) format
Metadata captures:
camera intrinsics/extrinsics
robot kinematics/limits
task descriptors / skill IDs
success/failure labels
scene identifiers and contextual information
Includes hundreds of thousands of episodes for cross-task and cross-robot learning (GitHub)
Why It’s Useful
Multi-robot, multi-skill real data
Ideal for embodied learning and transfer
Standardized metadata amenable to incremental learning pipelines (GitHub)
Links
GitHub & metadata references: https://github.com/google-deepmind/open_x_embodiment (GitHub)
Project overview & metadata info: https://robotics-transformer-x.github.io (project website referenced on dataset pages) (Emergent Mind)

# How to Use the Metadata
For each dataset, metadata is available either on the dataset pages or through accompanying GitHub repositories:
| Dataset | Where to Find Metadata |
| --- | --- |
| OpenLORIS-Object | Dataset page & GitHub repo (labels, difficulty, tasks) (Lifelong Robotic Vision) |
| OpenLORIS-Scene | Dataset page (sensor calibration, ground-truth, timestamps) (Lifelong Robotic Vision) |
| OpenLORIS-Location | Page with scene and location groupings (Lifelong Robotic Vision) |
| LIBERO | Documentation lists task splits, task metadata (lifelong-robot-learning.github.io) |
| Open X-Embodiment | GitHub and spreadsheet with metadata fields (robot type, skill labels) (GitHub) |

## References for Dataset Metadata
OpenLORIS main page with links to all dataset variants and papers. (Lifelong Robotic Vision)
Detailed descriptions of OpenLORIS-Object environmental factors for incremental learning. (Lifelong Robotic Vision)
Sensor modalities and ground truth in OpenLORIS-Scene. (Lifelong Robotic Vision)
LIBERO dataset listings and download instructions. (lifelong-robot-learning.github.io)
Open X-Embodiment dataset structure and metadata formats. (DeepWiki)


- travailler sur l’aspect collaboratif avec ces datatsets
















## 1. HARMONIC — Human & Robot Multimodal Collaborative Dataset
Description:
A multimodal dataset of human interactions with a 6-DOF robot arm during assistive eating tasks. It includes:
EMG (forearm) → muscle activation
Eye gaze + egocentric and third-person video
Robot joint positions
Joystick commands
Body pose, hand pose, facial keypoints
This is research data collected for shared autonomy and intent prediction. (arXiv)
Metadata & Facts
Subjects: 24 participants
Modalities: video (multiphttps://journals.sagepub.com/doi/epub/10.1177/02783649211050677le views), EMG, gaze, robot actions
Formats: CSV, YAML, videos
Dataset / Metadata:
ArXiv Paper (with data description):
https://arxiv.org/abs/1807.11154 (arXiv)
(Download links typically found via authors’ project page in the paper or consortium pages — search for “HARMONIC dataset download” linked from the paper.)
## 2. MultiPhysio-HRC — Multimodal Human-Robot Collaboration Physiology Dataset
Description:
A large dataset capturing physiological, behavioral, and audio data in real human-robot collaboration scenarios. Modalities include:
EEG (brain activity)
ECG (heart)
EMG (muscle)
EDA (skin conductance)
Respiration
Voice & video
Designed for human status estimation (stress/cognitive load) during collaborative tasks with robots. (arXiv)
Metadata & Facts
Rich physiological sensor streams
Baseline annotations (psychological labels)
Good for affective computing + HRC learning
Dataset / Metadata:
Data described in the paper; many such datasets provide download links or contact for access:
https://www.mdpi.com/2218-6581/14/12/184 (MDPI)
## 3. RoboMNIST — Multi-Modal Multi-Robot Activity Dataset
Description:
A dataset using two Franka Emika robot arms with data from:
WiFi channel state information (CSI)
Video
Audio
The dataset captures robot activities in context and is multimodal, though not wearable in the classic IMU/EEG sense. It’s research-oriented and stored in Figshare/GitHub. (Nature)
Metadata & Facts
Multimodal: WiFi, video, audio
Robot activity recognition focus
Documentation and example loaders on GitHub
Dataset / Metadata:
Figshare repository + GitHub:
https://github.com/SiamiLab/RoboMNIST (Nature)

4. A Dataset of Standard and Abrupt Industrial Gestures Recorded Through MIMUs
Description:
The dataset introduced in this paper (often referred to as DASIG) is created to support research in human–robot collaboration (HRC) with a focus on motion recognition, ergonomics, safety, and gesture analysis in industrial contexts. It specifically includes both standard industrial gestures and unexpected/abrupt movements that may occur due to human inattention or environmental triggers, which is critical for safe HRC.
## Dataset Content
### Sensor Modality
MIMUs (Magneto-Inertial Measurement Units)
Each MIMU provides:
3-axis accelerometer data
3-axis gyroscope data
(possibly) magnetometer data depending on device used
This multimodal inertial data tracks human arm and body motion during gesture execution, capturing both smooth and sudden movements imposed by alarms or hazards.
### Gesture Types
The dataset comprises two main categories:
Standard Industrial Movements
Typical gestures such as pick-and-place, reaching, lifting
Representative of regular operations in manufacturing or assembly
Abrupt / Unexpected Movements
Movements triggered by visual or acoustic alarms
Represent sudden, atypical motion due to inattention, hazards, or safety responses
This is important for real HRC safety research, because robots must detect and adapt to these events reliably.
Dataset / Metadata:
https://zenodo.org/records/17660014







# Best Way to Build a Wearable-Controlled Human–Robotic Arm Using Incremental Learning
## 1. Core Design Principle (Very Important)
Incremental learning should learn the mapping from human intent → robot skill,
not raw motor commands.
Why?
Wearable signals are noisy
Humans vary over time
Robot safety must be guaranteed
So the robot does not directly imitate the wearable — it interprets it.
## 2. System Architecture (Research-Grade & Safe)
Wearable Sensors (EMG / IMU / Gloves)
↓
Human Intent Encoder (Neural Network)
↓
Incremental Learning Module
(Intent → Skill / Parameters)
↓
Skill Library (Frozen + Expandable)
↓
Motion Planner (IK / Trajectory)
↓
Low-Level Control (Impedance / PID)
↓
Robotic Arm
This architecture is critical and widely accepted in HRI research.
## 3. Wearable Device: What to Use (Realistic Options)
### Option A — EMG Armband (Best for Intent)
Muscle activation signals
Strong link to human intent
Used in prosthetics & HRI
Examples:
Myo armband
Delsys EMG
Custom EMG sensors
### Option B — IMU-based Wearables (Best for Motion)
Wrist orientation
Arm motion
Low cost and robust
Often combined with EMG.
### Option C — Smart Gloves (Dexterous Control)
Finger flexion
Grasp intent
Force estimation
### Best Practice
Combine EMG + IMU
EMG → what the human wants
IMU → how the human moves
## 4. What Incremental Learning Learns (Key Decision)
You have three good learning targets, ordered from safest to most powerful:
### Skill Selection (Best Starting Point)
Wearable → Which skill to activate
Pick
Place
Rotate
Hand-over
Incremental learning adds new skills over time.
### Skill Parameter Adaptation
Wearable → Parameters of a skill
Speed
Grip force
Direction
Precision level
Incremental learning adapts to:
New users
Fatigue
Context changes
### Policy Refinement (Advanced)
Wearable → high-level action policy
More flexible
Needs safety constraints
## 5. Incremental Learning Algorithms (Best Choices)
### Replay + Distillation (DER / DER++) — Best Overall
Why it’s ideal here:
Wearable data drifts over time
Human behavior changes
Keeps old skills stable
How it works in your system:
Store a small buffer of old wearable–skill pairs
Store old network outputs
Train on:
New wearable data
Old “remembered” behavior
Used in robot manipulation and HRI research.
### Adapter Modules (Highly Scalable)
One adapter per user or skill
Shared backbone
Perfect for:
Multi-user systems
Long-term deployment
### Avoid (for this use case)
Pure EWC / SI → too weak for noisy wearable data
Progressive NNs → too heavy for real robots
## 6. Human Teaching Loop (Incremental Learning in Practice)
### Step-by-step:
Human wears device
Demonstrates or corrects motion
Robot executes a safe version
Human gives:
Confirmation
Correction
Data stored incrementally
Model updated without forgetting
This creates a human-in-the-loop lifelong learner.
## 7. Dataset Strategy (Realistic & Publishable)
### Initial Dataset
5–10 skills
2–3 users
10–20 demonstrations per skill
Record:
Wearable signals
Robot state
Selected skill
Outcome (success/failure)
### Incremental Phase
Each update introduces:
A new user or
A new skill or
A change in wearable placement
This is perfect for evaluating incremental learning.
## 8. Safety Layer (Mandatory)
Even if learning fails:
Impedance control
Joint limits
Speed limits
Human override
The learned model never directly controls motors.
## 9. Example Technology Stack
| Component | Recommendation |
| --- | --- |
| Middleware | ROS 2 |
| Wearable | EMG + IMU |
| Learning | PyTorch |
| IL Method | DER + replay buffer |
| Skills | DMPs or motion primitives |
| Planner | MoveIt |
| Controller | Impedance control |
## - download datasets
- MultiPhysio-HRC — Multimodal Human-Robot Collaboration Physiology Dataset
what did they do with it, papers…
how can we exploit it in our project