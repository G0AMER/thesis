<!-- converted from 4th_research_thesis_20Avr2026 .docx -->

## 1. MultiPhysio-HRC: A Multimodal Physiological Signals Dataset for Industrial Human–Robot Collaboration(Dec 2025)
## Methodology
### Created a Rich Multimodal Dataset
They collected a new dataset called MultiPhysio-HRC, which includes synchronized human physiological and behavioral signals recorded during real and realistic human–robot collaboration scenarios.
The modalities included:
Electroencephalography (EEG) – brain electrical activity
Electrocardiography (ECG) – heart activity
Electrodermal activity (EDA) – skin conductance (stress indicator)
Respiration (RESP)
Electromyography (EMG) – muscle activity
Voice recordings
Facial action units (facial expression features)
These modalities together offer a comprehensive view of mental, emotional, and physical states.
### Collected Data in Diverse Tasks
The dataset was not limited to one type of task. It includes:
Controlled Cognitive Tasks
Designed to elicit different levels of:
Stress
Cognitive load
Examples include Stroop tasks, n-back working memory tests, arithmetic tasks, and problem-solving tasks.
Immersive VR Stress Scenarios
Virtual reality experiences designed to induce stress and emotional responses.
Industrial-Like Collaboration with a Robot
Participants performed:
Manual industrial tasks (e.g., battery disassembly)
Robot-assisted tasks with a collaborative robot
This places data collection in a realistic human–robot collaborative context, not just in lab conditions.
### Baseline Models & Evaluation
The authors evaluated baseline machine learning models on the dataset to:
Predict stress
Predict cognitive load
They showed that these physiological and multimodal signals do contain meaningful information about human mental states, but also highlighted the challenges in achieving high accuracy — pointing to the need for more advanced models and multimodal fusion techniques in future research.
# Using MultiPhysio-HRC in Cobot Projects:
## Human Affective & Cognitive State Awareness
Cobots traditionally respond to physical motion or proximity (e.g., force, gestures, speed).
This dataset enables robots to understand internal states such as:
Cognitive load (workload)
Stress and anxiety
Physiological arousal
Muscle fatigue (via EMG)
Emotional valence
This can be critical for adaptive behavior — e.g., a robot slowing down or offering more assistance when the human is stressed or overloaded.
Example Applications:
Safety systems that reduce robot speed during high human stress
Workload-aware task allocation
Adaptive assistance levels in collaborative assembly
## Multimodal Sensor Fusion for Better Human Perception
The dataset includes synchronized multimodal signals such as:
EEG — brain activity
ECG — heart rate dynamics
EDA — skin conductance
RESP — respiration patterns
EMG — muscle activity
Facial features
Voice recordings
This rich sensory input is ideal for research into:
Multimodal adaptation models
Deep learning fusion algorithms
Real-time intent/state estimation
Such models can make a cobot more perceptive and responsive than if it only used vision or motion data.
## Affective Human–Robot Interaction (HRI)
Understanding emotional and cognitive human states enables:
Robots that empathize or adjust behavior based on human stress
More natural and safe interactions
Reduction of human workload and errors
For example:
Robot reduces speed if human is stressed
Robot offers suggestions or pauses when workload is high
Robot adjusts communication style based on affect









2. HARMONIC: A Multimodal Dataset of Assistive Human–Robot Collaboration(Dec2021)
## Methodology
### 1) Designed a Real Assistive HRC Task
Participants collaborated with a 6-DoF robot arm (e.g., platforms from KUKA) to perform an assistive feeding/manipulation task.
The human provided high-level intent and corrective inputs; the robot executed low-level motion—typical shared autonomy.
### 2) Collected Synchronized Multimodal Data
They recorded tightly time-aligned streams from human, robot, and environment:
Human sensing
Eye gaze (binocular eye cameras)
Egocentric (first-person) RGB video
Forearm EMG (muscle activity)
Joystick inputs (control intent)
Robot sensing
Joint positions/kinematics
End-effector state
Third-person perception
Stereo video of the scene
They also provide processed annotations/features, such as:
Gaze projected into egocentric video
2D/3D body and hand pose
Facial keypoints
All modalities are time-synchronized, enabling causal modeling across signals.
### 3) Structured the Dataset for Learning Problems
They organized trials into task phases (approach, manipulation, correction, completion) so the dataset can support:
Goal/intent prediction (early inference of user goal)
Action anticipation (predict next robot motion from human signals)
Shared autonomy policies (assistive control blending)
Multimodal fusion (vision + gaze + EMG + control)
### 4) Baseline Benchmarks
The authors report baseline results for representative tasks (e.g., intent inference and action prediction) to show:
Multimodal models oDesign and Development of an Autonomous Mobile Robot for Unstructured Indoor Environmentsutperform single-modality baselines
Gaze and control inputs are strong early indicators of intent
Physiological/control cues add predictive value beyond vision alone
This establishes benchmark protocols for future methods.
## Core Contributions
A high-quality, synchronized multimodal HRC dataset (human perception + physiology + control + robot state).
Real shared-autonomy interaction, not passive observation or teleoperation logs alone.
Standardized benchmarks for intent prediction and assistive control learning.
Tooling and annotations that reduce friction for multimodal learning research.
## What others did with this dataset (domain, year of release, ...)
Where HARMONIC Fits in a Cobot System
### 1) Shared Autonomy & Intention Inference
HARMONIC includes human control inputs (joystick), gaze, EMG, egocentric/third-person vision, and robot kinematics captured during real collaborative assistance. This supports:
Early goal/intent prediction
Action anticipation (predict next robot motion from human signals)
Control blending policies for shared autonomy
test algorithms
These are core problems in cobot interaction.
### 2) Multimodal Human State Estimation
Because modalities are synchronized, you can study:
Multimodal fusion (vision + gaze + EMG + control)
Robust perception under partial observability
User adaptation / personalization across subjects
This is directly applicable to adaptive cobots that tailor assistance to the operator.
3. A Dataset of Standard and Abrupt Industrial Gestures Recorded Through MIMUs
## Methodology
1. Created the DASIG dataset
They designed and collected a dataset called DASIG (Dataset of Standard and Abrupt Industrial Gestures) using MIMUs (Magneto-Inertial Measurement Units — wearable motion sensors).
How to measure safety using this dataset, test algorithms
2. Participants & tasks
60 healthy adult participants took part.
Participants performed a sequence of:
Standard industrial gestures (e.g., typical pick-and-place tasks).
Abrupt movements triggered by stimuli such as visual alarms (lights) or auditory alarms (buzzers) to simulate unexpected situations.
3. What the dataset includes
Sensor signals from multiple MIMUs recorded during movement.
Temporal information about when alarms were triggered.
Anthropometric (body measurement) data for each participant.
Example code/scripts showing how to use the dataset.
## Why It Matters
Rich data for ML and robotics: This dataset can be used to train and test algorithms for gesture recognition, safety systems, and human-motion-aware robot control.
Focus on abrupt motion: The inclusion of sudden unplanned movements helps develop more robust and safe industrial robots that can react properly when humans behave unpredictably.
Applicable for ergonomics and safety research: Besides robotics, researchers in ergonomics, motion planning, and human–machine interaction can also use this dataset.
# Why It Is Useful for Collaborative Robots (Cobots)
The dataset contains:
Wearable MIMU (IMU + magnetometer) signals
Standard industrial gestures
Abrupt/sudden movements triggered by alarms
Multiple subjects (inter-subject variability)
Temporal annotations
This makes it valuable for:
### 1. Human Motion Recognition
You can train models to:
Recognize normal work gestures
Detect abnormal or abrupt movements
Classify motion phases
This is directly useful in:
Human intention recognition
Gesture-based robot control
Adaptive robot behavior
### 2️. Safety in Human–Robot Collaboration
Abrupt movements are extremely important in HRC.
Example uses:
Detecting panic or emergency reactions
Detecting sudden human withdrawal
Predicting collision risk
Triggering robot slowdown or stop
This aligns very well with ISO safety requirements for collaborative robots.
Respect ISO normes
