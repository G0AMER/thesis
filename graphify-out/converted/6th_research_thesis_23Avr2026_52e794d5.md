<!-- converted from 6th_research_thesis_23Avr2026.docx -->

Reference 1: Kaiwu – A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction
Authors: Shuo Jiang, Haonan Li, Ruochen Ren, Yanmin Zhou, Zhipeng Wang, Bin He
Date of release: arXiv preprint (2503.05231v2) submitted 3 June 2025 (initial version March 2025). No journal publication noted at time of writing.
Scientific details and type of work: This is both a multimodal dataset and an accompanying data-collection/annotation framework designed for sophisticated real-world assembly tasks (e.g., robotic-arm component assembly with tools such as drills, pliers, and screwdrivers). It addresses the lack of synchronized, high-quality multimodal data that includes dynamics (force/tactile) and fine-grained labeling for robot learning, dexterous manipulation, human-intention inference, and human-robot collaboration (HRC). The framework integrates wearable and environment sensors with multi-threading for real-time synchronized streaming, calibration protocols, and timestamp-based annotation tools.
Dataset details (modalities, nature, availability, fields of use):
Participants/objects: 20 subjects (mean age 23.95 ± 5.1 years, mean height 172.46 ± 6.5 cm); 30 interaction objects; 11 664 integrated action instances across 15 assembly “links” per participant (~6.3 hours total).
Modalities captured (all synchronized via absolute Unix timestamps):
Hand/forearm/upper-arm kinematics (data gloves: 19 angle sensors + 19 pressure/tactile sensors at 100 Hz; quaternions, finger angles, tactile forces).
Muscle activity (16 Trigno EMG/ACC/IMU sensors at 40 Hz).
Multi-view vision (RGB-D cameras at 60 Hz → .mkv videos, .jpg images, .png depth maps, .pcd point clouds).
High-precision motion capture (optical system, 37 markers, 60 Hz, 3D skeleton ground truth).
Eye gaze + first-person video (binocular dark-pupil tracker at 25 Hz, 0.6° accuracy).
Ambient audio (4 microphones at 50 Hz, .wav).
Acceleration/IMU data.
Data nature: Real-world, ecologically valid assembly in confined/ample spaces; raw sensor streams + processed files; fine-grained multi-level annotations (timestamp-aligned .csv): action/gesture segmentation (left/right hand, 8 gesture states), semantic segmentation (30 objects), gesture classification, 298 areas of interest (AOIs).
Availability: https://www.scidb.cn/en/detail?dataSetId=33060cd729604d2ca7d41189a9fc492b
Fields of use: Robot learning / imitation learning, dexterous manipulation, human-intention prediction, multimodal fusion, embodied AI, HRC in assembly/manufacturing scenarios. Explicitly positioned to support human-level perception and long-term robot autonomy.

Reference 2: PoseTron: Enabling Close-Proximity Human-Robot Collaboration Through Multi-human Motion Prediction
Authors: Mohammad Samin Yasar, Md Mofijul Islam, Tariq Iqbal
Date of release: March 2024 (published in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction – HRI ’24).
Scientific details and type of work: It introduces PoseTron, a model for multi-human motion prediction that enables safe and fluent close-proximity HRC. The core contribution is real-time prediction of multiple humans’ future trajectories/poses to allow robots to anticipate and avoid collisions or coordinate movements in crowded, dynamic shared workspaces. It is evaluated on human-human and human-robot collaboration scenarios.
Data/framework details: INTERACT Dataset (introduced and evaluated in the PoseTron paper)
Date of release: March 2024 (published with the PoseTron paper at the ACM/IEEE International Conference on Human-Robot Interaction – HRI ’24).
Scientific details and type of work: INTERACT is a newly introduced large-scale multimodal dataset created specifically by the PoseTron authors to address the lack of comprehensive data for close-proximity human-robot collaboration (HRC) and multi-human motion prediction. It supports the development and evaluation of motion-prediction models (including the PoseTron framework itself) by capturing realistic team dynamics in both human-human (HH) and human-robot (HR) collaboration scenarios. The dataset enables robots to anticipate multi-agent trajectories/poses for safer and more fluent coordination in crowded, dynamic shared workspaces. It is used directly in the PoseTron paper for training/evaluation of the proposed transformer-based encoder-decoder model (with conditional and multimodal attention mechanisms). No separate framework or code for data collection is released; the focus is on the dataset as a benchmark resource for HRC motion forecasting.
Dataset details (modalities, nature, availability, fields of use):
Scale and episodes: Large-scale dataset with 250+ episodes.
Participants/teams:
Human-human collaboration: teams of 3 humans.
Human-robot collaboration: teams of 3 humans + 1 robot.
Demographics and exact number of unique participants not publicly detailed on the project page (typical for such controlled lab studies; full specs are in the paper).
Tasks/scenarios: Simulated assembly tasks performed in teams. Includes controlled variations with obstacles to study their impact on human motion and team dynamics. Tasks are designed to elicit natural close-proximity interactions relevant to real-world HRC (e.g., factories, shared workspaces).
Modalities captured (all synchronized for multi-agent, multimodal motion analysis):
3-D skeleton data (for all agents/participants; extracted or captured directly for pose tracking).
RGB+D data from two viewpoints (exocentric/third-person views with depth).
Ego-view (egocentric/first-person video from two participants).
Eye-tracking and gaze data (of two participants; provides intent and attention cues).
Robot joint data (positions/trajectories in HR scenarios).
Data nature: Multimodal, time-synchronized streams focused on motion and visual context. Includes raw/processed sensor data suitable for motion prediction (skeletons, RGB+D frames, gaze). Annotations implicitly include pose/joint trajectories and team interaction labels derived from the tasks (no fine-grained semantic action labels or external ground-truth questionnaires are detailed publicly). Data emphasize close-proximity team behaviors, making it ecologically valid for dynamic HRC.
Availability: Project website: https://mohammadyasar.github.io/Interact/ (includes paper BibTeX, overview, and links to related works). No public download links, Zenodo/GitHub repository, or direct access instructions are listed as of April 2026. The dataset is not openly hosted (no license or size details publicly available). It is likely available upon reasonable request from the authors (University of Virginia lab; common practice for recent HRI datasets to control access and ensure proper citation/use). Contact corresponding author Mohammad Samin Yasar or Tariq Iqbal for access inquiries. No processing/visualization code is publicly released on the project page.
Fields of use: Multi-human/multi-agent motion prediction, close-proximity HRC safety and coordination, robot perception/planning in teams, integration of multimodal cues (skeleton + vision + gaze) for anticipatory robotics. Explicitly positioned for real-world applications in collaborative manufacturing, healthcare, or shared workspaces where robots must operate safely near multiple humans. It fills a gap in benchmarks that include both HH and HR team data with robot joint information.


Reference 3: MultiPhysio-HRC: A Multimodal Physiological Signals Dataset for Industrial Human–Robot Collaboration
Authors: Andrea Bussolan, Stefano Baraldo, Oliver Avram, Pablo Urcola, Luis Montesano, Luca Maria Gambardella, Anna Valente
Date of release: 5 December 2025 (published in Robotics MDPI; arXiv preprint October 2025).
Scientific details and type of work: This is a public multimodal physiological dataset created specifically for industrial HRC (Industry 5.0). It captures psycho-physical states (stress, cognitive load) during real-world collaborative tasks to enable adaptive, human-aware robotics. Baseline models (RandomForest, AdaBoost, XGBoost) for stress/cognitive-load classification/regression are provided.
Dataset details (modalities, nature, availability, fields of use):
Participants: 55 (Day 1), 42 returned (Day 2); mean age ~28 years.
Modalities: EEG (12 channels, 256 Hz, Bitbrain Diadem), ECG, EDA, respiration (RESP), EMG (Versatile Bio sensors), voice recordings (48 kHz), facial action units (Py-Feat, 2 fps), video (webcam 30 Hz).
Tasks/scenarios: Cognitive load induction (Stroop, N-back, arithmetic, Tower of Hanoi), breathing exercises, immersive VR stress games, and industrial battery-disassembly (manual vs. Fanuc CRX-20 cobot-assisted).
Ground truth: Validated questionnaires (STAI-Y1 stress, NASA-TLX workload, SAM valence/arousal/dominance, NARS robot attitudes).
Data nature: Synchronized, real-world industrial-like; processed features (250+ per signal: time/frequency/complexity via Neurokit2; EEG PSD bands; voice MFCCs, etc.). LOSO validation baselines reported.
Availability: Public under CC BY license at https://automation-robotics-machines.github.io/MultiPhysio-HRC.github.io/ (full toolkit and data).
Fields of use: Affective computing, mental-state estimation, human-aware robotics, workplace well-being, adaptive HRC in Industry 5.0.


Reference 4: HARMONIC – A Multimodal Dataset of Assistive Human–Robot Collaboration
Authors: Benjamin A. Newman, Reuben M. Aronson, Siddhartha S. Srinivasa, Kris Kitani, Henny Admoni
Date of release: arXiv July 2018 (v2 2020); journal version December 2021 (The International Journal of Robotics Research).
Scientific details and type of work: This is a large public multimodal dataset collected in a shared-autonomy assistive-eating task (fork positioning over marshmallows). It provides synchronized human, robot, and environmental views to study intention prediction, mental-state modeling, and shared autonomy. The shared-autonomy controller (POMDP with MaxEntIOC goal inference and adjustable assistance parameter γ) is also released.
Dataset details (modalities, nature, availability, fields of use):
Participants: 24 novices (13 female, ages 18–45); 480 trials total (~5 h 19 min instrumented control).
Modalities (all timestamp-synchronized in nanoseconds):
Eye videos (both eyes, 120 Hz) + projected gaze on egocentric video.
Egocentric RGB video (head-mounted Pupil Labs, 30 Hz).
Forearm EMG + IMU (Myo sensor, ~20 % coverage).
Third-person stereo video (ZED cameras, 30 Hz) + OpenPose body/hand/facial keypoints.
Joystick commands (2-axis).
Robot joint positions and Cartesian states (Kinova Mico 6-DoF arm).
Assistance outputs (goal probabilities, applied twists).
Environmental: 3D marshmallow (morsel) positions.
Data nature: Real shared-autonomy interactions across 4 assistance levels (γ = 0/0.33/0.67/1); raw + processed files (MP4 videos, CSV/YAML, .npy); annotations include morsel labels, assistance info, gaze fixations, keypoints.
Availability: Public download (Google Drive links via https://harplab.github.io/harmonic/): full (~68 GB), minimal, sample, text-only versions. Processing/visualization code on GitHub. Semantic versioning and changelog maintained.
Fields of use: Intention prediction, human mental-state modeling, shared autonomy, multimodal HRC, assistive robotics, error detection via normative gaze/behavior models.

Below is a PhD-level, structured document entry for each of your three references. Every entry includes: release date, scientific details (data / framework), follow-up works in human–robot collaboration (HRC), and, where applicable, full dataset details.

## Reference 5. “Learning to Share Autonomy from Repeated Human-Robot Interaction” (SARI, Utah / VT-Collab)
Reference PDF
Date of release
May 18, 2022:
Scientific problem and contribution
Problem: Assistive robot arms need to learn shared autonomy without a predefined list of tasks or goals. Most existing shared-autonomy methods assume the robot knows a discrete set of possible goals; they fail when the human wants to perform a new or unexpected task (e.g., opening a fridge the robot has never seen).npe.law.stanford+1
Core idea: Humans repeatedly perform the same important tasks (e.g., opening the fridge, making coffee). The robot should learn assistance from scratch by exploiting these repeated interactions so that:
it recognizes the current task,
imitates similar past demonstrations,
and returns control when it is unsure.arxiv+1
SARI defines a learning-based shared-autonomy framework, not a fixed dataset.arxiv+1
Experiments are run in simulation and on a real robot arm (e.g., a wheelchair-mounted assistive arm); however, the authors focus on the algorithm and do not publish a centralized, named dataset (like “SARI dataset”) for external reuse.arxiv+1
Data characteristics implicitly used
If you build or want to reuse SARI-style data, typical traits are:
Modality:
Robot state: joint positions or end-effector pose (low-level kinematics).arxiv+1
Human input: joystick commands or equivalent teleoperation signals (discrete or continuous).npe.law.stanford
Data availability:
The core SARI code and, in some variants, teleoperation logs are available in the VT-Collab GitHub repository for “Learning to share autonomy across repeated interaction” / “SARI: Sharing autonomy across repeated interactions”; you can base your own data-collection pipeline on that code.github
Field of use:
Assistive robotics (wheelchair-mounted arms, reaching, manipulation).arxiv+1
Any long-term human–robot collaboration scenario where tasks are repeated and the robot must learn to assist over time (e.g., rehabilitation, domestic assistance).arxiv+1
Follow-up works in human–robot collaboration (HRC)
“SARI: Shared Autonomy across Repeated Interaction” (ACM, 2024-ish / HRI-adjacent journal paper)
Scientific details:
This is an extended / journal-style version of the Utah/Virginia Tech SARI work, with more emphasis on human–robot collaboration, user studies, and practical
It likely includes:
Additional user studies with disabled or non-expert users,
Broader HRC tasks (e.g., domestic chores, continuous manipulation skills),
Refined uncertainty quantification and arbitration
How it builds on the Utah PDF reference:
Uses the same SARI formulation but adds empirical HRC case studies, robustness analysis, and discussion of long-term learnability in real assistance settings
“SARI: Shared Autonomy across Repeated Interaction” (arXiv, 2022-05-18)
Date: May 18, 2022
Scientific details:
This is the canonical extended SARI paper: same core framework, but with:
Stability analysis (Lyapunov-style bound on error),
Capacity experiments (how many tasks SARI can learn),
Three user studies (including a disabled user pilot).arxiv+1
Relation to your Utah PDF:
The Utah PDF is a pedagogical version of the same SARI approach, while the arXiv 2022 paper is the primary research paper you should cite as the main SARI work.arxiv+2

## Reference 6. “SARI: Shared Autonomy across Repeated Interaction”

Date of release
Appears as part of a 2024–2026 ACM proceedings series (HRI-related or CSCW-related volume).
Exact conference year: the DOI entry is tied to a 2025 / 2026 publication in an ACM HRI-style or CSCW-style venue.acm+1
Scientific details of the work
Continuation of SARI: This ACM paper is a consolidated, extended edition of the SARI framework introduced in the Utah PDF and arXiv manuscripts
Key additions with respect to the Utah PDF:
More detailed HRC user studies:
Controlled experiments with human participants teleoperating a robot arm to reach discrete goals (cups, objects) and perform continuous skills (opening doors, drawers).
Comparison against:
classical shared-autonomy methods that assume a fixed goal set,
imitation-learning baselines that do not handle uncertainty
Quantitative metrics:
Objective performance (task completion time, trajectory error).
Subjective metrics (user workload, perceived trust, preference for SARI vs. baselines).
Deployment insights:
Lessons on how SARI behaves in real-world assistive scenarios (e.g., disabled-user pilot, noisy teleoperation).
Discussion of practical hyperparameters

The experiments naturally generate interaction datasets (teleoperation logs, human–robot trajectories) that could be reused if the authors release them.
Dataset characteristics 
Any dataset built around this ACM SARI paper would typically have:
Modalities:
Robot state: joint angles, end-effector position, velocity.
Human input: joystick or equivalent teleoperation signals.
Optional: timestamps, task indices (if the authors added explicit labels for reporting).
Data availability:
As of now, there is no explicit “SARI dataset” publication linked to this DOI; data sharing status depends on the specific ACM paper and its supplementary materials.
If not directly shared, you can treat the VT-Collab GitHub code as a reference for reproducing SARI-style data.github
Fields of use:
Assistive robotics, rehabilitation robotics, human–robot collaboration (long-term teleoperation, learning-based assistance).arxiv+2

## Reference 7. “ROS 4 Healthcare: Physiological Human Sensing for Social, Assistive, Rehabilitation, and Medical Robotics”
Date of release
Published in Frontiers in Robotics and AI, 2026-04 (article received and accepted in 2026).frontiersin
Scientific problem and contribution
Problem: There is a lack of standardization for integrating physiological, biological, and physical human-sensing signals into robotic systems (e.g., heart rate, ECG, EMG, GSR, motion).
Contribution: The authors introduce ROS 4 Healthcare (ROS4HC), an open-source ROS 2 framework that:
Defines unified ROS 2 message types for biosignals,
Provides sensor drivers for common biomedical devices,
Supplies processing libraries (filtering, feature extraction),
Offers visualization tools for real-time monitoring.
Framework / methodology
Architecture:
Unified message types: ROS4HC standardizes how biosignals are represented across ROS 2 systems (e.g., sensor_msgs/Biosignal, std_msgs/HR, etc.), enabling plug-and-play integration.
Sensor drivers: ROS4HC wraps or provides ROS 2 nodes for off-the-shelf wearables and medical devices (e.g., smartwatches, ECG bands, EMG, GSR, motion capture).
Processing libraries:
Filtering (low-pass, high-pass, notch),
Artifact removal,
Simple physiological metrics (HR, HRV, arousal indicators).
Visualization:
Supports RViz, web-based dashboards, and logging tools for real-time physiological monitoring.
Empirical case studies:
Heart-rate-adaptive wheelchair velocity:
Wheelchair speed is modulated based on the user’s heart rate, reducing perceived stress while maintaining mobility.
Autonomous robotic treadmill:
The treadmill speed and incline adapt according to the user’s cardiovascular load during rehabilitation or training.
Nocturnal monitoring system:
Continuous overnight monitoring of physiological signals (e.g., ECG, SpO₂, movement) for sleep-related diagnosis.
Is this work a dataset or a framework?
Framework, not a dataset: ROS4HC is a middleware and software framework, not a fixed dataset.
However, any experiment using ROS4HC naturally produces physiological–robot interaction datasets.
Dataset characteristics (if used in ROS4HC-based experiments)
Modalities:
Physiological signals:
HR, ECG, EEG, EMG, GSR (skin conductance), respiration, SpO₂, etc., depending on the sensors connected.
Physical / robot signals:
Robot state (joint positions, velocities, end-effector pose),
User motion (IMU data, pose from cameras or MoCap).
