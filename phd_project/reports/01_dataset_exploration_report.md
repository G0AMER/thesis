# Dataset Exploration Report — HARMONIC & DASIG

**Project**: Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale  
**Author**: Ameur Gargouri  
**Date**: February 2026  
**Notebook**: `notebooks/01_dataset_exploration.ipynb`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [HARMONIC Dataset](#2-harmonic-dataset)
   - 2.1 [Overview & Acquisition](#21-overview--acquisition)
   - 2.2 [File Structure](#22-file-structure)
   - 2.3 [Modalities & Signal Characteristics](#23-modalities--signal-characteristics)
   - 2.4 [Cross-Participant Statistics](#24-cross-participant-statistics)
   - 2.5 [Data Quality Issues](#25-data-quality-issues)
3. [DASIG Dataset](#3-dasig-dataset)
   - 3.1 [Overview & Acquisition](#31-overview--acquisition)
   - 3.2 [File Structure](#32-file-structure)
   - 3.3 [Subject Demographics](#33-subject-demographics)
   - 3.4 [MIMU Signal Characteristics](#34-mimu-signal-characteristics)
   - 3.5 [Abrupt Motion Analysis](#35-abrupt-motion-analysis)
   - 3.6 [PCA & Clustering](#36-pca--clustering)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Preprocessing Requirements](#5-preprocessing-requirements)
6. [Mapping to Research Axes](#6-mapping-to-research-axes)
7. [Conclusions & Next Steps](#7-conclusions--next-steps)

---

## 1. Executive Summary

This report documents the findings from the exploratory analysis of two human-robot interaction datasets selected for the PhD thesis. The **HARMONIC** dataset provides rich multimodal recordings (EMG, IMU, gaze, joystick commands, robot joint positions, body/hand/face pose) from 24 participants performing assistive eating tasks with a Kinova Mico robot arm. The **DASIG** dataset provides high-density MIMU (Magneto-Inertial Measurement Unit) data from 60 participants performing pick-and-place tasks under both standard and alarm-triggered (abrupt) conditions.

**Key findings**:
- HARMONIC contains **447 run trials** across 24 participants with up to **22 synchronized modalities** per trial, making it the primary dataset for imitation learning, IRL, and behavior generation experiments.
- EMG data availability is limited: only **96 out of 447 trials** (21.5%) contain non-empty EMG recordings.
- DASIG provides **180 MIMU recordings** (60 subjects × 3 conditions) at ~200 Hz sampling rate with 66 channels across 5 body segments, ideal for style modeling and abrupt motion detection.
- PCA on DASIG MIMU features explains **71.8% of variance** with just 2 components, and K-means clustering (K=3) partially separates the experimental conditions (FR_L, FR_R, LA_L).
- Both datasets require substantial preprocessing before use in learning algorithms.

---

## 2. HARMONIC Dataset

### 2.1 Overview & Acquisition

| Property | Value |
|----------|-------|
| **Full name** | HARMONIC (Human And Robot Multimodal Observations of Natural Interactive Collaboration) |
| **Version** | 0.5.0 (text-only archive `harmonic_1.0.0_text.tar.gz`) |
| **Download size** | ~4.25 GB (text modalities only; full dataset with video is ~68 GB) |
| **Source** | Google Drive (file ID: `1G4_fGdH0fB_8JCHj4vCEfcaej-oXs7M4`) |
| **Participants** | 24 (IDs: p100 – p123) |
| **Task** | Assistive eating with a Kinova Mico 6-DOF robot arm |
| **Total trials** | 532 (447 run + 67 check + calibration sessions) |

### 2.2 File Structure

```
harmonic_0.5.0/
├── p100/
│   ├── run/
│   │   ├── 000/text_data/   (22 CSV files)
│   │   ├── 001/text_data/
│   │   └── ...
│   ├── check/
│   │   └── 000/text_data/   (6 CSV files — gaze/calibration only)
│   └── calib/               (some participants only)
├── p101/
│   └── ...
└── p123/
```

**Trial types**:
- **Run trials** (447 total): Full multimodal recordings with 22 CSV files per trial. These are the main experimental recordings.
- **Check trials** (67 total): Calibration/verification sessions containing only 6 gaze-related CSV files.
- **Calib folders**: Present for some participants (p106, p108, p110, p112, p113, p114, p115, p116, p117, p118, p119, p120, p121, p123).

**Trials per participant** (from the 447 run trials):

| Participant | Trials | Participant | Trials | Participant | Trials |
|:-----------:|:------:|:-----------:|:------:|:-----------:|:------:|
| p100 | 19 | p108 | 14 | p116 | 19 |
| p101 | 20 | p109 | 20 | p117 | 16 |
| p102 | 20 | p110 | 20 | p118 | 20 |
| p103 | 20 | p111 | 20 | p119 | 20 |
| p104 | 14 | p112 | 20 | p120 | 20 |
| p105 | 10 | p113 | 19 | p121 | 20 |
| p106 | 19 | p114 | 20 | p122 | 20 |
| p107 | 17 | p115 | 20 | p123 | 20 |

Most participants have 19–20 trials. Notable exceptions: **p105 (10 trials)**, **p104 (14 trials)**, **p108 (14 trials)**.

### 2.3 Modalities & Signal Characteristics

Each **run trial** contains the following 22 CSV files (example: p100/run/000):

| File | Rows | Cols | Description | Relevance |
|------|:----:|:----:|-------------|-----------|
| `ada_joy.csv` | 2,002 | 8 | Joystick commands from operator | IRL (Axis A2) — operator input |
| `assistance_info.csv` | 2,589 | 13 | Assistance mode & control info | Task segmentation |
| `gaze_positions.csv` | 4,302 | 22 | Pupil Labs gaze tracking | Intent prediction |
| `input_info.csv` | 2,589 | 8 | Input device metadata | Control mode analysis |
| `joint_positions.csv` | 6,202 | 27 | Robot joint positions (6-DOF Kinova Mico) | Behavior generation (Axis A3) |
| `myo_emg.csv` | 0–3,100 | 12 | Myo armband EMG (8 channels) | Style modeling (Axis A1) |
| `myo_imu.csv` | 0–3,103 | 40 | Myo armband IMU (accel, gyro, orientation) | Motion analysis |
| `myo_ori.csv` | 0–var | 6 | Myo orientation quaternions | Arm posture |
| `pose_left_face_2d.csv` | 2,205 | 211 | Left camera face landmarks | Facial expression |
| `pose_left_hand_left_2d.csv` | 2,205 | 64 | Left camera left-hand keypoints | Gesture analysis |
| `pose_left_hand_right_2d.csv` | 2,205 | 64 | Left camera right-hand keypoints | Gesture analysis |
| `pose_left_pose_2d.csv` | 2,205 | 55 | Left camera body pose | Body posture |
| `pose_right_face_2d.csv` | 2,205 | 211 | Right camera face landmarks | Facial expression |
| `pose_right_hand_left_2d.csv` | 2,205 | 64 | Right camera left-hand keypoints | Gesture analysis |
| `pose_right_hand_right_2d.csv` | 2,205 | 64 | Right camera right-hand keypoints | Gesture analysis |
| `pose_right_pose_2d.csv` | 2,205 | 55 | Right camera body pose | Body posture |
| `pupil_cal_eye0.csv` | 7,827 | 34 | Calibrated pupil data (eye 0) | Gaze calibration |
| `pupil_cal_eye1.csv` | 7,575 | 34 | Calibrated pupil data (eye 1) | Gaze calibration |
| `pupil_eye0.csv` | 8,555 | 34 | Raw pupil data (eye 0) | Pupillometry |
| `pupil_eye1.csv` | 8,595 | 34 | Raw pupil data (eye 1) | Pupillometry |
| `robot_position.csv` | 6,202 | 30 | Robot link positions (Cartesian x,y,z) | End-effector trajectory |
| `world_cal_positions.csv` | 616 | 5 | World calibration reference | Coordinate mapping |

#### EMG (Myo Armband)
- **8 channels** (emg0–emg7), raw sensor values (not calibrated to µV)
- Typical values: 20–600 (integer)
- Clear muscle activation patterns visible in the signals
- Different activation patterns across channels suggest usable for style modeling
- **Critical issue**: EMG data is missing (0 rows) in many trials — only **96/447 trials (21.5%)** have non-empty EMG recordings

#### IMU (Myo Armband)
- **37 channels** (after excluding timestamps): orientation (quaternion + covariance), angular velocity (3-axis + covariance), linear acceleration (3-axis + covariance)
- Linear acceleration range: approximately −1.0 to −0.15 (normalized values)
- Same availability issue as EMG — only available in trials where the Myo armband was active

#### Gaze (Pupil Labs)
- **22 columns** including `norm_pos_x`, `norm_pos_y` (normalized 0–1), `confidence`, 3D gaze point, eye centers, gaze normals
- Confidence: all 4,302 samples passed the >0.5 filter in the example trial (100% retention)
- Gaze heatmap shows clear fixation patterns, concentrated in the upper-center region of the visual field
- Available in **all 447 run trials**

#### Joystick (Ada Joy)
- **5 data columns** after removing timestamps: operator commands to the robot
- Available in **all 447 run trials**
- Directly maps to operator intent — key for IRL reward inference

#### Robot Position
- **30 columns**: Cartesian (x, y, z) coordinates for 10 robot links:
  - `mico_link_base`, `mico_link_1` through `mico_link_5`, `mico_link_hand`, `mico_end_effector`, `mico_fork_tip`
- End-effector trajectory range: X ∈ [0.04, 0.26], Y ∈ [−0.42, −0.28], Z ∈ [0.11, 0.36] (meters)
- The 3D trajectory visualization shows clear task-related motion patterns (reaching, grasping, feeding sequence)
- Available in **all 447 run trials**

#### Joint Positions
- **27 columns**: 6-DOF joint angles plus timestamps
- Available in **all 447 run trials**
- Together with robot_position, provides complete kinematic state

### 2.4 Cross-Participant Statistics

Statistics were computed across all 447 run trials:

**Modality availability across trials**:
| Modality | Available trials | Percentage |
|----------|:----------------:|:----------:|
| Joystick (ada_joy) | 447/447 | 100% |
| Joint positions | 447/447 | 100% |
| Gaze positions | 447/447 | 100% |
| EMG (myo_emg) | 96/447 | 21.5% |

**EMG inter-subject variability** (based on available trials):
- Mean EMG activation varies from ~70 (p105) to ~160 (p121) across participants
- EMG variability (std) ranges from ~35 to ~110
- Clear inter-subject differences — promising for style modeling
- Participants p101, p107, p121 show highest activation and variability

### 2.5 Data Quality Issues

1. **Missing EMG/IMU data**: The majority of trials (78.5%) have empty Myo armband recordings. This limits EMG-based style modeling to ~96 trials across a subset of participants.
2. **Variable sampling rates**: Different modalities have different recording rates (gaze ~60Hz, joints ~100Hz, EMG ~50Hz). Temporal alignment/resampling will be required.
3. **Some empty DataFrames**: Certain CSV files in specific trials contain headers but 0 data rows (e.g., `myo_emg.csv` at 0.1 KB).
4. **No explicit episode boundaries**: Continuous recordings without clear start/end markers for individual task episodes (e.g., single reach-grasp-feed sequences).
5. **Pose data dimensionality**: Face landmark files (211 columns each) are very high-dimensional and will need dimensionality reduction or feature selection.

---

## 3. DASIG Dataset

### 3.1 Overview & Acquisition

| Property | Value |
|----------|-------|
| **Full name** | DASIG (Dataset for Abrupt and Standard Industrial Gestures) |
| **Download size** | ~3.5 GB |
| **Source** | Zenodo (record ID: 17660014) |
| **Participants** | 60 (sub001 – sub060) |
| **Task** | Industrial pick-and-place with alarm-triggered interruptions |
| **Conditions** | 3 (FR_L, FR_R, LA_L) |
| **Total recordings** | 180 MIMU + 180 Arduino (60 subjects × 3 conditions) |

### 3.2 File Structure

```
DASIG/
├── subjects_info.csv          (60 rows × 10 columns)
├── sub001/
│   ├── sub001_FR_L_MIMU.csv   (~12 MB, 19,385 rows × 66 cols)
│   ├── sub001_FR_L_Arduino.csv (35 rows × 2 cols)
│   ├── sub001_FR_R_MIMU.csv
│   ├── sub001_FR_R_Arduino.csv
│   ├── sub001_LA_L_MIMU.csv
│   └── sub001_LA_L_Arduino.csv
├── sub002/
│   └── ...
└── sub060/
```

**CSV format**: European-style with `;` separator and `,` as the decimal point. The `subjects_info.csv` file additionally uses BOM encoding (`utf-8-sig`).

**MIMU file structure**: 3 header rows (body segment names, sensor types, axis labels) followed by numerical data. Each file has 66 columns.

**Conditions**:
- **FR_L**: Free movement, left hand
- **FR_R**: Free movement, right hand
- **LA_L**: Left-arm movement with alarm interruptions

### 3.3 Subject Demographics

The dataset includes detailed anthropometric data for all 60 participants:

| Measurement | Mean | Std | Min | Max |
|-------------|:----:|:---:|:---:|:---:|
| Height (m) | ~1.72 | ~0.08 | 1.51 | ~1.88 |
| Weight (kg) | ~66 | ~13 | ~45 | ~100 |
| Right upper arm length (m) | ~0.34 | ~0.03 | ~0.27 | ~0.40 |
| Left upper arm length (m) | ~0.34 | ~0.03 | ~0.28 | ~0.41 |
| Right forearm length (m) | ~0.28 | ~0.03 | ~0.23 | ~0.37 |
| Left forearm length (m) | ~0.29 | ~0.03 | ~0.24 | ~0.36 |

The demographics show good variability in body dimensions, which is important for evaluating whether style models can generalize across different body types.

### 3.4 MIMU Signal Characteristics

**Sensor configuration**: 5 body segments × 4 sensor types × 3–4 axes = 65 channels + 1 time column = 66 total

| Body Segment | Abbreviation | Channels |
|-------------|:------------:|:--------:|
| Right Forearm | RF | 13 (Acc XYZ, Gyro XYZ, Mag XYZ, Ori SXYZ) |
| Left Forearm | LF | 13 |
| Sternum | ST | 13 |
| Right Upper Arm | RUA | 13 |
| Left Upper Arm | LUA | 13 |

**Signal properties** (from sub001_FR_L):
- **Recording duration**: ~97 seconds per trial
- **Sampling rate**: ~200 Hz (19,385 samples / 96.9s)
- **Accelerometer range**: approximately −35 to +20 m/s²
- **Gyroscope range**: approximately −5 to +5 rad/s
- **Magnetometer range**: approximately −25 to +50 G (Gauss)

**Observations from signal visualization**:
- Accelerometer signals are relatively stable with occasional sharp spikes corresponding to pick-and-place movements
- A prominent transient spike appears at ~13s in sub001_FR_L, likely corresponding to a large abrupt movement or alarm trigger
- Gyroscope signals show clear bursts during active motion phases
- Magnetometer readings are stable with gradual drift, useful for orientation estimation

### 3.5 Abrupt Motion Analysis

Using Right Forearm accelerometer data and jerk (time derivative of acceleration magnitude) as the primary metric for detecting abrupt movements:

**Detection method**: Threshold-based on jerk magnitude (μ + 3σ)

| Metric | Value (sub001_FR_L) |
|--------|:-------------------:|
| Jerk threshold (μ + 3σ) | 131.34 m/s³ |
| Samples flagged as abrupt | 31 / 19,384 (0.16%) |
| Dominant abrupt event | ~13s (coincides with large acceleration spike) |

**Observations**:
- Abrupt movements are rare events (< 0.2% of samples), consistent with the experimental protocol where alarms are periodic interruptions
- The alarm timestamps (from Arduino files) align visually with periods of elevated motion
- The jerk-based detection successfully isolates the large transient events
- Most of the recording shows steady pick-and-place patterns with low jerk values

**Arduino alarm data** (sub001_FR_L):
- 35 alarm events over the ~97-second recording
- Contains timestamps and stimuli codes
- Provides ground truth for segmenting standard vs. alarm-triggered movements

### 3.6 PCA & Clustering

#### Feature Extraction
Summary features were computed for 179/180 MIMU recordings (1 file was skipped due to parsing issues). Features include:
- Per-channel statistics (mean, std, max, min, range) for acceleration and gyroscope
- Acceleration magnitude statistics (mean, std, max)
- Jerk statistics (mean |jerk|, std, max |jerk|)
- Total: ~27 features per recording

#### PCA Results
- **PC1** explains **64.9%** of variance
- **PC2** explains **6.9%** of variance
- **Total (2 components)**: **71.8%** of variance

**PCA by Subject**: No clearly distinct per-subject clusters, but subjects are spread across the PC space. Some subjects form tight groups while others are more dispersed — suggests inter-subject variability exists but is interleaved with condition effects.

**PCA by Condition**: Clear separation emerges:
- **FR_L** (free movement, left): Concentrated in the left side of PC1 (negative values)
- **FR_R** (free movement, right): Spread toward the right side of PC1 (positive values)
- **LA_L** (alarm, left): Positioned between FR_L and FR_R on PC1, with higher PC2 values

The dominant PC1 axis appears to capture **handedness** (left vs. right arm usage), while PC2 may capture aspects related to alarm interruptions.

#### K-Means Clustering (K=3)

Cluster × Condition cross-tabulation:

| Cluster | FR_L | FR_R | LA_L |
|:-------:|:----:|:----:|:----:|
| 0 | 60 | 0 | 55 |
| 1 | 0 | 19 | 0 |
| 2 | 0 | 41 | 4 |

**Interpretation**:
- **Cluster 0** captures left-arm movements (all 60 FR_L + 55 of 59 LA_L = 115/119 left-arm trials)
- **Clusters 1 and 2** split right-arm movements (FR_R) into two subgroups (19 vs. 41), suggesting two distinct motion styles within right-arm free movement
- The LA_L condition largely co-clusters with FR_L, meaning **alarm interruptions alone do not create a globally distinct motion signature** in aggregate statistics. Finer temporal analysis (windowed features around alarm events) will be needed.

---

## 4. Comparative Analysis

| Property | HARMONIC | DASIG |
|----------|----------|-------|
| **Participants** | 24 | 60 |
| **Trials per participant** | ~10–20 | 3 |
| **Total recordings** | 447 (run) | 180 |
| **Sampling rate** | Variable (50–100 Hz) | ~200 Hz |
| **EMG** | ✅ 8ch Myo (21.5% of trials) | ❌ |
| **IMU/MIMU** | ✅ Myo IMU (limited availability) | ✅ 5-segment × 13-channel MIMU |
| **Gaze tracking** | ✅ Pupil Labs (100% of trials) | ❌ |
| **Joystick / operator input** | ✅ Ada joystick (100%) | ❌ |
| **Robot kinematics** | ✅ 6-DOF Kinova Mico (100%) | ❌ |
| **Body/hand/face pose** | ✅ OpenPose (stereo cameras) | ❌ |
| **Alarm / abrupt events** | ❌ | ✅ Arduino timestamps |
| **Subject anthropometry** | ❌ | ✅ Height, weight, arm lengths |
| **File format** | Standard CSV (`,` separator) | European CSV (`;` separator, `,` decimal) |

**Complementarity**: The datasets are highly complementary. HARMONIC provides the multimodal richness needed for imitation learning and behavior generation but lacks abrupt motion data. DASIG provides the large subject pool and explicit standard/abrupt paradigm needed for safety modeling but has no robot data.

---

## 5. Preprocessing Requirements

### 5.1 HARMONIC

| Step | Priority | Description |
|------|:--------:|-------------|
| Timestamp alignment | **High** | Resample all modalities to a common rate (e.g., 50 Hz) using timestamps. Different sensors record at different rates. |
| EMG preprocessing | **High** | Rectification → band-pass filter (20–450 Hz) → RMS envelope → normalization (z-score per subject, since no MVC available). |
| Gaze filtering | **Medium** | Remove low-confidence samples (though exploration showed 100% pass rate in tested trial), interpolate gaps, smooth with moving average. |
| Episode segmentation | **High** | Split continuous recordings into individual task episodes (reach → grasp → transport → feed). Use `assistance_info.csv` mode changes and joystick activity as segmentation cues. |
| Label extraction | **High** | For IRL: extract implicit reward signals from joystick corrections (sudden reversals = negative reward). For IL: define demonstrations from smooth task completions. |
| Participant splits | **Medium** | Define train/validation/test splits respecting subject boundaries for generalization evaluation. |
| Pose dimensionality reduction | **Low** | Face landmarks (211 cols) and hand keypoints (64 cols) need PCA or selection of key joints. |

### 5.2 DASIG

| Step | Priority | Description |
|------|:--------:|-------------|
| MIMU calibration | **High** | Remove sensor offset/bias (visible in magnetometer baseline). Apply low-pass Butterworth filter to accelerometer. |
| Gesture segmentation | **High** | Use Arduino alarm timestamps to define temporal windows: standard gesture (before alarm) vs. abrupt reaction (after alarm). |
| Feature extraction | **High** | Time-domain: mean, std, range, RMS, zero-crossing rate, jerk statistics. Frequency-domain: FFT dominant frequency, spectral energy. |
| Windowing | **Medium** | Sliding window (e.g., 200 ms with 50% overlap) for segment-level classification. |
| Subject-independent splits | **High** | Leave-N-subjects-out cross-validation for evaluating generalization (critical with 60 subjects). |
| Anthropometric normalization | **Low** | Consider normalizing acceleration by arm length to account for body size differences. |

---

## 6. Mapping to Research Axes

| Experiment | Primary Dataset | Key Modalities | Rationale |
|-----------|----------------|----------------|-----------|
| **Exp 1** — IL comparison | HARMONIC | Joystick, robot joints, gaze | Sequential sessions per subject enable incremental learning evaluation |
| **Exp 2** — IRL comparison | HARMONIC | Joystick commands, corrections, robot state | Commands serve as demonstrations, corrections inform reward inference |
| **Exp 3** — Style modeling | DASIG (primary) + HARMONIC | MIMU features (DASIG), EMG + gaze + pose (HARMONIC) | 60 subjects in DASIG provide high statistical power for inter-subject variability; HARMONIC adds richer modalities |
| **Exp 4** — Behavior generation | HARMONIC | Robot trajectories, operator style | End-effector 3D trajectories + joint positions provide complete robot state for trajectory generation |
| **Exp 5** — Full system | HARMONIC + DASIG | All modalities | HARMONIC for perception/generation pipeline, DASIG for safety layer validation |
| **Exp 6** — Longitudinal | Own data collection | — | — |

---

## 7. Conclusions & Next Steps

### Key Takeaways

1. **HARMONIC is the primary dataset** for the thesis, providing synchronized multimodal data (gaze, joystick, robot kinematics, body pose) for all 447 run trials. However, **EMG availability is limited to 21.5% of trials**, which constrains EMG-based style modeling.

2. **DASIG is the primary dataset for style modeling (Exp 3)** thanks to its 60 subjects and explicit standard/abrupt experimental conditions. The PCA analysis confirms meaningful inter-subject and inter-condition variability exists in the MIMU features. However, the aggregate K-means clustering shows that **alarm effects are subtle at the whole-recording level** — temporal windowing around alarm events will be necessary.

3. **Data quality is generally good** for both datasets, but each requires specific preprocessing:
   - HARMONIC: temporal alignment across modalities is the most critical step.
   - DASIG: proper CSV parsing (handled), calibration bias removal, and alarm-based temporal segmentation.

4. **The PCA condition separation in DASIG** (PC1 captures handedness at 64.9% variance) suggests that hand-specific models or explicit handedness features should be incorporated in the style modeling pipeline.

5. **The robot 3D trajectory** visualization confirms that HARMONIC captures meaningful task-level behavior patterns, suitable for trajectory-based behavior generation.

### Immediate Next Steps

1. **Implement HARMONIC preprocessing pipeline** (`src/perception/`): timestamp alignment, EMG filtering, episode segmentation.
2. **Implement DASIG preprocessing pipeline** (`src/style_modeling/`): MIMU calibration, alarm-based windowing, feature extraction.
3. **Begin Experiment 1** (IL comparison): Use preprocessed HARMONIC joystick + robot data.
4. **Begin Experiment 3** (Style modeling): Use DASIG windowed features for inter-subject clustering and style identification.
5. **Define formal train/val/test splits** for both datasets with subject-level stratification.

---

*Report generated from the analysis in `notebooks/01_dataset_exploration.ipynb`.*
