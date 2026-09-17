"""
DASIG Dataset Loader & Safety-State Label Generator
====================================================
Loads all MIMU and Arduino CSVs from the DASIG dataset,
aligns timestamps, and generates per-sample safety-state labels.

Safety states:
  0 = SAFE    (standard pick-and-place movement)
  1 = WARNING (transition — alarm fired but reaction not yet started)
  2 = DANGER  (abrupt movement in progress)

References:
  - Digo et al., "DASIG: A Dataset of Standard and Abrupt Industrial Gestures",
    Robotics 2025, 14, 176.
  - ISO/TS 15066:2016 — Collaborative robot safety.
"""

import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─── Constants ───────────────────────────────────────────────────────────────

SAMPLING_RATE_HZ = 200
TRIAL_DURATION_S = 90.0
NUM_SUBJECTS = 60
TRIALS = ["FR_R", "FR_L", "LA_L"]

# MIMU sensor layout: 5 sensors, each with 13 channels
SENSOR_NAMES = ["RFA", "LFA", "STR", "RUA", "LUA"]  # order in CSV columns
SENSOR_CHANNELS = ["acc_x", "acc_y", "acc_z",
                    "gyr_x", "gyr_y", "gyr_z",
                    "mag_x", "mag_y", "mag_z",
                    "quat_s", "quat_x", "quat_y", "quat_z"]
CHANNELS_PER_SENSOR = len(SENSOR_CHANNELS)  # 13
TOTAL_CHANNELS = len(SENSOR_NAMES) * CHANNELS_PER_SENSOR  # 65

# Arduino event codes → safety mapping
STANDARD_CODES = {2, 3, 4, 5}       # Green LEDs at stations SA-SD
VISUAL_ALARM_CODES = {6, 7, 8, 9}   # Red LEDs at stations SA-SD
ACOUSTIC_ALARM_CODE = 10             # Sound buzzer
ABRUPT_CODES = VISUAL_ALARM_CODES | {ACOUSTIC_ALARM_CODE}

# Safety-state labels
SAFE = 0
WARNING = 1
DANGER = 2

# Label generation parameters (in seconds)
DEFAULT_REACTION_TIME = 0.3   # Typical human reaction time to visual/acoustic
DEFAULT_ABRUPT_DURATION = 2.5 # Duration of abrupt gesture after reaction


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TrialData:
    """All data for a single trial (one subject, one condition)."""
    subject_id: str          # e.g. "sub001"
    trial_name: str          # e.g. "FR_R"
    time: np.ndarray         # (N,) timestamps in seconds
    imu_data: np.ndarray     # (N, 65) all MIMU channels
    events_time: np.ndarray  # (M,) Arduino event timestamps
    events_code: np.ndarray  # (M,) Arduino event codes
    labels: Optional[np.ndarray] = None  # (N,) safety-state labels

    @property
    def n_samples(self) -> int:
        return len(self.time)

    @property
    def duration_s(self) -> float:
        return self.time[-1] - self.time[0] if len(self.time) > 0 else 0.0

    @property
    def n_abrupt_events(self) -> int:
        return int(np.sum(np.isin(self.events_code, list(ABRUPT_CODES))))


@dataclass
class SubjectInfo:
    """Anthropometric data for a subject."""
    subject_id: str
    gender: str
    age_range: str
    height_m: float
    weight_kg: float
    dominant_arm: str
    right_upper_arm_m: float
    left_upper_arm_m: float
    right_forearm_m: float
    left_forearm_m: float


# ─── Column name builder ─────────────────────────────────────────────────────

def build_column_names() -> list[str]:
    """Build the 65 standardized column names for MIMU data."""
    cols = []
    for sensor in SENSOR_NAMES:
        for channel in SENSOR_CHANNELS:
            cols.append(f"{sensor}_{channel}")
    return cols


COLUMN_NAMES = build_column_names()


# ─── CSV Loaders ─────────────────────────────────────────────────────────────

def load_mimu_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a MIMU CSV file (European decimal format, semicolon-separated).

    The file has 3 header rows:
      Row 0: sensor group names (sparse)
      Row 1: signal type names (Accelerometer, Gyroscope, etc.)
      Row 2: axis labels (X, Y, Z, S)

    Returns:
        time: (N,) array of timestamps in seconds
        data: (N, 65) array of all sensor channels
    """
    # Skip the 3 header rows, use semicolon separator, European decimal
    df = pd.read_csv(
        filepath,
        sep=';',
        decimal=',',
        skiprows=3,
        header=None,
        dtype=np.float64,
    )

    # Column 0 is time, columns 1-65 are sensor data
    time = df.iloc[:, 0].values
    data = df.iloc[:, 1:TOTAL_CHANNELS + 1].values

    # Validate shape
    assert data.shape[1] == TOTAL_CHANNELS, (
        f"Expected {TOTAL_CHANNELS} channels, got {data.shape[1]} in {filepath}"
    )

    return time, data


def load_arduino_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an Arduino event CSV file.

    Returns:
        event_times: (M,) array of event timestamps in seconds
        event_codes: (M,) array of integer event codes
    """
    df = pd.read_csv(filepath, sep=';', decimal=',')
    if df.iloc[:, 0].dtype == object:
        event_times = df.iloc[:, 0].str.replace(',', '.').astype(np.float64).values
    else:
        event_times = df.iloc[:, 0].values.astype(np.float64)
    event_codes = df.iloc[:, 1].values.astype(np.int32)
    return event_times, event_codes


def load_subject_info(filepath: str) -> SubjectInfo:
    """Load a single subject's info CSV."""
    df = pd.read_csv(filepath, sep=';', decimal=',')
    row = df.iloc[0]
    return SubjectInfo(
        subject_id=os.path.basename(filepath).split('_')[0],
        gender=str(row.iloc[0]),
        age_range=str(row.iloc[1]),
        height_m=float(row.iloc[2]),
        weight_kg=float(row.iloc[3]),
        dominant_arm=str(row.iloc[4]),
        right_upper_arm_m=float(row.iloc[5]),
        left_upper_arm_m=float(row.iloc[6]),
        right_forearm_m=float(row.iloc[7]),
        left_forearm_m=float(row.iloc[8]),
    )


# ─── Label Generation ────────────────────────────────────────────────────────

def generate_safety_labels(
    time: np.ndarray,
    event_times: np.ndarray,
    event_codes: np.ndarray,
    imu_data: Optional[np.ndarray] = None,
    reaction_time_s: float = DEFAULT_REACTION_TIME,
    abrupt_duration_s: float = DEFAULT_ABRUPT_DURATION,
) -> np.ndarray:
    """
    Generate per-sample safety-state labels from Arduino events.
    Uses dynamic motion-onset labeling based on the peak sternum acceleration
    if imu_data is provided.
    """
    labels = np.zeros(len(time), dtype=np.int32)  # Default: SAFE

    for evt_time, evt_code in zip(event_times, event_codes):
        if evt_code not in ABRUPT_CODES:
            continue

        if imu_data is not None:
            # PROPOSAL 2: Jerk/Acceleration-based relabeling
            # Search for the peak sternum acceleration within 1.0s after the alarm
            search_start = evt_time
            search_end = evt_time + 1.0
            mask = (time >= search_start) & (time <= search_end)
            if np.any(mask):
                str_acc = imu_data[mask, 0:3] # STR is the first sensor, channels 0-2
                acc_mag = np.sqrt(np.sum(str_acc**2, axis=1))
                peak_idx_in_mask = np.argmax(acc_mag)
                time_in_mask = time[mask]
                true_danger_start = time_in_mask[peak_idx_in_mask]
            else:
                true_danger_start = evt_time + reaction_time_s
                
            warn_start = evt_time
            warn_end = true_danger_start
            danger_start = true_danger_start
            danger_end = true_danger_start + abrupt_duration_s
        else:
            warn_start = evt_time
            warn_end = evt_time + reaction_time_s
            danger_start = warn_end
            danger_end = warn_end + abrupt_duration_s

        warn_mask = (time >= warn_start) & (time < warn_end)
        danger_mask = (time >= danger_start) & (time < danger_end)

        labels[warn_mask] = WARNING
        labels[danger_mask] = DANGER

    return labels


# ─── Full Dataset Loader ─────────────────────────────────────────────────────

def load_trial(
    data_dir: str,
    subject_id: str,
    trial_name: str,
    generate_labels: bool = True,
    reaction_time_s: float = DEFAULT_REACTION_TIME,
    abrupt_duration_s: float = DEFAULT_ABRUPT_DURATION,
) -> TrialData:
    """
    Load a single trial's MIMU + Arduino data and optionally generate labels.

    Args:
        data_dir: Path to the DASIG root directory
        subject_id: e.g. "sub001"
        trial_name: one of "FR_R", "FR_L", "LA_L"
        generate_labels: Whether to compute safety-state labels
        reaction_time_s: Reaction time parameter for label generation
        abrupt_duration_s: Abrupt gesture duration for label generation

    Returns:
        TrialData with all fields populated
    """
    subject_dir = os.path.join(data_dir, subject_id)

    mimu_path = os.path.join(subject_dir, f"{subject_id}_{trial_name}_MIMU.csv")
    arduino_path = os.path.join(subject_dir, f"{subject_id}_{trial_name}_Arduino.csv")

    time, imu_data = load_mimu_csv(mimu_path)
    event_times, event_codes = load_arduino_csv(arduino_path)

    labels = None
    if generate_labels:
        labels = generate_safety_labels(
            time, event_times, event_codes,
            imu_data=imu_data,
            reaction_time_s=reaction_time_s,
            abrupt_duration_s=abrupt_duration_s,
        )

    return TrialData(
        subject_id=subject_id,
        trial_name=trial_name,
        time=time,
        imu_data=imu_data,
        events_time=event_times,
        events_code=event_codes,
        labels=labels,
    )


def load_all_trials(
    data_dir: str,
    subjects: Optional[list[str]] = None,
    trials: Optional[list[str]] = None,
    generate_labels: bool = True,
    reaction_time_s: float = DEFAULT_REACTION_TIME,
    abrupt_duration_s: float = DEFAULT_ABRUPT_DURATION,
    verbose: bool = True,
) -> list[TrialData]:
    """
    Load all (or selected) trials from the DASIG dataset.

    Args:
        data_dir: Path to the DASIG root directory
        subjects: List of subject IDs to load (default: all 60)
        trials: List of trial names to load (default: all 3)
        generate_labels: Whether to compute safety-state labels
        verbose: Whether to print progress

    Returns:
        List of TrialData objects
    """
    if subjects is None:
        subjects = [f"sub{i:03d}" for i in range(1, NUM_SUBJECTS + 1)]
    if trials is None:
        trials = TRIALS

    all_trials = []
    total = len(subjects) * len(trials)
    loaded = 0

    for subj in subjects:
        for trial in trials:
            try:
                td = load_trial(
                    data_dir, subj, trial,
                    generate_labels=generate_labels,
                    reaction_time_s=reaction_time_s,
                    abrupt_duration_s=abrupt_duration_s,
                )
                all_trials.append(td)
                loaded += 1
                if verbose and loaded % 30 == 0:
                    print(f"  Loaded {loaded}/{total} trials...")
            except Exception as e:
                print(f"  WARNING: Failed to load {subj}/{trial}: {e}")

    if verbose:
        print(f"  Done: {loaded}/{total} trials loaded successfully.")
        # Summary stats
        n_safe = sum(np.sum(t.labels == SAFE) for t in all_trials if t.labels is not None)
        n_warn = sum(np.sum(t.labels == WARNING) for t in all_trials if t.labels is not None)
        n_danger = sum(np.sum(t.labels == DANGER) for t in all_trials if t.labels is not None)
        total_samples = n_safe + n_warn + n_danger
        print(f"  Label distribution:")
        print(f"    SAFE:    {n_safe:>10,} samples ({100*n_safe/total_samples:.1f}%)")
        print(f"    WARNING: {n_warn:>10,} samples ({100*n_warn/total_samples:.1f}%)")
        print(f"    DANGER:  {n_danger:>10,} samples ({100*n_danger/total_samples:.1f}%)")

    return all_trials


def load_all_subject_info(data_dir: str) -> list[SubjectInfo]:
    """Load anthropometric info for all subjects."""
    infos = []
    for i in range(1, NUM_SUBJECTS + 1):
        subj_id = f"sub{i:03d}"
        info_path = os.path.join(data_dir, subj_id, f"{subj_id}_info.csv")
        if os.path.exists(info_path):
            infos.append(load_subject_info(info_path))
    return infos


# ─── Train/Test Splitting (by subject) ───────────────────────────────────────

def split_by_subject(
    trials: list[TrialData],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[TrialData], list[TrialData], list[TrialData]]:
    """
    Split trials into train/val/test sets BY SUBJECT to prevent data leakage.

    Args:
        trials: List of all TrialData
        train_ratio: Fraction of subjects for training
        val_ratio: Fraction of subjects for validation
        seed: Random seed for reproducibility

    Returns:
        (train_trials, val_trials, test_trials)
    """
    rng = np.random.RandomState(seed)

    # Get unique subjects
    subjects = sorted(set(t.subject_id for t in trials))
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Shuffle and split
    perm = rng.permutation(n)
    train_subjs = set(subjects[i] for i in perm[:n_train])
    val_subjs = set(subjects[i] for i in perm[n_train:n_train + n_val])
    test_subjs = set(subjects[i] for i in perm[n_train + n_val:])

    train_trials = [t for t in trials if t.subject_id in train_subjs]
    val_trials = [t for t in trials if t.subject_id in val_subjs]
    test_trials = [t for t in trials if t.subject_id in test_subjs]

    print(f"Split: {len(train_subjs)} train subjects ({len(train_trials)} trials), "
          f"{len(val_subjs)} val subjects ({len(val_trials)} trials), "
          f"{len(test_subjs)} test subjects ({len(test_trials)} trials)")

    return train_trials, val_trials, test_trials
