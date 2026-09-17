"""
Feature Engineering for Cobot Safety-State Detection
=====================================================
Sliding window segmentation + handcrafted feature extraction
from MIMU sensor data.

Features are designed to capture the kinematic signatures that
distinguish standard movements from abrupt/dangerous ones:
  - RMS values (validated by the DASIG paper as discriminative)
  - Peak magnitudes (abrupt = high peak)
  - Jerk (rate of acceleration change — key abruptness indicator)
  - Signal energy
  - Statistical moments (mean, std, skew, kurtosis)
  - Quaternion angular displacement
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .data_loader import (
    TrialData, CHANNELS_PER_SENSOR, SENSOR_NAMES, TOTAL_CHANNELS,
    SAFE, WARNING, DANGER, SAMPLING_RATE_HZ,
)


# ─── Sliding Window Segmentation ─────────────────────────────────────────────

def sliding_window_segment(
    trial: TrialData,
    window_size_s: float = 1.0,
    step_size_s: float = 0.5,
    label_strategy: str = "majority",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment a trial into overlapping windows with labels.

    Args:
        trial: TrialData object with labels
        window_size_s: Window duration in seconds
        step_size_s: Step (hop) duration in seconds
        label_strategy: How to assign a label to a window:
            - "majority": Most frequent label in the window
            - "any_danger": DANGER if any sample is DANGER, else WARNING if any, else SAFE
            - "center": Label of the center sample

    Returns:
        windows: (W, window_samples, 65) array of IMU data segments
        labels: (W,) array of integer labels
        timestamps: (W,) array of window start times
    """
    assert trial.labels is not None, "Trial must have labels for segmentation"

    window_samples = int(window_size_s * SAMPLING_RATE_HZ)
    step_samples = int(step_size_s * SAMPLING_RATE_HZ)

    n = trial.n_samples
    windows = []
    labels = []
    timestamps = []

    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = trial.imu_data[start:end]
        window_labels = trial.labels[start:end]

        # Assign window-level label
        if label_strategy == "majority":
            label = int(np.bincount(window_labels).argmax())
        elif label_strategy == "any_danger":
            if DANGER in window_labels:
                label = DANGER
            elif WARNING in window_labels:
                label = WARNING
            else:
                label = SAFE
        elif label_strategy == "center":
            label = int(window_labels[window_samples // 2])
        elif label_strategy == "all":
            label = window_labels
        else:
            raise ValueError(f"Unknown label strategy: {label_strategy}")

        windows.append(window_data)
        labels.append(label)
        timestamps.append(trial.time[start])

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(timestamps, dtype=np.float64),
    )


def segment_all_trials(
    trials: list[TrialData],
    window_size_s: float = 1.0,
    step_size_s: float = 0.5,
    label_strategy: str = "majority",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment all trials and concatenate into a single dataset.

    Returns:
        X: (total_windows, window_samples, 65) feature matrix
        y: (total_windows,) label vector
    """
    all_windows = []
    all_labels = []

    for trial in trials:
        w, l, _ = sliding_window_segment(
            trial, window_size_s, step_size_s, label_strategy
        )
        all_windows.append(w)
        all_labels.append(l)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    if verbose:
        print(f"Segmented {len(trials)} trials → {X.shape[0]} windows "
              f"(shape: {X.shape})")
        for cls, name in enumerate(["SAFE", "WARNING", "DANGER"]):
            count = np.sum(y == cls)
            print(f"  {name}: {count:>6} windows ({100*count/len(y):.1f}%)")

    return X, y


# ─── Feature Extraction ──────────────────────────────────────────────────────

def _vector_norm(data: np.ndarray) -> np.ndarray:
    """Compute L2 norm along the last axis (for 3-axis signals)."""
    return np.sqrt(np.sum(data ** 2, axis=-1))


def extract_sensor_features(
    window: np.ndarray,
    sensor_offset: int,
) -> dict[str, float]:
    """
    Extract features from a single sensor's data within a window.

    Args:
        window: (window_samples, 65) single window
        sensor_offset: Starting column index for this sensor

    Returns:
        Dictionary of feature_name → feature_value
    """
    # Extract raw signals for this sensor
    acc = window[:, sensor_offset:sensor_offset + 3]       # 3-axis acceleration
    gyr = window[:, sensor_offset + 3:sensor_offset + 6]   # 3-axis angular velocity
    mag = window[:, sensor_offset + 6:sensor_offset + 9]   # 3-axis magnetometer
    quat = window[:, sensor_offset + 9:sensor_offset + 13] # quaternion (S, X, Y, Z)

    # Norms
    acc_norm = _vector_norm(acc)
    gyr_norm = _vector_norm(gyr)

    # Jerk (derivative of acceleration)
    dt = 1.0 / SAMPLING_RATE_HZ
    jerk = np.diff(acc, axis=0) / dt
    jerk_norm = _vector_norm(jerk)

    features = {}

    # ── Acceleration features ──
    features["acc_rms"] = float(np.sqrt(np.mean(acc_norm ** 2)))
    features["acc_peak"] = float(np.max(acc_norm))
    features["acc_mean"] = float(np.mean(acc_norm))
    features["acc_std"] = float(np.std(acc_norm))
    features["acc_energy"] = float(np.sum(acc_norm ** 2))
    features["acc_range"] = float(np.ptp(acc_norm))

    # Per-axis acceleration stats
    for ax_idx, ax_name in enumerate(["x", "y", "z"]):
        features[f"acc_{ax_name}_mean"] = float(np.mean(acc[:, ax_idx]))
        features[f"acc_{ax_name}_std"] = float(np.std(acc[:, ax_idx]))
        features[f"acc_{ax_name}_skew"] = float(scipy_stats.skew(acc[:, ax_idx]))
        features[f"acc_{ax_name}_kurtosis"] = float(scipy_stats.kurtosis(acc[:, ax_idx]))

    # ── Angular velocity features ──
    features["gyr_rms"] = float(np.sqrt(np.mean(gyr_norm ** 2)))
    features["gyr_peak"] = float(np.max(gyr_norm))
    features["gyr_mean"] = float(np.mean(gyr_norm))
    features["gyr_std"] = float(np.std(gyr_norm))
    features["gyr_energy"] = float(np.sum(gyr_norm ** 2))
    features["gyr_range"] = float(np.ptp(gyr_norm))

    # Per-axis gyroscope stats
    for ax_idx, ax_name in enumerate(["x", "y", "z"]):
        features[f"gyr_{ax_name}_mean"] = float(np.mean(gyr[:, ax_idx]))
        features[f"gyr_{ax_name}_std"] = float(np.std(gyr[:, ax_idx]))

    # ── Jerk features (key for abruptness detection!) ──
    features["jerk_rms"] = float(np.sqrt(np.mean(jerk_norm ** 2)))
    features["jerk_peak"] = float(np.max(jerk_norm))
    features["jerk_mean"] = float(np.mean(jerk_norm))
    features["jerk_std"] = float(np.std(jerk_norm))

    # ── Quaternion features ──
    # Angular displacement: angle between first and last quaternion
    q_start = quat[0]
    q_end = quat[-1]
    # Relative quaternion: q_rel = q_end * conj(q_start)
    # For unit quaternions, the rotation angle = 2 * arccos(|dot product|)
    dot = np.clip(np.abs(np.sum(q_start * q_end)), -1.0, 1.0)
    features["quat_angular_disp"] = float(2.0 * np.arccos(dot))

    # Quaternion variability (spread of S component)
    features["quat_s_std"] = float(np.std(quat[:, 0]))

    return features


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract all features from a single window across all sensors.

    Args:
        window: (window_samples, 65) single window

    Returns:
        1D feature vector
    """
    all_features = {}

    for i, sensor_name in enumerate(SENSOR_NAMES):
        offset = i * CHANNELS_PER_SENSOR
        sensor_feats = extract_sensor_features(window, offset)
        # Prefix with sensor name
        for key, val in sensor_feats.items():
            all_features[f"{sensor_name}_{key}"] = val

    return np.array(list(all_features.values()), dtype=np.float32)


def extract_features_bulk(
    windows: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract handcrafted features from all windows.

    Args:
        windows: (W, window_samples, 65) array of windows

    Returns:
        features: (W, F) feature matrix
        feature_names: list of F feature names
    """
    # Get feature names from first window
    sample_features = {}
    for i, sensor_name in enumerate(SENSOR_NAMES):
        offset = i * CHANNELS_PER_SENSOR
        sensor_feats = extract_sensor_features(windows[0], offset)
        for key, val in sensor_feats.items():
            sample_features[f"{sensor_name}_{key}"] = val

    feature_names = list(sample_features.keys())
    n_features = len(feature_names)

    if verbose:
        print(f"Extracting {n_features} features from {windows.shape[0]} windows...")

    features = np.zeros((windows.shape[0], n_features), dtype=np.float32)
    for i in range(windows.shape[0]):
        features[i] = extract_window_features(windows[i])
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{windows.shape[0]} windows...")

    if verbose:
        print(f"  Done: feature matrix shape = {features.shape}")

    return features, feature_names


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
) -> tuple:
    """
    Z-score normalize features using training set statistics.

    Returns normalized arrays and the (mean, std) used.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    X_train_norm = (X_train - mean) / std

    results = [X_train_norm]
    if X_val is not None:
        results.append((X_val - mean) / std)
    if X_test is not None:
        results.append((X_test - mean) / std)

    results.append((mean, std))
    return tuple(results)
