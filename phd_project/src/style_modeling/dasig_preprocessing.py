"""
DASIG Signal Preprocessing
===========================
Preprocessing functions for DASIG MIMU sensor data:
  - Bias / gravity removal
  - Low-pass Butterworth filtering
  - Alarm-based temporal windowing
  - Windowed feature extraction (time + frequency domain)
  - Per-subject normalisation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DASIGPreprocessConfig:
    """Configuration for DASIG preprocessing."""

    # Filtering
    mimu_sampling_rate: float = 200.0  # nominal ~200 Hz
    accel_lowpass_hz: float = 20.0
    gyro_lowpass_hz: float = 15.0
    filter_order: int = 4

    # Windowing around alarms
    alarm_pre_s: float = 2.0  # seconds before alarm
    alarm_post_s: float = 3.0  # seconds after alarm

    # Feature extraction
    feature_window_s: float = 1.0  # sliding window size
    feature_step_s: float = 0.5  # sliding window step

    # Normalization
    normalization_method: str = "zscore"  # 'zscore' or 'minmax'

    # Abruptness detection thresholds (from exploration)
    jerk_threshold_percentile: float = 95.0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        logger.warning(
            f"Cutoff {cutoff_hz} Hz >= Nyquist {nyq} Hz — skipping filter"
        )
        return data
    b, a = sp_signal.butter(order, cutoff_hz / nyq, btype="low")
    if len(data) < 3 * max(len(b), len(a)):
        return data
    return sp_signal.filtfilt(b, a, data, axis=0)


def preprocess_mimu(
    df: pd.DataFrame,
    config: DASIGPreprocessConfig | None = None,
) -> pd.DataFrame:
    """
    Apply physical preprocessing to a MIMU DataFrame.

    Steps:
        1. Subtract per-column mean from accelerometer data (gravity/bias removal)
        2. Low-pass filter accelerometers at accel_lowpass_hz
        3. Low-pass filter gyroscopes at gyro_lowpass_hz
        4. Leave magnetometer and orientation quaternions untouched

    Returns a new DataFrame with filtered values.
    """
    if config is None:
        config = DASIGPreprocessConfig()

    result = df.copy()
    fs = config.mimu_sampling_rate

    for col in result.columns:
        if col == "Time_s":
            continue
        arr = result[col].values.astype(float)

        if "_Acc_" in col:
            arr = arr - np.nanmean(arr)
            arr = butterworth_lowpass(arr, config.accel_lowpass_hz, fs, config.filter_order)
        elif "_Gyro_" in col:
            arr = butterworth_lowpass(arr, config.gyro_lowpass_hz, fs, config.filter_order)
        # Mag and Ori: left as-is

        result[col] = arr

    return result


# ---------------------------------------------------------------------------
# Alarm-based windowing
# ---------------------------------------------------------------------------


def segment_around_alarms(
    mimu: pd.DataFrame,
    arduino: pd.DataFrame,
    config: DASIGPreprocessConfig | None = None,
) -> list[pd.DataFrame]:
    """
    Extract temporal windows around each alarm event.

    Parameters
    ----------
    mimu : pd.DataFrame
        Preprocessed MIMU data with 'Time_s' column.
    arduino : pd.DataFrame
        Arduino data with time column (first column).
    config : DASIGPreprocessConfig

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per alarm window.
    """
    if config is None:
        config = DASIGPreprocessConfig()

    if arduino is None or arduino.empty:
        logger.info("No alarm data — returning entire recording as single segment")
        return [mimu]

    # Arduino time column may be named differently
    time_col = arduino.columns[0]
    alarm_times = arduino[time_col].values.astype(float)
    mimu_time = mimu["Time_s"].values

    segments = []
    for t_alarm in alarm_times:
        t_start = t_alarm - config.alarm_pre_s
        t_end = t_alarm + config.alarm_post_s
        mask = (mimu_time >= t_start) & (mimu_time <= t_end)
        window = mimu.loc[mask].copy()
        if len(window) > 10:
            # Reset time relative to alarm
            window["Time_relative_s"] = window["Time_s"] - t_alarm
            segments.append(window)

    logger.info(f"Extracted {len(segments)} alarm segments from {len(alarm_times)} alarms")
    return segments


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def compute_time_features(window: np.ndarray) -> dict[str, float]:
    """Compute time-domain features for a single window of data."""
    if len(window) == 0:
        return {}
    return {
        "mean": float(np.nanmean(window)),
        "std": float(np.nanstd(window)),
        "min": float(np.nanmin(window)),
        "max": float(np.nanmax(window)),
        "range": float(np.nanmax(window) - np.nanmin(window)),
        "rms": float(np.sqrt(np.nanmean(window**2))),
        "median": float(np.nanmedian(window)),
        "iqr": float(np.nanpercentile(window, 75) - np.nanpercentile(window, 25)),
        "skewness": float(_safe_skew(window)),
        "kurtosis": float(_safe_kurtosis(window)),
    }


def compute_frequency_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """Compute frequency-domain features via FFT."""
    if len(window) < 4:
        return {}

    # Remove mean, apply Hanning window
    w = window - np.nanmean(window)
    w = w * np.hanning(len(w))
    fft_vals = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / fs)

    # Avoid division by zero
    total_power = np.sum(fft_vals**2)
    if total_power < 1e-12:
        return {"dominant_freq": 0.0, "spectral_entropy": 0.0, "band_power_low": 0.0}

    # Power spectral density (normalized)
    psd = fft_vals**2 / total_power
    psd_safe = psd[psd > 0]

    # Dominant frequency
    dominant_idx = np.argmax(fft_vals[1:]) + 1
    dominant_freq = float(freqs[dominant_idx])

    # Spectral entropy
    spectral_entropy = float(-np.sum(psd_safe * np.log2(psd_safe)))

    # Band powers
    low_mask = freqs <= 5.0
    mid_mask = (freqs > 5.0) & (freqs <= 20.0)
    high_mask = freqs > 20.0

    return {
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "band_power_low": float(np.sum(fft_vals[low_mask] ** 2)),
        "band_power_mid": float(np.sum(fft_vals[mid_mask] ** 2)),
        "band_power_high": float(np.sum(fft_vals[high_mask] ** 2)),
    }


def compute_jerk_features(window: np.ndarray, fs: float) -> dict[str, float]:
    """Compute jerk (derivative of acceleration) features."""
    if len(window) < 3:
        return {}
    jerk = np.gradient(window, 1.0 / fs)
    return {
        "jerk_mean": float(np.nanmean(np.abs(jerk))),
        "jerk_max": float(np.nanmax(np.abs(jerk))),
        "jerk_rms": float(np.sqrt(np.nanmean(jerk**2))),
    }


def extract_features_windowed(
    df: pd.DataFrame,
    config: DASIGPreprocessConfig | None = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extract sliding-window features from MIMU data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed MIMU data with 'Time_s'.
    config : DASIGPreprocessConfig
    columns : list[str], optional
        Which sensor columns to extract features from.
        Defaults to all Acc and Gyro columns.

    Returns
    -------
    pd.DataFrame
        One row per window with all features.
    """
    if config is None:
        config = DASIGPreprocessConfig()

    if columns is None:
        columns = [c for c in df.columns if "_Acc_" in c or "_Gyro_" in c]

    fs = config.mimu_sampling_rate
    win_samples = int(config.feature_window_s * fs)
    step_samples = int(config.feature_step_s * fs)
    times = df["Time_s"].values

    all_features: list[dict] = []

    for start_idx in range(0, len(df) - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        t_center = float(times[start_idx] + times[end_idx - 1]) / 2.0

        row = {"t_center": t_center, "t_start": float(times[start_idx])}

        for col in columns:
            arr = df[col].values[start_idx:end_idx].astype(float)
            prefix = col

            # Time-domain
            tf = compute_time_features(arr)
            for k, v in tf.items():
                row[f"{prefix}_{k}"] = v

            # Frequency-domain
            ff = compute_frequency_features(arr, fs)
            for k, v in ff.items():
                row[f"{prefix}_{k}"] = v

            # Jerk (for accelerometer only)
            if "_Acc_" in col:
                jf = compute_jerk_features(arr, fs)
                for k, v in jf.items():
                    row[f"{prefix}_{k}"] = v

        all_features.append(row)

    return pd.DataFrame(all_features)


# ---------------------------------------------------------------------------
# Per-subject normalization
# ---------------------------------------------------------------------------


def compute_subject_stats(
    feature_dfs: list[pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """
    Compute per-column mean and std across multiple feature DataFrames
    for a given subject.
    """
    if not feature_dfs:
        return {}

    combined = pd.concat(feature_dfs, ignore_index=True)
    stats = {}
    for col in combined.columns:
        if col.startswith("t_"):
            continue
        vals = combined[col].dropna()
        stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 1.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return stats


def normalize_features(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    method: str = "zscore",
) -> pd.DataFrame:
    """Apply z-score or min-max normalization using precomputed stats."""
    result = df.copy()
    for col in result.columns:
        if col.startswith("t_") or col not in stats:
            continue
        s = stats[col]
        if method == "zscore":
            std = s["std"] if s["std"] > 1e-12 else 1.0
            result[col] = (result[col] - s["mean"]) / std
        elif method == "minmax":
            rng = s["max"] - s["min"]
            rng = rng if rng > 1e-12 else 1.0
            result[col] = (result[col] - s["min"]) / rng
    return result


# ---------------------------------------------------------------------------
# Abrupt motion detection
# ---------------------------------------------------------------------------


def detect_abrupt_motions(
    df: pd.DataFrame,
    config: DASIGPreprocessConfig | None = None,
) -> pd.DataFrame:
    """
    Detect abrupt motions based on jerk magnitude exceeding a
    percentile-based threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed MIMU data with 'Time_s'.
    config : DASIGPreprocessConfig

    Returns
    -------
    pd.DataFrame
        Rows from df where abrupt motion was detected, with added
        column 'jerk_magnitude'.
    """
    if config is None:
        config = DASIGPreprocessConfig()

    acc_cols = [c for c in df.columns if "_Acc_" in c]
    if not acc_cols:
        return pd.DataFrame()

    fs = config.mimu_sampling_rate

    # Compute jerk magnitude across all accelerometer axes
    jerk_mag = np.zeros(len(df))
    for col in acc_cols:
        arr = df[col].values.astype(float)
        jerk = np.gradient(arr, 1.0 / fs)
        jerk_mag += jerk**2
    jerk_mag = np.sqrt(jerk_mag)

    threshold = np.percentile(jerk_mag, config.jerk_threshold_percentile)
    abrupt_mask = jerk_mag > threshold

    result = df.loc[abrupt_mask].copy()
    result["jerk_magnitude"] = jerk_mag[abrupt_mask]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_skew(arr: np.ndarray) -> float:
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-12:
        return 0.0
    return float(np.nanmean(((arr - m) / s) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-12:
        return 0.0
    return float(np.nanmean(((arr - m) / s) ** 4) - 3.0)
