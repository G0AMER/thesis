"""
HARMONIC Signal Preprocessing
==============================
Timestamp alignment, EMG filtering, and signal normalization for the
HARMONIC dataset modalities.

Key preprocessing steps (from exploration report):
1. Timestamp alignment — resample all modalities to a common rate
2. EMG filtering — band-pass → rectification → RMS envelope → z-score
3. Gaze filtering — confidence-based filtering + interpolation
4. Signal normalization — per-participant z-score normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Configuration for HARMONIC signal preprocessing."""

    # Resampling
    target_rate_hz: float = 50.0  # Common resampling rate for all modalities

    # EMG filtering
    emg_bandpass_low: float = 20.0  # Hz — high-pass cutoff
    emg_bandpass_high: float = 450.0  # Hz — low-pass cutoff (Nyquist-limited)
    emg_notch_freq: float = 50.0  # Hz — power-line notch filter (EU)
    emg_rms_window_ms: float = 100.0  # RMS envelope window in ms
    emg_original_rate_hz: float = 50.0  # Myo armband EMG sample rate (approximate)

    # Gaze filtering
    gaze_confidence_threshold: float = 0.5
    gaze_interpolation_method: str = "linear"
    gaze_smoothing_window: int = 5  # Moving average window

    # Normalization
    normalization_method: str = "zscore"  # 'zscore', 'minmax', or 'none'

    # General
    filter_order: int = 4  # Butterworth filter order


# ---------------------------------------------------------------------------
# Timestamp Alignment
# ---------------------------------------------------------------------------


def align_timestamps(
    modalities: dict[str, pd.DataFrame],
    target_rate_hz: float = 50.0,
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """
    Resample all modalities to a common time base using linear interpolation.

    Each modality is resampled to a uniform grid at `target_rate_hz` Hz,
    covering the time range common to all modalities.

    Parameters
    ----------
    modalities : dict[str, pd.DataFrame]
        Dict of modality name → DataFrame. Each must have a 'timestamp' column.
    target_rate_hz : float
        Target uniform sampling rate in Hz.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    dict[str, pd.DataFrame]
        Resampled DataFrames with a new 'time_s' column (seconds from start).
    """
    # Find the common time range
    t_starts = []
    t_ends = []
    for name, df in modalities.items():
        if df.empty or timestamp_col not in df.columns:
            continue
        ts = df[timestamp_col].values
        t_starts.append(ts.min())
        t_ends.append(ts.max())

    if not t_starts:
        logger.warning("No modalities with timestamps found.")
        return modalities

    t_start = max(t_starts)  # Latest start (intersection)
    t_end = min(t_ends)  # Earliest end (intersection)

    if t_end <= t_start:
        logger.warning(f"No overlapping time range: start={t_start}, end={t_end}")
        return modalities

    duration = t_end - t_start
    n_samples = int(duration * target_rate_hz) + 1
    common_time = np.linspace(t_start, t_end, n_samples)
    time_s = common_time - t_start  # Relative seconds from start

    logger.info(
        f"Aligning to common time base: {duration:.2f}s, "
        f"{n_samples} samples at {target_rate_hz} Hz"
    )

    resampled: dict[str, pd.DataFrame] = {}

    for name, df in modalities.items():
        if df.empty or timestamp_col not in df.columns:
            resampled[name] = df
            continue

        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).copy()
        ts_orig = df_sorted[timestamp_col].values

        # Identify numeric data columns (exclude timestamp and index columns)
        exclude = {timestamp_col, "world_index", "world_index_corrected"}
        data_cols = [
            c
            for c in df_sorted.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df_sorted[c])
        ]

        if not data_cols:
            resampled[name] = df
            continue

        # Interpolate each column to the common time grid
        new_data = {"time_s": time_s}
        for col in data_cols:
            vals = df_sorted[col].values.astype(float)
            # Remove NaN for interpolation
            mask = ~np.isnan(vals)
            if mask.sum() < 2:
                new_data[col] = np.full(n_samples, np.nan)
            else:
                new_data[col] = np.interp(common_time, ts_orig[mask], vals[mask])

        resampled[name] = pd.DataFrame(new_data)
        logger.debug(
            f"  {name}: {len(df)} → {n_samples} samples "
            f"({len(data_cols)} columns)"
        )

    return resampled


# ---------------------------------------------------------------------------
# EMG Preprocessing
# ---------------------------------------------------------------------------


def preprocess_emg(
    emg_df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Full EMG preprocessing pipeline.

    Steps:
        1. Band-pass filter (20–450 Hz) — remove DC offset and high-freq noise
        2. Full-wave rectification (absolute value)
        3. RMS envelope (sliding window)
        4. Z-score normalization per channel

    Parameters
    ----------
    emg_df : pd.DataFrame
        Raw EMG data with columns 'emg0'–'emg7' (and optionally timestamp cols).
    config : PreprocessConfig, optional
        Filtering parameters.

    Returns
    -------
    pd.DataFrame
        Preprocessed EMG with same columns. Contains RMS envelope values,
        z-score normalized per channel.
    """
    if config is None:
        config = PreprocessConfig()

    if emg_df.empty:
        return emg_df

    # Identify EMG channels
    emg_cols = [c for c in emg_df.columns if c.startswith("emg")]
    if not emg_cols:
        logger.warning("No EMG columns found (expected emg0–emg7)")
        return emg_df

    result = emg_df.copy()
    fs = config.emg_original_rate_hz

    for col in emg_cols:
        raw = result[col].values.astype(float)

        # Skip if all zeros or NaN
        if np.all(raw == 0) or np.all(np.isnan(raw)):
            continue

        # Step 1: Band-pass filter (only if Nyquist allows)
        nyquist = fs / 2.0
        if config.emg_bandpass_high < nyquist and config.emg_bandpass_low < nyquist:
            low = config.emg_bandpass_low / nyquist
            high = config.emg_bandpass_high / nyquist
            b, a = scipy_signal.butter(config.filter_order, [low, high], btype="band")
            filtered = scipy_signal.filtfilt(b, a, raw)
        else:
            # Myo armband is ~50 Hz — too low for standard EMG band-pass.
            # Apply only a high-pass to remove DC offset.
            if config.emg_bandpass_low < nyquist:
                low = config.emg_bandpass_low / nyquist
                b, a = scipy_signal.butter(config.filter_order, low, btype="high")
                filtered = scipy_signal.filtfilt(b, a, raw)
            else:
                # Can't even high-pass — just remove mean
                filtered = raw - np.nanmean(raw)
                logger.debug(
                    f"EMG rate ({fs} Hz) too low for bandpass; "
                    f"using mean subtraction for {col}"
                )

        # Step 2: Full-wave rectification
        rectified = np.abs(filtered)

        # Step 3: RMS envelope
        window_samples = max(1, int(config.emg_rms_window_ms / 1000.0 * fs))
        envelope = _rms_envelope(rectified, window_samples)

        # Step 4: Z-score normalization
        if config.normalization_method == "zscore":
            mean = np.nanmean(envelope)
            std = np.nanstd(envelope)
            if std > 1e-8:
                envelope = (envelope - mean) / std
            else:
                envelope = envelope - mean
        elif config.normalization_method == "minmax":
            vmin, vmax = np.nanmin(envelope), np.nanmax(envelope)
            if (vmax - vmin) > 1e-8:
                envelope = (envelope - vmin) / (vmax - vmin)

        result[col] = envelope

    return result


def _rms_envelope(signal_data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute RMS envelope using a sliding window."""
    if window_size <= 1:
        return np.abs(signal_data)
    squared = signal_data**2
    kernel = np.ones(window_size) / window_size
    mean_sq = np.convolve(squared, kernel, mode="same")
    return np.sqrt(np.maximum(mean_sq, 0))


# ---------------------------------------------------------------------------
# Gaze Preprocessing
# ---------------------------------------------------------------------------


def preprocess_gaze(
    gaze_df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Clean and smooth gaze tracking data.

    Steps:
        1. Filter by confidence threshold
        2. Interpolate missing samples
        3. Smooth with moving average

    Parameters
    ----------
    gaze_df : pd.DataFrame
        Raw gaze data with norm_pos_x, norm_pos_y, confidence columns.
    config : PreprocessConfig, optional

    Returns
    -------
    pd.DataFrame
        Cleaned gaze data.
    """
    if config is None:
        config = PreprocessConfig()

    if gaze_df.empty:
        return gaze_df

    result = gaze_df.copy()

    # Step 1: Mark low-confidence samples as NaN
    if "confidence" in result.columns:
        low_conf = result["confidence"] < config.gaze_confidence_threshold
        n_removed = low_conf.sum()
        logger.debug(
            f"Gaze: removing {n_removed}/{len(result)} low-confidence samples "
            f"(< {config.gaze_confidence_threshold})"
        )
        for col in ["norm_pos_x", "norm_pos_y"]:
            if col in result.columns:
                result.loc[low_conf, col] = np.nan

    # Step 2: Interpolate gaps
    gaze_cols = ["norm_pos_x", "norm_pos_y"]
    for col in gaze_cols:
        if col in result.columns:
            result[col] = result[col].interpolate(
                method=config.gaze_interpolation_method, limit_direction="both"
            )

    # Step 3: Smooth
    if config.gaze_smoothing_window > 1:
        for col in gaze_cols:
            if col in result.columns:
                result[col] = (
                    result[col]
                    .rolling(window=config.gaze_smoothing_window, center=True, min_periods=1)
                    .mean()
                )

    return result


# ---------------------------------------------------------------------------
# IMU Preprocessing
# ---------------------------------------------------------------------------


def preprocess_imu(
    imu_df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Preprocess IMU data: extract key channels and low-pass filter.

    Extracts linear_acceleration (x,y,z) and angular_velocity (x,y,z),
    applies low-pass Butterworth filter to remove high-frequency noise.
    """
    if config is None:
        config = PreprocessConfig()

    if imu_df.empty:
        return imu_df

    result = imu_df.copy()

    # Low-pass filter at 20 Hz (assumes ~50 Hz sample rate)
    fs = config.emg_original_rate_hz  # Myo IMU has similar rate to EMG
    cutoff = 20.0
    nyquist = fs / 2.0

    accel_cols = [c for c in result.columns if c.startswith("linear_acceleration")]
    gyro_cols = [c for c in result.columns if c.startswith("angular_velocity")]
    target_cols = accel_cols + gyro_cols

    if cutoff < nyquist and target_cols:
        b, a = scipy_signal.butter(config.filter_order, cutoff / nyquist, btype="low")
        for col in target_cols:
            vals = result[col].values.astype(float)
            if not np.all(np.isnan(vals)) and len(vals) > config.filter_order * 3:
                result[col] = scipy_signal.filtfilt(b, a, vals)

    return result


# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------


def compute_participant_stats(
    trial_data_list: list[pd.DataFrame],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute per-column mean and std across all trials for a participant.

    Used for per-participant z-score normalization.
    """
    all_values: dict[str, list[np.ndarray]] = {col: [] for col in columns}

    for df in trial_data_list:
        for col in columns:
            if col in df.columns:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    all_values[col].append(vals)

    stats: dict[str, dict[str, float]] = {}
    for col in columns:
        if all_values[col]:
            concatenated = np.concatenate(all_values[col])
            stats[col] = {
                "mean": float(np.mean(concatenated)),
                "std": float(np.std(concatenated)),
                "min": float(np.min(concatenated)),
                "max": float(np.max(concatenated)),
            }
        else:
            stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}

    return stats


def normalize_dataframe(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    method: str = "zscore",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Normalize columns of a DataFrame using pre-computed statistics.

    Parameters
    ----------
    df : pd.DataFrame
    stats : dict
        From `compute_participant_stats()`.
    method : str
        'zscore' or 'minmax'.
    columns : list[str], optional
        Which columns to normalize (default: all in stats).
    """
    result = df.copy()
    target_cols = columns or list(stats.keys())

    for col in target_cols:
        if col not in result.columns or col not in stats:
            continue
        s = stats[col]
        if method == "zscore":
            std = s["std"] if s["std"] > 1e-8 else 1.0
            result[col] = (result[col] - s["mean"]) / std
        elif method == "minmax":
            rng = s["max"] - s["min"]
            rng = rng if rng > 1e-8 else 1.0
            result[col] = (result[col] - s["min"]) / rng

    return result
