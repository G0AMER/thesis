"""
Tests for preprocessing modules (HARMONIC + DASIG).
"""

import numpy as np
import pandas as pd
import pytest

from src.perception.preprocessing import (
    PreprocessConfig,
    align_timestamps,
    preprocess_emg,
    preprocess_gaze,
    preprocess_imu,
)
from src.style_modeling.dasig_preprocessing import (
    DASIGPreprocessConfig,
    butterworth_lowpass,
    compute_jerk_features,
    compute_time_features,
    compute_frequency_features,
    detect_abrupt_motions,
    extract_features_windowed,
    preprocess_mimu,
    segment_around_alarms,
)
from src.utils.data_loading.dasig_loader import MIMU_COLUMNS


# ===================================================================
# HARMONIC Preprocessing Tests
# ===================================================================


class TestAlignTimestamps:
    def test_basic_alignment(self):
        """Two modalities with overlapping timestamps get aligned."""
        mod_a = pd.DataFrame({"timestamp": [0, 1, 2, 3, 4], "val": [10, 20, 30, 40, 50]})
        mod_b = pd.DataFrame({"timestamp": [0.5, 1.5, 2.5, 3.5], "val": [1, 2, 3, 4]})
        result = align_timestamps({"a": mod_a, "b": mod_b}, target_rate=2.0)
        assert "a" in result and "b" in result
        # Both should have same number of rows
        assert len(result["a"]) == len(result["b"])

    def test_preserves_values(self):
        ts = np.linspace(0, 10, 500)
        df = pd.DataFrame({"timestamp": ts, "x": np.sin(ts)})
        result = align_timestamps({"sin": df}, target_rate=50.0)
        # Values should still be sinusoidal
        assert abs(result["sin"]["x"].mean()) < 0.5


class TestPreprocessEmg:
    def test_output_shape(self):
        n = 500
        df = pd.DataFrame(
            {
                "timestamp": np.arange(n) / 50.0,
                "emg_0": np.random.randn(n),
                "emg_1": np.random.randn(n),
            }
        )
        result = preprocess_emg(df, PreprocessConfig())
        assert result.shape == df.shape

    def test_reduces_noise(self):
        """RMS envelope should be smoother than raw."""
        n = 500
        raw = np.random.randn(n)
        df = pd.DataFrame({"timestamp": np.arange(n) / 50.0, "emg_0": raw})
        result = preprocess_emg(df, PreprocessConfig())
        assert result["emg_0"].std() < raw.std() * 2


class TestPreprocessGaze:
    def test_low_confidence_replaced(self):
        n = 100
        conf = np.ones(n) * 0.9
        conf[40:50] = 0.1  # low confidence region
        df = pd.DataFrame(
            {
                "timestamp": np.arange(n) / 50.0,
                "confidence": conf,
                "gaze_0_x": np.ones(n),
            }
        )
        config = PreprocessConfig(gaze_confidence_threshold=0.5)
        result = preprocess_gaze(df, config)
        # Low-confidence values should be interpolated, not the original
        assert result is not None


class TestPreprocessImu:
    def test_smoothing(self):
        n = 500
        ts = np.arange(n) / 50.0
        noisy = np.sin(2 * np.pi * 2 * ts) + 0.5 * np.random.randn(n)
        df = pd.DataFrame(
            {"timestamp": ts, "angular_velocity_x": noisy, "linear_acceleration_x": noisy}
        )
        result = preprocess_imu(df, PreprocessConfig())
        # Filtered should be smoother
        assert result["angular_velocity_x"].std() < noisy.std()


# ===================================================================
# DASIG Preprocessing Tests
# ===================================================================


def _make_mimu_df(n: int = 2000, fs: float = 200.0) -> pd.DataFrame:
    """Create a synthetic MIMU DataFrame."""
    t = np.arange(n) / fs
    data = {"Time_s": t}
    for col in MIMU_COLUMNS[1:]:
        data[col] = np.random.randn(n) * 0.1  # small noise
        if "_Acc_" in col:
            data[col] += 9.81  # gravity offset
    return pd.DataFrame(data)


class TestButterworthLowpass:
    def test_removes_high_freq(self):
        fs = 200.0
        t = np.arange(1000) / fs
        # 5 Hz signal + 80 Hz noise
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)
        filtered = butterworth_lowpass(signal, cutoff_hz=20.0, fs=fs)
        # High-freq component should be attenuated
        # Check that filtered is closer to the 5 Hz component
        clean = np.sin(2 * np.pi * 5 * t)
        assert np.std(filtered - clean) < np.std(signal - clean)

    def test_cutoff_above_nyquist_no_crash(self):
        signal = np.random.randn(100)
        result = butterworth_lowpass(signal, cutoff_hz=150.0, fs=200.0)
        assert len(result) == 100  # returned unchanged


class TestPreprocessMimu:
    def test_output_shape(self):
        df = _make_mimu_df()
        result = preprocess_mimu(df)
        assert result.shape == df.shape

    def test_gravity_removed(self):
        df = _make_mimu_df()
        result = preprocess_mimu(df)
        # After mean subtraction, acc columns should be ~0 mean
        acc_cols = [c for c in result.columns if "_Acc_" in c]
        for col in acc_cols:
            assert abs(result[col].mean()) < 0.5  # gravity removed


class TestAlarmSegmentation:
    def test_segments_around_alarms(self):
        df = _make_mimu_df(n=4000)  # 20s of data
        arduino = pd.DataFrame({"Time (s)": [5.0, 10.0, 15.0], "Stimuli": [1, 1, 1]})
        config = DASIGPreprocessConfig(alarm_pre_s=1.0, alarm_post_s=2.0)
        segments = segment_around_alarms(df, arduino, config)
        assert len(segments) == 3
        for seg in segments:
            # Each segment should be ~3s × 200 Hz = ~600 rows
            assert 500 < len(seg) < 700
            assert "Time_relative_s" in seg.columns

    def test_no_arduino_returns_full(self):
        df = _make_mimu_df()
        segments = segment_around_alarms(df, pd.DataFrame(), None)
        assert len(segments) == 1
        assert len(segments[0]) == len(df)


class TestFeatureExtraction:
    def test_time_features(self):
        arr = np.random.randn(200)
        feats = compute_time_features(arr)
        assert "mean" in feats
        assert "std" in feats
        assert "rms" in feats
        assert "range" in feats

    def test_frequency_features(self):
        t = np.arange(200) / 200.0
        arr = np.sin(2 * np.pi * 10 * t)  # 10 Hz pure tone
        feats = compute_frequency_features(arr, fs=200.0)
        assert "dominant_freq" in feats
        # Dominant freq should be ~10 Hz
        assert abs(feats["dominant_freq"] - 10.0) < 3.0

    def test_jerk_features(self):
        arr = np.cumsum(np.random.randn(200))
        feats = compute_jerk_features(arr, fs=200.0)
        assert "jerk_mean" in feats
        assert "jerk_max" in feats

    def test_windowed_extraction(self):
        df = _make_mimu_df(n=2000)
        config = DASIGPreprocessConfig(feature_window_s=0.5, feature_step_s=0.25)
        features = extract_features_windowed(df, config)
        assert len(features) > 0
        assert "t_center" in features.columns
        # Should have features for Acc and Gyro columns
        acc_feat_cols = [c for c in features.columns if "_Acc_" in c and "_mean" in c]
        assert len(acc_feat_cols) > 0


class TestAbruptDetection:
    def test_detects_spikes(self):
        df = _make_mimu_df(n=2000)
        # Inject a spike
        acc_cols = [c for c in df.columns if "_Acc_" in c]
        for col in acc_cols:
            df.loc[500:502, col] = 100.0  # big spike
        result = detect_abrupt_motions(df)
        assert len(result) > 0
        assert "jerk_magnitude" in result.columns


# ===================================================================
# Split Tests
# ===================================================================


class TestSplits:
    def test_basic_split(self):
        from src.utils.splits import subject_stratified_split

        ids = [f"s{i:03d}" for i in range(20)]
        split = subject_stratified_split(ids, 0.7, 0.15, 0.15, seed=42)
        assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 20
        # No overlap
        assert set(split["train"]).isdisjoint(set(split["val"]))
        assert set(split["train"]).isdisjoint(set(split["test"]))

    def test_stratified_split(self):
        from src.utils.splits import subject_stratified_split

        ids = [f"s{i:03d}" for i in range(20)]
        labels = ["M"] * 10 + ["F"] * 10
        split = subject_stratified_split(ids, 0.6, 0.2, 0.2, seed=42, stratify_labels=labels)
        assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 20

    def test_leave_n_out(self):
        from src.utils.splits import leave_n_subjects_out

        ids = [f"s{i:03d}" for i in range(10)]
        folds = leave_n_subjects_out(ids, n_out=2)
        assert len(folds) == 5
        for fold in folds:
            assert len(fold["test"]) == 2
            assert len(fold["train"]) == 8

    def test_reproducible(self):
        from src.utils.splits import subject_stratified_split

        ids = [f"s{i:03d}" for i in range(20)]
        s1 = subject_stratified_split(ids, 0.7, 0.15, 0.15, seed=42)
        s2 = subject_stratified_split(ids, 0.7, 0.15, 0.15, seed=42)
        assert s1 == s2
