"""
Tests for HARMONIC data loader.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.data_loading.harmonic_loader import (
    CORE_MODALITIES,
    HARMONIC_MODALITIES,
    TrialInfo,
    discover_participants,
    discover_trials,
    get_data_columns,
    get_timestamp_column,
    load_trial,
)


# ---------------------------------------------------------------------------
# Fixtures: create a minimal HARMONIC-style directory tree
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_tree(tmp_path: Path) -> Path:
    """Create a minimal fake HARMONIC directory structure."""
    root = tmp_path / "harmonic"
    # 2 participants, each with 1 run and 1 rehearsal trial
    for pid in ["p100", "p101"]:
        for trial in ["run_001", "rehearsal_001"]:
            trial_dir = root / "text_data" / pid / trial
            trial_dir.mkdir(parents=True)

            # Create joystick CSV
            ts = np.arange(0, 5, 0.02)  # 5s at 50 Hz
            joy_df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "axes_0": np.sin(ts),
                    "axes_1": np.cos(ts),
                    "buttons_0": np.zeros(len(ts)),
                }
            )
            joy_df.to_csv(trial_dir / "joystick.csv", index=False)

            # Create joint_states CSV
            js_df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "position_0": ts * 0.1,
                    "velocity_0": np.ones(len(ts)) * 0.1,
                }
            )
            js_df.to_csv(trial_dir / "joint_states.csv", index=False)

            # Create gaze CSV (only for run)
            if "run" in trial:
                gaze_df = pd.DataFrame(
                    {
                        "timestamp": ts,
                        "confidence": np.ones(len(ts)) * 0.9,
                        "gaze_0_x": np.random.randn(len(ts)) * 0.1,
                        "gaze_0_y": np.random.randn(len(ts)) * 0.1,
                    }
                )
                gaze_df.to_csv(trial_dir / "gaze.csv", index=False)

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_participants(self, harmonic_tree: Path):
        parts = discover_participants(harmonic_tree)
        assert set(parts) == {"p100", "p101"}

    def test_discover_trials_all(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree)
        assert len(trials) == 4  # 2 participants × 2 trials

    def test_discover_trials_filter_type(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, trial_type="run")
        assert len(trials) == 2
        assert all(t.trial_type == "run" for t in trials)

    def test_discover_trials_filter_participant(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, participants=["p100"])
        assert len(trials) == 2
        assert all(t.participant == "p100" for t in trials)

    def test_discover_trials_filter_modality(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, required_modalities=["gaze"])
        assert len(trials) == 2  # only run trials have gaze
        assert all(t.trial_type == "run" for t in trials)

    def test_trial_info_fields(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, trial_type="run", participants=["p100"])
        assert len(trials) == 1
        t = trials[0]
        assert t.participant == "p100"
        assert t.trial_type == "run"
        assert "joystick" in t.available_modalities


class TestLoading:
    def test_load_trial(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, trial_type="run", participants=["p100"])
        data = load_trial(trials[0], modalities=["joystick", "joint_states"])
        assert "joystick" in data.modalities
        assert "joint_states" in data.modalities
        assert len(data.modalities["joystick"]) == 250  # 5s × 50 Hz

    def test_load_trial_missing_modality(self, harmonic_tree: Path):
        trials = discover_trials(harmonic_tree, trial_type="rehearsal", participants=["p100"])
        data = load_trial(trials[0], modalities=["gaze"])
        assert "gaze" not in data.modalities  # gaze not present in rehearsal


class TestConstants:
    def test_modalities_count(self):
        assert len(HARMONIC_MODALITIES) == 22

    def test_core_modalities_subset(self):
        assert all(m in HARMONIC_MODALITIES for m in CORE_MODALITIES)


class TestHelpers:
    def test_timestamp_column_detection(self):
        df = pd.DataFrame({"timestamp": [1, 2], "value": [3, 4]})
        assert get_timestamp_column(df) == "timestamp"

    def test_data_columns_exclude_timestamp(self):
        df = pd.DataFrame({"timestamp": [1, 2], "x": [3, 4], "y": [5, 6]})
        cols = get_data_columns(df)
        assert "timestamp" not in cols
        assert "x" in cols
        assert "y" in cols
