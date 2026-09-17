"""
HARMONIC Dataset Loader
=======================
Provides structured access to the HARMONIC dataset (text modalities).

Dataset structure:
    harmonic_0.5.0/
        pXXX/
            run/NNN/text_data/*.csv   (22 CSV files per trial)
            check/NNN/text_data/*.csv (6 CSV files — gaze/calibration only)

Key findings from exploration:
    - 24 participants (p100–p123), 447 run trials total
    - Modalities: ada_joy, assistance_info, gaze_positions, input_info,
      joint_positions, myo_emg, myo_imu, myo_ori, pose_*, pupil_*, robot_position
    - EMG available in only 96/447 (21.5%) of run trials
    - Different modalities have different sampling rates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARMONIC_MODALITIES = [
    "ada_joy",
    "assistance_info",
    "gaze_positions",
    "input_info",
    "joint_positions",
    "myo_emg",
    "myo_imu",
    "myo_ori",
    "pose_left_face_2d",
    "pose_left_hand_left_2d",
    "pose_left_hand_right_2d",
    "pose_left_pose_2d",
    "pose_right_face_2d",
    "pose_right_hand_left_2d",
    "pose_right_hand_right_2d",
    "pose_right_pose_2d",
    "pupil_cal_eye0",
    "pupil_cal_eye1",
    "pupil_eye0",
    "pupil_eye1",
    "robot_position",
    "world_cal_positions",
]

# Modalities critical for the main experiments
CORE_MODALITIES = [
    "ada_joy",
    "gaze_positions",
    "joint_positions",
    "robot_position",
    "myo_emg",
    "myo_imu",
    "assistance_info",
    "input_info",
]

# Robot link names in robot_position.csv
ROBOT_LINKS = [
    "mico_link_base",
    "mico_link_1",
    "mico_link_2",
    "mico_link_3",
    "mico_link_4",
    "mico_link_5",
    "mico_link_hand",
    "mico_end_effector",
    "mico_fork_tip",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialInfo:
    """Metadata for a single HARMONIC trial."""

    participant: str  # e.g., "p100"
    trial_type: str  # "run" or "check"
    trial_id: str  # e.g., "000"
    text_data_dir: Path
    n_csv_files: int = 0
    available_modalities: list[str] = field(default_factory=list)
    has_emg: bool = False

    @property
    def trial_key(self) -> str:
        """Unique identifier: 'p100/run/000'."""
        return f"{self.participant}/{self.trial_type}/{self.trial_id}"


@dataclass
class TrialData:
    """Loaded data for a single HARMONIC trial."""

    info: TrialInfo
    modalities: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, modality: str) -> pd.DataFrame:
        return self.modalities[modality]

    def has(self, modality: str) -> bool:
        return modality in self.modalities and len(self.modalities[modality]) > 0

    @property
    def available(self) -> list[str]:
        return [k for k, v in self.modalities.items() if len(v) > 0]


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------


def discover_participants(harmonic_root: Path) -> list[str]:
    """Return sorted list of participant IDs (e.g., ['p100', 'p101', ...])."""
    harmonic_root = Path(harmonic_root)
    # Look for the versioned subfolder (e.g., harmonic_0.5.0/)
    versioned = list(harmonic_root.glob("harmonic_*"))
    base = versioned[0] if versioned else harmonic_root

    participants = sorted(
        d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("p")
    )
    logger.info(f"Found {len(participants)} participants in {base}")
    return participants


def discover_trials(
    harmonic_root: Path,
    trial_type: str = "run",
    participants: Optional[list[str]] = None,
    require_modalities: Optional[list[str]] = None,
    min_csv_size_kb: float = 0.5,
) -> list[TrialInfo]:
    """
    Discover all trials of a given type.

    Parameters
    ----------
    harmonic_root : Path
        Root directory containing the HARMONIC dataset.
    trial_type : str
        'run', 'check', or 'all'.
    participants : list[str], optional
        Filter to specific participants. None = all.
    require_modalities : list[str], optional
        Only return trials that have non-empty data for these modalities.
    min_csv_size_kb : float
        Minimum file size in KB to consider a modality as "available".

    Returns
    -------
    list[TrialInfo]
        Sorted list of discovered trials.
    """
    harmonic_root = Path(harmonic_root)
    versioned = list(harmonic_root.glob("harmonic_*"))
    base = versioned[0] if versioned else harmonic_root

    if participants is None:
        participants = discover_participants(harmonic_root)

    trials: list[TrialInfo] = []
    types_to_scan = ["run", "check"] if trial_type == "all" else [trial_type]

    for pid in participants:
        for ttype in types_to_scan:
            type_dir = base / pid / ttype
            if not type_dir.exists():
                continue
            for trial_dir in sorted(type_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                td = trial_dir / "text_data"
                if not td.exists():
                    continue

                csv_files = sorted(td.glob("*.csv"))
                available = []
                for f in csv_files:
                    if f.stat().st_size >= min_csv_size_kb * 1024:
                        available.append(f.stem)

                info = TrialInfo(
                    participant=pid,
                    trial_type=ttype,
                    trial_id=trial_dir.name,
                    text_data_dir=td,
                    n_csv_files=len(csv_files),
                    available_modalities=available,
                    has_emg="myo_emg" in available,
                )

                if require_modalities:
                    if all(m in available for m in require_modalities):
                        trials.append(info)
                else:
                    trials.append(info)

    logger.info(
        f"Discovered {len(trials)} {trial_type} trials "
        f"across {len(set(t.participant for t in trials))} participants"
    )
    return trials


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_trial(
    trial: TrialInfo,
    modalities: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
) -> TrialData:
    """
    Load CSV data for a single trial.

    Parameters
    ----------
    trial : TrialInfo
        Trial to load.
    modalities : list[str], optional
        Which modalities to load. None = all available CSVs.
    max_rows : int, optional
        Limit rows per modality (for quick previews).

    Returns
    -------
    TrialData
        Loaded trial with DataFrames for each modality.
    """
    td = trial.text_data_dir
    target_modalities = modalities or trial.available_modalities
    loaded: dict[str, pd.DataFrame] = {}

    for mod_name in target_modalities:
        csv_path = td / f"{mod_name}.csv"
        if not csv_path.exists():
            logger.debug(f"Modality {mod_name} not found in {trial.trial_key}")
            continue
        try:
            df = pd.read_csv(csv_path, nrows=max_rows)
            loaded[mod_name] = df
        except Exception as e:
            logger.warning(f"Error loading {mod_name} for {trial.trial_key}: {e}")
            loaded[mod_name] = pd.DataFrame()

    return TrialData(info=trial, modalities=loaded)


def load_participant_trials(
    harmonic_root: Path,
    participant: str,
    trial_type: str = "run",
    modalities: Optional[list[str]] = None,
) -> list[TrialData]:
    """Load all trials for a single participant."""
    trials = discover_trials(
        harmonic_root, trial_type=trial_type, participants=[participant]
    )
    return [load_trial(t, modalities=modalities) for t in trials]


def get_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the timestamp column in a HARMONIC CSV."""
    for col in df.columns:
        if col.lower() == "timestamp":
            return col
    return None


def get_data_columns(df: pd.DataFrame) -> list[str]:
    """Return non-timestamp, non-index numeric columns."""
    exclude = {"timestamp", "world_index", "world_index_corrected"}
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() not in exclude
    ]
