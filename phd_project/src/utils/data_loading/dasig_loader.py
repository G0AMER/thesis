"""
DASIG Dataset Loader
====================
Provides structured access to the DASIG dataset (Abrupt and Standard
Industrial Gestures).

Dataset structure:
    DASIG/
        subjects_info.csv            (60 rows × 10 cols, BOM-encoded, ';' sep)
        sub001/
            sub001_FR_L_MIMU.csv     (~12 MB, ';' sep, ',' decimal, 3 header rows)
            sub001_FR_L_Arduino.csv  (';' sep, ',' decimal)
            sub001_FR_R_MIMU.csv
            sub001_FR_R_Arduino.csv
            sub001_LA_L_MIMU.csv
            sub001_LA_L_Arduino.csv
        sub002/ ...
        sub060/ ...

Key findings from exploration:
    - 60 subjects, 3 conditions each (FR_L, FR_R, LA_L), 180 MIMU + 180 Arduino
    - MIMU: 66 columns (Time_s + 5 segments × 13 channels), ~200 Hz, ~97s per trial
    - Arduino: alarm timestamps (Time (s), Stimuli)
    - CSV format: ';' separator, ',' decimal point
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

# Body segments
SEGMENTS = ["RF", "LF", "ST", "RUA", "LUA"]
SEGMENT_NAMES = {
    "RF": "Right Forearm",
    "LF": "Left Forearm",
    "ST": "Sternum",
    "RUA": "Right Upper Arm",
    "LUA": "Left Upper Arm",
}

# Sensor types per segment
SENSORS = [
    ("Acc", ["X", "Y", "Z"]),
    ("Gyro", ["X", "Y", "Z"]),
    ("Mag", ["X", "Y", "Z"]),
    ("Ori", ["S", "X", "Y", "Z"]),
]

# Build clean column names: Time_s, RF_Acc_X, RF_Acc_Y, ..., LUA_Ori_Z
MIMU_COLUMNS = ["Time_s"]
for _seg in SEGMENTS:
    for _sensor_name, _axes in SENSORS:
        for _axis in _axes:
            MIMU_COLUMNS.append(f"{_seg}_{_sensor_name}_{_axis}")

assert len(MIMU_COLUMNS) == 66, f"Expected 66 columns, got {len(MIMU_COLUMNS)}"

# Experimental conditions
CONDITIONS = ["FR_L", "FR_R", "LA_L"]
CONDITION_NAMES = {
    "FR_L": "Free movement, Left hand",
    "FR_R": "Free movement, Right hand",
    "LA_L": "Left arm with alarm interruptions",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubjectInfo:
    """Anthropometric info for a DASIG subject."""

    subject_id: str
    gender: Optional[str] = None
    age_range: Optional[str] = None
    height_m: Optional[float] = None
    weight_kg: Optional[float] = None
    dominant_arm: Optional[str] = None
    right_upper_arm_m: Optional[float] = None
    left_upper_arm_m: Optional[float] = None
    right_forearm_m: Optional[float] = None
    left_forearm_m: Optional[float] = None


@dataclass
class RecordingInfo:
    """Metadata for a single DASIG recording."""

    subject_id: str  # e.g., "sub001"
    condition: str  # "FR_L", "FR_R", or "LA_L"
    mimu_path: Path
    arduino_path: Optional[Path] = None

    @property
    def recording_key(self) -> str:
        return f"{self.subject_id}/{self.condition}"


@dataclass
class RecordingData:
    """Loaded data for a single DASIG recording."""

    info: RecordingInfo
    mimu: Optional[pd.DataFrame] = None
    arduino: Optional[pd.DataFrame] = None

    @property
    def duration_s(self) -> float:
        if self.mimu is not None and "Time_s" in self.mimu.columns:
            return float(self.mimu["Time_s"].iloc[-1] - self.mimu["Time_s"].iloc[0])
        return 0.0

    @property
    def sampling_rate(self) -> float:
        if self.mimu is not None and len(self.mimu) > 1 and "Time_s" in self.mimu.columns:
            return float(len(self.mimu) / self.duration_s)
        return 0.0

    @property
    def n_alarms(self) -> int:
        if self.arduino is not None:
            return len(self.arduino)
        return 0


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_subjects_info(dasig_root: Path) -> pd.DataFrame:
    """Load subjects_info.csv with proper encoding and separators."""
    paths = sorted(Path(dasig_root).rglob("subjects_info.csv"))
    if not paths:
        logger.warning(f"subjects_info.csv not found in {dasig_root}")
        return pd.DataFrame()
    return pd.read_csv(paths[0], sep=";", decimal=",", encoding="utf-8-sig")


def load_mimu(mimu_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a DASIG MIMU CSV file with proper parsing.

    The file has 3 header rows (segment names, sensor types, axis labels)
    followed by numerical data with ';' separator and ',' decimal.
    """
    try:
        df = pd.read_csv(
            mimu_path,
            skiprows=3,
            sep=";",
            decimal=",",
            header=None,
        )
        if len(df.columns) == len(MIMU_COLUMNS):
            df.columns = MIMU_COLUMNS
        else:
            logger.warning(
                f"Expected {len(MIMU_COLUMNS)} columns, got {len(df.columns)} "
                f"in {mimu_path.name}"
            )
            return None
        return df
    except Exception as e:
        logger.error(f"Error loading MIMU {mimu_path}: {e}")
        return None


def load_arduino(arduino_path: Path) -> Optional[pd.DataFrame]:
    """Load a DASIG Arduino CSV file (alarm timestamps)."""
    try:
        df = pd.read_csv(arduino_path, sep=";", decimal=",")
        return df
    except Exception as e:
        logger.error(f"Error loading Arduino {arduino_path}: {e}")
        return None


def discover_recordings(
    dasig_root: Path,
    subjects: Optional[list[str]] = None,
    conditions: Optional[list[str]] = None,
) -> list[RecordingInfo]:
    """
    Discover all MIMU/Arduino recording pairs.

    Parameters
    ----------
    dasig_root : Path
        Root directory containing the DASIG dataset.
    subjects : list[str], optional
        Filter to specific subjects. None = all.
    conditions : list[str], optional
        Filter to specific conditions. None = all.

    Returns
    -------
    list[RecordingInfo]
    """
    dasig_root = Path(dasig_root)
    # Find the DASIG subfolder if it exists
    dasig_dir = dasig_root / "DASIG" if (dasig_root / "DASIG").exists() else dasig_root

    mimu_files = sorted(dasig_dir.rglob("*_MIMU.csv"))
    recordings: list[RecordingInfo] = []

    for mimu_path in mimu_files:
        # Parse filename: sub001_FR_L_MIMU.csv
        stem = mimu_path.stem  # sub001_FR_L_MIMU
        parts = stem.split("_")
        if len(parts) < 4:
            continue

        subject_id = parts[0]
        condition = "_".join(parts[1:-1])  # FR_L, FR_R, or LA_L

        if subjects and subject_id not in subjects:
            continue
        if conditions and condition not in conditions:
            continue

        # Find matching Arduino file
        arduino_name = stem.replace("_MIMU", "_Arduino") + ".csv"
        arduino_path = mimu_path.parent / arduino_name

        recordings.append(
            RecordingInfo(
                subject_id=subject_id,
                condition=condition,
                mimu_path=mimu_path,
                arduino_path=arduino_path if arduino_path.exists() else None,
            )
        )

    logger.info(
        f"Discovered {len(recordings)} recordings "
        f"({len(set(r.subject_id for r in recordings))} subjects)"
    )
    return recordings


def load_recording(recording: RecordingInfo) -> RecordingData:
    """Load MIMU and Arduino data for a single recording."""
    mimu = load_mimu(recording.mimu_path)
    arduino = load_arduino(recording.arduino_path) if recording.arduino_path else None
    return RecordingData(info=recording, mimu=mimu, arduino=arduino)
