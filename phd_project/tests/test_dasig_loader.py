"""
Tests for DASIG data loader.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.data_loading.dasig_loader import (
    CONDITIONS,
    MIMU_COLUMNS,
    SEGMENTS,
    RecordingInfo,
    discover_recordings,
    load_arduino,
    load_mimu,
    load_recording,
    load_subjects_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mimu_csv(path: Path, n_rows: int = 1000):
    """Create a minimal MIMU CSV with correct format."""
    # 3 header rows
    header_1 = ";".join([""] + [seg for seg in SEGMENTS for _ in range(13)])
    header_2 = ";".join(["Time"] + ["Acc"] * 3 + ["Gyro"] * 3 + ["Mag"] * 3 + ["Ori"] * 4) * 1
    # Simplified: just write 3 junk header lines, then data
    lines = ["header1", "header2", "header3"]

    for i in range(n_rows):
        t = i / 200.0
        vals = [f"{t:.4f}".replace(".", ",")] + [
            f"{np.random.randn():.6f}".replace(".", ",") for _ in range(65)
        ]
        lines.append(";".join(vals))

    path.write_text("\n".join(lines), encoding="utf-8")


def _make_arduino_csv(path: Path, n_alarms: int = 5, duration: float = 5.0):
    """Create a minimal Arduino CSV."""
    lines = ["Time (s);Stimuli"]
    for i in range(n_alarms):
        t = (i + 1) * duration / (n_alarms + 1)
        lines.append(f"{t:.4f};1".replace(".", ","))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def dasig_tree(tmp_path: Path) -> Path:
    """Create a minimal fake DASIG directory structure."""
    root = tmp_path / "DASIG"
    root.mkdir()

    # subjects_info.csv
    info_df = pd.DataFrame(
        {
            "Subject": ["sub001", "sub002"],
            "Genre": ["M", "F"],
            "Age": ["25-30", "30-35"],
        }
    )
    info_df.to_csv(root / "subjects_info.csv", sep=";", index=False, encoding="utf-8-sig")

    # 2 subjects × 3 conditions
    for sub in ["sub001", "sub002"]:
        sub_dir = root / sub
        sub_dir.mkdir()
        for cond in CONDITIONS:
            _make_mimu_csv(sub_dir / f"{sub}_{cond}_MIMU.csv")
            _make_arduino_csv(sub_dir / f"{sub}_{cond}_Arduino.csv")

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_mimu_columns_count(self):
        assert len(MIMU_COLUMNS) == 66

    def test_mimu_columns_start_with_time(self):
        assert MIMU_COLUMNS[0] == "Time_s"

    def test_segments(self):
        assert len(SEGMENTS) == 5
        assert "RF" in SEGMENTS

    def test_conditions(self):
        assert len(CONDITIONS) == 3


class TestDiscovery:
    def test_discover_all(self, dasig_tree: Path):
        recs = discover_recordings(dasig_tree)
        assert len(recs) == 6  # 2 subjects × 3 conditions

    def test_discover_filter_subjects(self, dasig_tree: Path):
        recs = discover_recordings(dasig_tree, subjects=["sub001"])
        assert len(recs) == 3
        assert all(r.subject_id == "sub001" for r in recs)

    def test_discover_filter_conditions(self, dasig_tree: Path):
        recs = discover_recordings(dasig_tree, conditions=["LA_L"])
        assert len(recs) == 2
        assert all(r.condition == "LA_L" for r in recs)

    def test_recording_info_fields(self, dasig_tree: Path):
        recs = discover_recordings(dasig_tree, subjects=["sub001"], conditions=["FR_L"])
        assert len(recs) == 1
        r = recs[0]
        assert r.subject_id == "sub001"
        assert r.condition == "FR_L"
        assert r.arduino_path is not None


class TestLoading:
    def test_load_mimu(self, dasig_tree: Path):
        mimu_path = dasig_tree / "sub001" / "sub001_FR_L_MIMU.csv"
        df = load_mimu(mimu_path)
        assert df is not None
        assert len(df.columns) == 66
        assert df.columns[0] == "Time_s"
        assert len(df) == 1000

    def test_load_arduino(self, dasig_tree: Path):
        arduino_path = dasig_tree / "sub001" / "sub001_LA_L_Arduino.csv"
        df = load_arduino(arduino_path)
        assert df is not None
        assert len(df) == 5

    def test_load_recording(self, dasig_tree: Path):
        recs = discover_recordings(dasig_tree, subjects=["sub001"], conditions=["LA_L"])
        data = load_recording(recs[0])
        assert data.mimu is not None
        assert data.arduino is not None
        assert data.n_alarms == 5

    def test_load_subjects_info(self, dasig_tree: Path):
        df = load_subjects_info(dasig_tree)
        assert not df.empty
        assert len(df) == 2


class TestSubjectsInfo:
    def test_missing_subjects_info(self, tmp_path: Path):
        df = load_subjects_info(tmp_path)
        assert df.empty
