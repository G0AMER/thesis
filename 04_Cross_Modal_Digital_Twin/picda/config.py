"""Paths and dataset constants shared by the whole pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Raw segmented skeleton clips (one BVH per action instance, folder name = class)
RAW_DT_DIR = DATA_DIR / "inhard_dt" / "raw" / "InHARD-DT" / "Segmented" / "Skeleton"
RAW_REAL_DIR = DATA_DIR / "inhard" / "raw" / "Segmented" / "SkletonSegmented"  # sic, dataset spelling

# Outputs of this pipeline (kept separate from the legacy, invalid artefacts)
PROCESSED_DIR = DATA_DIR / "processed_v2"
INDEX_CSV = PROCESSED_DIR / "index.csv"
JOINT_NAMES_JSON = PROCESSED_DIR / "joint_names.json"
SPLITS_JSON = PROCESSED_DIR / "splits.json"
OUTPUT_DIR = PROJECT_ROOT / "output_v2"

TARGET_FPS = 30.0

# Meso-level InHARD classes. InHARD-DT file prefixes A00..A12 follow this exact order.
# "No action" exists only in InHARD and is excluded (no DT counterpart).
CLASS_NAMES = [
    "Assemble system",
    "Consult sheets",
    "Picking in front",
    "Picking left",
    "Put down component",
    "Put down measuring rod",
    "Put down screwdriver",
    "Put down subsystem",
    "Take component",
    "Take measuring rod",
    "Take screwdriver",
    "Take subsystem",
    "Turn sheets",
]
SHORT_NAMES = [
    "Assemble", "Consult", "PickFront", "PickLeft", "PutComp", "PutRod", "PutScrew",
    "PutSubsys", "TakeComp", "TakeRod", "TakeScrew", "TakeSubsys", "TurnSheets",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
EXCLUDED_CLASSES = {"no action"}

DOMAIN_DT = "dt"      # InHARD-DT, VR digital twin, 12 subjects
DOMAIN_REAL = "real"  # InHARD, physical workstation, 16 subjects
