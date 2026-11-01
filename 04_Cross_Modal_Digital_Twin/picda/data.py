"""Loading processed clips, subject-disjoint splits, augmentation and torch datasets."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C
from .skeleton import canonicalize, joint_names, parent_indices, selection_indices


# --------------------------------------------------------------------------- loading

def load_index(only_ok: bool = True) -> pd.DataFrame:
    if not C.INDEX_CSV.exists():
        raise FileNotFoundError(f"{C.INDEX_CSV} not found. Run notebook 01 first.")
    df = pd.read_csv(C.INDEX_CSV)
    if only_ok:
        df = df[df["status"] == "ok"].copy()
        df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def all_joint_names() -> List[str]:
    with open(C.JOINT_NAMES_JSON) as f:
        return json.load(f)


def load_raw(df: pd.DataFrame, joint_set: Optional[str] = "body19") -> List[np.ndarray]:
    """World coordinates (cm), 30 Hz, (T, V, 3) per clip. joint_set=None keeps all 59 joints."""
    sel = None if joint_set is None else selection_indices(all_joint_names(), joint_set)
    clips = []
    for rel in df["relpath"]:
        j = np.load(C.PROCESSED_DIR / rel)["joints"]
        clips.append(j if sel is None else j[:, sel])
    return clips


@dataclass
class Canon:
    center: str = "first_frame"
    yaw: str = "first_frame"
    scale: str = "torso"


def load_clips(df: pd.DataFrame, joint_set: str = "body19", canon: Canon = Canon()) -> List[np.ndarray]:
    names = joint_names(joint_set)
    return [canonicalize(c, names, canon.center, canon.yaw, canon.scale) for c in load_raw(df, joint_set)]


# --------------------------------------------------------------------------- splits

def subject_folds(subjects: Sequence[str], n_folds: int) -> List[List[str]]:
    """Deterministic, interleaved subject-disjoint folds (sorted subjects, round robin)."""
    s = sorted(set(subjects))
    return [s[i::n_folds] for i in range(n_folds)]


def make_splits(index: pd.DataFrame, n_folds_real: int = 4, n_folds_dt: int = 4) -> Dict:
    real_subj = index.loc[index.domain == C.DOMAIN_REAL, "subject"].unique()
    dt_subj = index.loc[index.domain == C.DOMAIN_DT, "subject"].unique()
    splits = {
        "real_test_folds": subject_folds(real_subj, n_folds_real),
        "dt_test_folds": subject_folds(dt_subj, n_folds_dt),
    }
    C.SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(C.SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=1)
    return splits


def load_splits() -> Dict:
    with open(C.SPLITS_JSON) as f:
        return json.load(f)


def few_shot_indices(labels: np.ndarray, pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k clips per class drawn from `pool` (indices into labels). Classes with < k clips give all."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(C.NUM_CLASSES):
        cand = pool[labels[pool] == c]
        if len(cand):
            chosen.append(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.concatenate(chosen)


# --------------------------------------------------------------------------- augmentation

def resize_time(seq: np.ndarray, T: int) -> np.ndarray:
    n = seq.shape[0]
    if n == 1:
        return np.repeat(seq, T, axis=0)
    idx = np.linspace(0, n - 1, T)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None, None]
    return ((1 - w) * seq[i0] + w * seq[i1]).astype(np.float32)


def random_crop(seq: np.ndarray, rng, lo: float, hi: float) -> np.ndarray:
    n = seq.shape[0]
    L = max(2, int(round(n * rng.uniform(lo, hi))))
    if L >= n:
        return seq
    s = rng.integers(0, n - L + 1)
    return seq[s:s + L]


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class AugParams:
    crop: tuple = (1.0, 1.0)
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    scale: tuple = (1.0, 1.0)
    jitter: float = 0.0


NO_AUG = AugParams()
WEAK_AUG = AugParams(crop=(0.85, 1.0), yaw_deg=10, tilt_deg=3, scale=(0.95, 1.05), jitter=0.0)
STRONG_AUG = AugParams(crop=(0.5, 1.0), yaw_deg=30, tilt_deg=10, scale=(0.8, 1.2), jitter=0.02)
# Boundary randomisation: 58% of InHARD-DT clips are fixed windows (0.8/2/2.5/3/3.4 s), so after
# resizing to 64 frames clip length becomes an apparent-speed cue that does not exist in the
# hand-cut InHARD clips. Aggressive random sub-windows break that shortcut.
TEMPORAL_AUG = AugParams(crop=(0.3, 1.0), yaw_deg=10, tilt_deg=3, scale=(0.95, 1.05))
SOURCE_AUGS = {"weak": WEAK_AUG, "temporal": TEMPORAL_AUG}


def freeze_left_arm(seq: np.ndarray, joint_set: str = "body19") -> np.ndarray:
    """Remove left-arm motion relative to the left shoulder (elbow, wrist, knuckle keep their
    frame-0 offset). InHARD operators work bimanually while VR operators mostly use the right
    hand, so this tests whether left-arm motion is a transfer distractor."""
    n = joint_names(joint_set)
    sh = n.index("LeftArm")
    out = seq.copy()
    for j in ("LeftForeArm", "LeftHand", "LeftHandMiddle1"):
        k = n.index(j)
        out[:, k] = seq[0, k] + (seq[:, sh] - seq[0, sh])
    return out


def augment(seq: np.ndarray, p: AugParams, rng) -> np.ndarray:
    if p.crop != (1.0, 1.0):
        seq = random_crop(seq, rng, *p.crop)
    R = np.eye(3)
    if p.yaw_deg:
        R = _rot("y", np.deg2rad(rng.uniform(-p.yaw_deg, p.yaw_deg))) @ R
    if p.tilt_deg:
        R = _rot("x", np.deg2rad(rng.uniform(-p.tilt_deg, p.tilt_deg))) @ R
        R = _rot("z", np.deg2rad(rng.uniform(-p.tilt_deg, p.tilt_deg))) @ R
    seq = seq @ R.T.astype(np.float32)
    if p.scale != (1.0, 1.0):
        seq = seq * rng.uniform(*p.scale)
    if p.jitter:
        seq = seq + rng.normal(0, p.jitter, seq.shape).astype(np.float32)
    return seq.astype(np.float32)


# --------------------------------------------------------------------------- features

def to_features(seq: np.ndarray, features: str, parents: List[int]) -> np.ndarray:
    """(T, V, 3) -> (C, T, V)."""
    if features == "joint":
        x = seq
    elif features == "joint_motion":
        motion = np.zeros_like(seq)
        motion[1:] = seq[1:] - seq[:-1]
        x = np.concatenate([seq, motion], axis=-1)
    elif features == "joint_bone":
        bone = np.zeros_like(seq)
        for i, p in enumerate(parents):
            if p >= 0:
                bone[:, i] = seq[:, i] - seq[:, p]
        x = np.concatenate([seq, bone], axis=-1)
    else:
        raise ValueError(features)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


FEATURE_CHANNELS = {"joint": 3, "joint_motion": 6, "joint_bone": 6}


class SkeletonDataset(Dataset):
    """Returns (view_1[, view_2 ...], label, index). label = -1 when unlabeled."""

    def __init__(self, clips: List[np.ndarray], labels: Optional[np.ndarray], indices: np.ndarray,
                 window: int = 64, views: Sequence[AugParams] = (NO_AUG,), features: str = "joint",
                 joint_set: str = "body19", hide_labels: bool = False, transform=None):
        self.clips, self.indices = clips, np.asarray(indices)
        self.labels = labels
        self.window, self.views, self.features = window, list(views), features
        self.parents = parent_indices(joint_set)
        self.hide_labels = hide_labels
        self.transform = transform  # deterministic per-clip transform applied before augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        # torch's per-worker RNG is derived from torch.manual_seed, so this stays reproducible
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        seq = self.clips[idx]
        if self.transform is not None:
            seq = self.transform(seq)
        out = []
        for p in self.views:
            v = augment(seq, p, rng) if p is not NO_AUG else seq
            out.append(torch.from_numpy(to_features(resize_time(v, self.window), self.features, self.parents)))
        y = -1 if (self.labels is None or self.hide_labels) else int(self.labels[idx])
        return (*out, y, idx)


def worker_init_fn(worker_id: int):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
