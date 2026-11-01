"""Joint subsets, graph construction and per-clip canonicalisation.

The legacy 17-joint map assigned BVH `RightArm` (the shoulder joint) to "elbow" and
`RightForeArm` (the elbow) to "wrist", so the hands were never part of the input. In a
Perception Neuron hierarchy the joint *position* of `XArm` is the shoulder, `XForeArm` the
elbow and `XHand` the wrist. The sets below use that convention.
"""
from typing import Dict, List, Sequence

import numpy as np

# name -> parent name (None for the root). Order defines the joint index.
JOINT_SETS: Dict[str, List[tuple]] = {
    "body19": [
        ("Hips", None),
        ("Spine1", "Hips"),
        ("Spine3", "Spine1"),
        ("Neck", "Spine3"),
        ("Head", "Neck"),
        ("RightArm", "Spine3"),          # right shoulder
        ("RightForeArm", "RightArm"),    # right elbow
        ("RightHand", "RightForeArm"),   # right wrist
        ("RightHandMiddle1", "RightHand"),  # right knuckle (hand orientation)
        ("LeftArm", "Spine3"),
        ("LeftForeArm", "LeftArm"),
        ("LeftHand", "LeftForeArm"),
        ("LeftHandMiddle1", "LeftHand"),
        ("RightUpLeg", "Hips"),          # right hip
        ("RightLeg", "RightUpLeg"),      # right knee
        ("RightFoot", "RightLeg"),       # right ankle
        ("LeftUpLeg", "Hips"),
        ("LeftLeg", "LeftUpLeg"),
        ("LeftFoot", "LeftLeg"),
    ],
}


def joint_names(joint_set: str) -> List[str]:
    return [n for n, _ in JOINT_SETS[joint_set]]


def parent_indices(joint_set: str) -> List[int]:
    names = joint_names(joint_set)
    return [-1 if p is None else names.index(p) for _, p in JOINT_SETS[joint_set]]


def selection_indices(all_names: Sequence[str], joint_set: str) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(all_names)}
    missing = [n for n in joint_names(joint_set) if n not in lookup]
    if missing:
        raise KeyError(f"Joints {missing} not found in BVH hierarchy")
    return np.array([lookup[n] for n in joint_names(joint_set)])


def _edge2mat(links, V):
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in links:
        A[j, i] = 1.0
    return A


def _normalize_digraph(A):
    deg = A.sum(0)
    Dn = np.zeros_like(A)
    nz = deg > 0
    Dn[nz, nz] = 1.0 / deg[nz]
    return A @ Dn


def spatial_graph(joint_set: str) -> np.ndarray:
    """CTR-GCN style 3-subset graph (self, inward, outward), shape (3, V, V)."""
    parents = parent_indices(joint_set)
    V = len(parents)
    inward = [(i, p) for i, p in enumerate(parents) if p >= 0]
    outward = [(p, i) for i, p in inward]
    I = np.eye(V, dtype=np.float32)
    return np.stack([I, _normalize_digraph(_edge2mat(inward, V)), _normalize_digraph(_edge2mat(outward, V))])


def yaw_rotation(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def facing_yaw(frame: np.ndarray, names: Sequence[str]) -> float:
    """Yaw (rad, around +Y) of the hip line RightUpLeg->LeftUpLeg in the horizontal plane."""
    v = frame[names.index("LeftUpLeg")] - frame[names.index("RightUpLeg")]
    return float(np.arctan2(v[2], v[0]))


def canonicalize(seq: np.ndarray, names: Sequence[str], center: str = "first_frame",
                 yaw: str = "first_frame", scale: str = "torso") -> np.ndarray:
    """seq: (T, V, 3) world coordinates (Y up). Returns a float32 copy.

    center: 'first_frame' subtracts the Hips of frame 0 (keeps within-clip translation, which
            matters for e.g. "Picking left"), 'per_frame' subtracts Hips at every frame, 'none'.
    yaw:    'first_frame' rotates about Y so the frame-0 hip line points along +X, 'none'.
            The two workstations face different world directions, so without this the
            network can separate domains by absolute heading alone.
    scale:  'torso' divides by the clip-mean Hips->Neck distance, 'none'.
    """
    seq = np.asarray(seq, dtype=np.float64).copy()
    h = names.index("Hips")
    if center == "first_frame":
        seq -= seq[0, h]
    elif center == "per_frame":
        seq -= seq[:, h:h + 1]
    elif center != "none":
        raise ValueError(center)
    if yaw == "first_frame":
        R = yaw_rotation(facing_yaw(seq[0], names))
        seq = seq @ R.T
    elif yaw != "none":
        raise ValueError(yaw)
    if scale == "torso":
        torso = np.linalg.norm(seq[:, names.index("Neck")] - seq[:, h], axis=-1).mean()
        seq /= max(torso, 1e-6)
    elif scale != "none":
        raise ValueError(scale)
    return seq.astype(np.float32)
