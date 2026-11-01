"""Minimal, vectorised BVH reader and forward kinematics.

Both InHARD and InHARD-DT are exported from a Perception Neuron suit with the same
59-joint hierarchy, 6 channels per joint (Xpos Ypos Zpos Yrot Xrot Zrot), Y-up, centimetres,
~119 Hz. Forward kinematics is vectorised over frames, so a clip parses in milliseconds
(bvhio's per-frame Python loop was the bottleneck of the legacy notebook).

Notebook 01 cross-checks this implementation against bvhio on a few files.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BVHClip:
    names: List[str]
    parents: List[int]
    offsets: np.ndarray        # (J, 3)
    channels: List[List[str]]  # per joint channel names, in file order
    frame_time: float
    motion: np.ndarray         # (N, total_channels)
    declared_frames: int

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time


def read_bvh(path) -> BVHClip:
    with open(path, "r") as f:
        text = f.read()
    if "MOTION" not in text:
        raise ValueError(f"No MOTION section in {path}")
    head, motion_txt = text.split("MOTION", 1)

    names, parents, offsets, channels = [], [], [], []
    stack: List[int] = []
    last = None  # index of the joint whose block opens next, -1 for an End Site
    for line in head.splitlines():
        tok = line.split()
        if not tok:
            continue
        key = tok[0]
        if key in ("ROOT", "JOINT"):
            parents.append(stack[-1] if stack else -1)
            names.append(tok[1])
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            last = len(names) - 1
        elif key == "End":
            last = -1
        elif key == "{":
            stack.append(last)
        elif key == "}":
            stack.pop()
        elif key == "OFFSET" and last is not None and last >= 0:
            offsets[last] = [float(v) for v in tok[1:4]]
        elif key == "CHANNELS" and last is not None and last >= 0:
            channels[last] = tok[2:2 + int(tok[1])]

    lines = motion_txt.strip().splitlines()
    declared = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    values = np.array(" ".join(lines[2:]).split(), dtype=np.float64)
    n_ch = sum(len(c) for c in channels)
    n_rows = values.size // n_ch
    if n_rows == 0:
        raise ValueError(f"No motion frames in {path}")
    motion = values[: n_rows * n_ch].reshape(n_rows, n_ch)
    return BVHClip(names, parents, np.asarray(offsets), channels, frame_time, motion, declared)


def _axis_rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    if axis == "X":
        R[..., 0, 0] = 1
        R[..., 1, 1], R[..., 1, 2] = c, -s
        R[..., 2, 1], R[..., 2, 2] = s, c
    elif axis == "Y":
        R[..., 0, 0], R[..., 0, 2] = c, s
        R[..., 1, 1] = 1
        R[..., 2, 0], R[..., 2, 2] = -s, c
    elif axis == "Z":
        R[..., 0, 0], R[..., 0, 1] = c, -s
        R[..., 1, 0], R[..., 1, 1] = s, c
        R[..., 2, 2] = 1
    else:
        raise ValueError(axis)
    return R


def forward_kinematics(clip: BVHClip) -> np.ndarray:
    """World joint positions, shape (N, J, 3), in the file's units (cm)."""
    N, J = clip.motion.shape[0], len(clip.names)
    pos = np.zeros((N, J, 3))
    rot = np.zeros((N, J, 3, 3))
    col = 0
    for j in range(J):
        chs = clip.channels[j]
        data = clip.motion[:, col:col + len(chs)]
        col += len(chs)
        t = np.tile(clip.offsets[j], (N, 1))
        R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        for k, ch in enumerate(chs):
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] = data[:, k]
            elif ch.endswith("rotation"):
                R = R @ _axis_rotation(ch[0], data[:, k])
        p = clip.parents[j]
        if p < 0:
            pos[:, j] = t
            rot[:, j] = R
        else:
            pos[:, j] = pos[:, p] + np.einsum("nij,nj->ni", rot[:, p], t)
            rot[:, j] = rot[:, p] @ R
    return pos


def resample(seq: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear-interpolation resampling along axis 0 of a (N, ...) array."""
    n = seq.shape[0]
    if n < 2 or abs(src_fps - dst_fps) < 1e-6:
        return seq.copy()
    duration = (n - 1) / src_fps
    t_dst = np.arange(0.0, duration + 1e-9, 1.0 / dst_fps)
    idx = t_dst * src_fps
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    w = (idx - i0).reshape(-1, *([1] * (seq.ndim - 1)))
    return (1 - w) * seq[i0] + w * seq[i1]
