"""Descriptive characterisation of the InHARD-DT (VR) vs InHARD (physical) gap.

Both datasets use the same Perception Neuron suit and skeleton, so the gap is not a sensor
gap. These clip-level kinematic descriptors test what actually differs: execution time,
heading, locomotion, hand speed and reach, posture and smoothness.
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import TARGET_FPS
from .skeleton import canonicalize, facing_yaw, joint_names

FPS = TARGET_FPS


def _speed(p):  # (T, 3) cm -> m/s per frame transition
    return np.linalg.norm(np.diff(p, axis=0), axis=-1) * FPS / 100.0


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _ldlj(p):
    """Log dimensionless jerk of a (T, 3) trajectory in m (Balasubramanian et al., 2015)."""
    if len(p) < 6:
        return np.nan
    dt = 1.0 / FPS
    v = np.gradient(p, dt, axis=0)
    j = np.gradient(np.gradient(v, dt, axis=0), dt, axis=0)
    dur = (len(p) - 1) * dt
    vpeak = np.linalg.norm(v, axis=-1).max()
    if vpeak < 1e-6:
        return np.nan
    integral = np.trapezoid((j ** 2).sum(-1), dx=dt)
    return -np.log(dur ** 3 / vpeak ** 2 * integral + 1e-12)


def _hf_ratio(p, cutoff_hz=4.0):
    """Share of wrist-velocity power above `cutoff_hz` (tremor / tracking jitter proxy)."""
    if len(p) < 16:
        return np.nan
    v = np.diff(p, axis=0) * FPS
    f, pxx = signal.welch(v, fs=FPS, axis=0, nperseg=min(64, len(v)))
    pxx = pxx.sum(-1)
    tot = pxx.sum()
    return float(pxx[f >= cutoff_hz].sum() / tot) if tot > 0 else np.nan


def _hf_power(p, cutoff_hz=4.0):
    """Absolute velocity power above `cutoff_hz` in (m/s)^2, summed over xyz. Unlike the
    ratio, this is not inflated when the low-frequency (voluntary) motion is small."""
    if len(p) < 16:
        return np.nan
    v = np.diff(p, axis=0) * FPS
    f, pxx = signal.welch(v, fs=FPS, axis=0, nperseg=min(64, len(v)))
    df = f[1] - f[0]
    return float(pxx[f >= cutoff_hz].sum() * df)


def clip_descriptors(raw: np.ndarray, joint_set: str = "body19") -> Dict[str, float]:
    """raw: (T, V, 3) world coordinates in cm at 30 Hz."""
    n = joint_names(joint_set)
    J = {k: n.index(k) for k in ("Hips", "Neck", "Head", "RightHand", "LeftHand", "RightArm",
                                 "RightForeArm", "LeftArm", "LeftForeArm")}
    can = canonicalize(raw, n, "first_frame", "first_frame", "torso")
    hips_xz = raw[:, J["Hips"]][:, [0, 2]] / 100.0
    rw, lw = raw[:, J["RightHand"]] / 100.0, raw[:, J["LeftHand"]] / 100.0
    rs, ls = _speed(raw[:, J["RightHand"]]), _speed(raw[:, J["LeftHand"]])
    trunk = raw[:, J["Neck"]] - raw[:, J["Hips"]]
    flex = np.degrees(np.arccos(np.clip(trunk[:, 1] / (np.linalg.norm(trunk, axis=-1) + 1e-8), -1, 1)))
    hand_can = can[:, [J["RightHand"], J["LeftHand"]]]
    r_el = _angle(raw[:, J["RightArm"]], raw[:, J["RightForeArm"]], raw[:, J["RightHand"]])
    l_el = _angle(raw[:, J["LeftArm"]], raw[:, J["LeftForeArm"]], raw[:, J["LeftHand"]])
    rpath, lpath = rs.sum() / FPS if len(rs) else 0.0, ls.sum() / FPS if len(ls) else 0.0
    return {
        "duration_s": (len(raw) - 1) / FPS,
        "facing_yaw_deg": float(np.degrees(facing_yaw(raw[0], n))),
        "root_displacement_m": float(np.linalg.norm(hips_xz[-1] - hips_xz[0])),
        "root_path_m": float(np.linalg.norm(np.diff(hips_xz, axis=0), axis=-1).sum()),
        "rwrist_mean_speed": float(rs.mean()) if len(rs) else np.nan,
        "rwrist_peak_speed": float(np.percentile(rs, 95)) if len(rs) else np.nan,
        "lwrist_mean_speed": float(ls.mean()) if len(ls) else np.nan,
        "lwrist_peak_speed": float(np.percentile(ls, 95)) if len(ls) else np.nan,
        "right_hand_share": float(rpath / (rpath + lpath + 1e-8)),
        "hand_height_torso": float(hand_can[..., 1].mean()),
        "hand_reach_torso": float(np.linalg.norm(hand_can[..., [0, 2]], axis=-1).max()),
        "trunk_flexion_deg": float(flex.mean()),
        "relbow_range_deg": float(np.ptp(r_el)),
        "lelbow_range_deg": float(np.ptp(l_el)),
        "rwrist_ldlj": float(_ldlj(rw)),
        "rwrist_hf_ratio": _hf_ratio(rw),
        "head_height_torso": float(can[:, J["Head"], 1].mean()),
    }


def descriptor_table(index: pd.DataFrame, raw_clips: Sequence[np.ndarray], joint_set="body19") -> pd.DataFrame:
    rows = [clip_descriptors(c, joint_set) for c in raw_clips]
    d = pd.DataFrame(rows)
    meta = index[["clip_id", "domain", "subject", "session", "label", "class_folder"]].reset_index(drop=True)
    return pd.concat([meta, d], axis=1)


def cliffs_delta_from_u(u, n1, n2):
    return 2.0 * u / (n1 * n2) - 1.0


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def domain_shift_tests(desc: pd.DataFrame, features: List[str], by_class: bool = True) -> pd.DataFrame:
    """Mann-Whitney U, DT vs real, per feature (and per class). Cliff's delta > 0 means DT larger."""
    groups = desc.groupby("class_folder") if by_class else [("all", desc)]
    rows = []
    for cls, g in groups:
        for f in features:
            a = g.loc[g.domain == "dt", f].dropna().values
            b = g.loc[g.domain == "real", f].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"class": cls, "feature": f, "dt_median": np.median(a), "real_median": np.median(b),
                         "cliffs_delta": cliffs_delta_from_u(u, len(a), len(b)), "p": p,
                         "n_dt": len(a), "n_real": len(b)})
    out = pd.DataFrame(rows)
    out["q_bh"] = bh_fdr(out["p"].values)
    return out
