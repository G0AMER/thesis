"""Convert the segmented BVH clips of both datasets into one npz per clip plus an index CSV.

Each npz stores the full 59-joint world-coordinate skeleton (cm, Y-up) resampled to 30 Hz.
Joint selection and canonicalisation happen at load time (picda.data), so they can be
ablated without re-parsing.
"""
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config as C
from .bvh import forward_kinematics, read_bvh, resample

DT_PATTERN = re.compile(r"^A(\d{2})_((LMVR_\d+)_([A-Za-z\-]+)_.+?)_(\d+\.\d+)_(\d+\.\d+)$")
REAL_PATTERN = re.compile(r"^(P\d{2})_(R\d{2})_(\d+\.\d+)_(\d+\.\d+)$")


def parse_clip_name(domain: str, path: Path) -> dict:
    stem, folder = path.stem, path.parent.name
    if folder.lower() in C.EXCLUDED_CLASSES:
        return {"skip": f"excluded class '{folder}'"}
    if folder.lower() not in C.CLASS_TO_IDX:
        return {"skip": f"unknown class folder '{folder}'"}
    label = C.CLASS_TO_IDX[folder.lower()]
    if domain == C.DOMAIN_DT:
        m = DT_PATTERN.match(stem)
        if not m:
            return {"skip": f"unparsed DT name '{stem}'"}
        if int(m.group(1)) != label:
            return {"skip": f"prefix A{m.group(1)} disagrees with folder '{folder}'"}
        return dict(label=label, subject=m.group(4), session=m.group(2),
                    start_s=float(m.group(5)), end_s=float(m.group(6)))
    m = REAL_PATTERN.match(stem)
    if not m:
        return {"skip": f"unparsed InHARD name '{stem}'"}
    return dict(label=label, subject=m.group(1), session=f"{m.group(1)}_{m.group(2)}",
                start_s=float(m.group(3)), end_s=float(m.group(4)))


def process_domain(domain: str, raw_dir: Path, out_dir: Path, overwrite: bool = False,
                   limit: Optional[int] = None):
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory missing: {raw_dir}")
    files = sorted(raw_dir.rglob("*.bvh"))
    if not files:
        raise FileNotFoundError(f"No BVH files under {raw_dir}")
    if limit:
        files = files[:limit]

    rows, names_ref = [], None
    for path in tqdm(files, desc=f"{domain}"):
        meta = parse_clip_name(domain, path)
        rel = Path(domain) / path.parent.name / (path.stem + ".npz")
        row = dict(clip_id=f"{domain}/{path.stem}", domain=domain, relpath=str(rel),
                   class_folder=path.parent.name)
        if "skip" in meta:
            rows.append({**row, "status": meta["skip"]})
            continue
        row.update(meta)
        out = out_dir / rel
        try:
            clip = read_bvh(path)
            if names_ref is None:
                names_ref = clip.names
            elif clip.names != names_ref:
                raise ValueError("joint hierarchy differs from the first file")
            if out.exists() and not overwrite:
                n30 = int(np.load(out)["joints"].shape[0])
            else:
                world = forward_kinematics(clip)
                j30 = resample(world, clip.fps, C.TARGET_FPS).astype(np.float32)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, joints=j30)
                n30 = j30.shape[0]
            row.update(status="ok", fps_raw=clip.fps, n_frames_raw=clip.motion.shape[0],
                       n_frames_declared=clip.declared_frames, n_frames=n30,
                       duration_s=(clip.motion.shape[0] - 1) / clip.fps)
        except Exception as e:  # keep going, but record why
            row["status"] = f"error: {e}"
        rows.append(row)
    return pd.DataFrame(rows), names_ref


def build_all(overwrite: bool = False, limit: Optional[int] = None) -> pd.DataFrame:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_dt, names_dt = process_domain(C.DOMAIN_DT, C.RAW_DT_DIR, C.PROCESSED_DIR, overwrite, limit)
    df_real, names_real = process_domain(C.DOMAIN_REAL, C.RAW_REAL_DIR, C.PROCESSED_DIR, overwrite, limit)
    if names_dt != names_real:
        raise ValueError("InHARD and InHARD-DT hierarchies differ; joint mapping must be revisited")
    index = pd.concat([df_dt, df_real], ignore_index=True)
    index.to_csv(C.INDEX_CSV, index=False)
    with open(C.JOINT_NAMES_JSON, "w") as f:
        json.dump(names_dt, f, indent=1)
    return index
