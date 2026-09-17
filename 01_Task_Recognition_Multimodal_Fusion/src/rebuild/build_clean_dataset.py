#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_rebuild — Stage 1: build the CLEAN dataset for the rejection-response rebuild.

Fixes applied (see REJECTION_RESPONSE_PLAN.md):
  D2: drop 16 metadata-leak columns  ID__*, Repetition__*
  D3: drop 96 duplicated EEG columns  eeg_features_5s__EEG_channel_* (byte-identical
      to the canonical EEG_channel_* set)
  D4: preserve task_name + the documented task->label rule; export the schema table
Keeps subject/session/task/window identifiers so any downstream split is group-aware.

Outputs (under output/research_outputs/fusion_training/v3_rebuild/):
  dataset_clean.parquet      cleaned feature matrix + keys
  feature_manifest.csv       every source column -> modality group -> kept/dropped reason
  task_label_schema.csv      task_name -> label rule (reproducibility of D4)
  data_audit.json            counts, class balance, missingness, modality sizes
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FUSION = REPO / "output" / "research_outputs" / "fusion" / "v1_fusion" / "fusion_dataset.csv"
OUT = REPO / "output" / "research_outputs" / "fusion_training" / "v3_rebuild"

# --- identifier columns that must never enter the feature matrix -----------------
KEY_COLS_ALL = ["subject_id", "task_name", "task_file", "split", "window_idx",
                "start_idx", "end_idx", "n_samples", "label", "pseudo_label"]

# --- leak features: recording identifiers / trial counters (D2) ------------------
def is_id_repetition_leak(c: str) -> bool:
    return c.split("__", 1)[0] in {"ID", "Repetition"}

# --- duplicated EEG export (D3) --------------------------------------------------
DUPE_PREFIX = "eeg_features_5s__EEG_channel_"

# --- label rule verbatim from 03_fusion_dataset_builder.ipynb (D4) ---------------
def task_rule_label(task_name: str) -> str:
    name = str(task_name).lower()
    if any(x in name for x in ["stroop", "n-back", "mat", "hanoi"]):
        return "cognitive_load"
    if any(x in name for x in ["vr-plank"]):
        return "high_stress"
    if any(x in name for x in ["manual", "cobot"]):
        return "industrial_task"
    if any(x in name for x in ["rest", "meditation"]):
        return "low_load"
    return "other"

MODALITY = {
    "EEG": lambda c: re.match(r"EEG_channel_\d+__", c) is not None,
    "ECG": lambda c: c.startswith("ECG__"),
    "EDA": lambda c: c.startswith("EDA__"),
    "EMG": lambda c: c.startswith("EMG__"),
    "RESP": lambda c: c.startswith("RESP__"),
}
PHYSIO_GROUP = {"ECG", "EDA", "EMG", "RESP"}


def modality_of(c: str) -> str:
    for name, fn in MODALITY.items():
        if fn(c):
            return name
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default=str(FUSION))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.fusion, low_memory=False)
    KEY_COLS = [c for c in KEY_COLS_ALL if c in df.columns]

    # 1) manifest of every column ---------------------------------------------
    rows = []
    for c in df.columns:
        if c in KEY_COLS:
            rows.append({"column": c, "group": "meta", "decision": "keep_as_key", "reason": "identifier / label key"})
            continue
        if is_id_repetition_leak(c):
            rows.append({"column": c, "group": "leak", "decision": "drop", "reason": "recording ID/trial-counter metadata (D2)"})
            continue
        if c.startswith(DUPE_PREFIX):
            short = c[len("eeg_features_5s__"):]
            rows.append({"column": c, "group": "EEG", "decision": "drop_duplicate",
                         "reason": f"byte-identical duplicate of {short} (D3)"})
            continue
        g = modality_of(c)
        if g == "other":
            rows.append({"column": c, "group": "other", "decision": "drop",
                         "reason": "dataset metadata / presence flag, not a physiological channel"})
            continue
        rows.append({"column": c, "group": g, "decision": "keep",
                     "reason": "physiological feature"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "feature_manifest.csv", index=False)

    keep_cols = [r["column"] for r in rows if r["decision"] == "keep"]
    group_of = {r["column"]: r["group"] for r in rows if r["decision"] == "keep"}

    # 2) numeric coercion + missingness audit -----------------------------------
    X = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    missing = int(X.isna().sum().sum())
    n_rows, n_feats = X.shape

    # 3) clean frame with keys --------------------------------------------------
    clean = pd.concat([df[KEY_COLS], X], axis=1)
    clean.to_parquet(out / "dataset_clean.parquet", index=False)

    # 4) label schema (D4): reconstruct the rule output from task_name ----------
    schema = (df[["subject_id", "task_name", "pseudo_label"]]
                .drop_duplicates("task_name")[["task_name", "pseudo_label"]]
                .rename(columns={"pseudo_label": "label_by_rule"})
                .sort_values(["label_by_rule", "task_name"]))
    schema.to_csv(out / "task_label_schema.csv", index=False)

    # 5) audit -----------------------------------------------------------------
    groups = pd.Series([group_of[c] for c in keep_cols]).value_counts().to_dict()
    audit = {
        "source_rows": int(len(df)),
        "source_cols": int(df.shape[1]),
        "key_cols": KEY_COLS,
        "dropped_leak_cols": int(((manifest.decision == "drop") & (manifest.group == "leak")).sum()),
        "dropped_dup_eeg_cols": int((manifest.decision == "drop_duplicate").sum()),
        "kept_feature_cols": int(n_feats),
        "kept_feature_cols_by_group": {k: int(v) for k, v in sorted(groups.items())},
        "missing_values_in_features": missing,
        "rows": int(n_rows),
        "subjects": int(df.subject_id.nunique()),
        "tasks": int(df.task_name.nunique()),
        "class_counts": {k: int(v) for k, v in df.pseudo_label.value_counts().items()},
        "outputs": {
            "dataset_clean": str(out / "dataset_clean.parquet"),
            "feature_manifest": str(out / "feature_manifest.csv"),
            "task_label_schema": str(out / "task_label_schema.csv"),
        },
    }
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))
    print(f"\nschema:\n{schema.to_string(index=False)}")


if __name__ == "__main__":
    main()
