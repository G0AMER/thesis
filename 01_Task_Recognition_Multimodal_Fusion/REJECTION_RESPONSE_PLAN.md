# CoopIS 2026 Paper #52 — Rejection Response & Rebuild Plan

**Paper:** *Task Recognition from Physiological Data Using Multimodal Sensor Fusion For Human-Robot Collaboration*
**Status of this document:** Working plan. Maps every reviewer comment to a verified defect and a concrete fix, then defines the full rebuild.

> All file paths are relative to `01_Task_Recognition_Multimodal_Fusion/`.

---

## 1. Verified technical defects (found by inspecting the actual data & code)

These are the root causes behind the reviews. Each was confirmed against the checked-in artifacts, not guessed.

### D1 — The 80/20 split is a *row-level* (window) split, not subject-disjoint → leakage risk
[src/08_classic_split_benchmark.py](src/08_classic_split_benchmark.py) (`train_test_split(..., stratify=y)` on the 5,640 window rows, line ~148). Repeated windows of the *same subject and same task* appear in both train and test, so the model memorizes subject/task-specific signal. This is exactly Reviewer 3's "row-level split of repeated observations would create leakage risk."

### D2 — `ID__*` and `Repetition__*` are used as model features (metadata leakage)
The fusion dataset contains 16 columns `ID__mean/std/min/max/median/p25/p75/energy` and `Repetition__*` (verified). `ID` and `Repetition` are recording **identifiers / trial counters**, not physiology. The LOSO harness excludes `subject_id` etc. but **does not** exclude these 16 (`final_detector_loso_10models.py` exclude set, lines 67–71). `Repetition` correlates with trial order → inflates accuracy and can even leak label structure.

### D3 — EEG features are duplicated → EEG double-weighted
96 EEG columns exist **twice**: as `EEG_channel_N__stat` and as identical copies `eeg_features_5s__EEG_channel_N__stat` (verified byte-identical, correlation = 1.0). The "242 features" are really ~128 unique (96 EEG + 32 physio) + 16 leak + 96 duplicate. Every model silently received 2× EEG weight.

### D4 — Labels are NOT dataset ground truth; they are a keyword rule on `task_name`
The dataset's real annotation tables (`labels.csv`, `bio_features_60s.csv`, STAI-Y1 / NASA-TLX / SAM questionnaires) live in the dataset's `features/` directory, which was never obtained (`01_data_exploration.ipynb` logs `features root … exists: False`). Because labels were missing, `03_fusion_dataset_builder.ipynb` **invented** the five classes with `create_task_rule_label()` (string matching: `stroop/n-back/mat/hanoi→cognitive_load`, `vr-plank→high_stress`, `manual/cobot→industrial_task`, `rest/meditation→low_load`, else `other`). The paper never says this. This is Reviewer 1's "not clear how task types have been identified", Reviewer 4's "How is MultiPhysio-HRC mapped to the 5 task-types?", and Reviewer 3's "no reproducible mapping."

### D5 — The reported numbers do not match the artifacts, and are not reproducible
- Reported XGBoost 0.9371 acc / 0.9142 macro-F1 match the tuned row-level run — an **upper bound inflated by D1+D2**.
- A subject-disjoint LOSO already run in this repo (`final_detector_loso_10models/benchmark_summary_ranked.csv`) shows the honest picture: best macro-F1 ≈ **0.56** (hist-gradient-boosting, 52 folds), i.e. ≈0.36 lower than the headline.
- The Two-Tower row in Table 2 (0.8874/0.8908/0.8635) matches **no** checked-in CSV (checked-in value is 0.8839/0.8691/0.8397). MLP has identical Balanced-Acc and Macro-F1 (0.8727 = 0.8727), flagged by Reviewer 4.
- No hyperparameters, seed/tuning details, feature dimensionality `d`, per-fold files, class counts, or missingness are given → Reviewers 1, 3, 4 all say "cannot be reproduced / no code."

### D6 — Figure 1 shows models the paper no longer describes
`assets/task_pipeline.png` still contains a **sequence-encoding / BiLSTM path**, but the LaTeX removed all sequence-model text (comment: "Sequence model acronyms removed"). Figure–text mismatch (Reviewer 3.7).

### D7 — Broken / wrong references
- `brodersen2010biostatistics` is actually *Collot et al., "Balanced accuracy: the right metric for evaluating LLM judges" (2026, arXiv)* — an LLM-judge paper cited as the source of balanced accuracy (Reviewers 3 & 4). The real source is Brodersen et al. 2010 (ICPR).
- `jeni2023imbalanced` has bib-year 2023 but the entry is ACII **2013**.
- Related work leans on continual-learning refs (`delange2022`, `fan2024continualhrc`, `auddy2023`) whose relevance is never argued (Reviewer 4: "why are [4] and [5] here?"). Nothing grounds the paper in *multimodal physiological task/stress/cognitive-load recognition* (the actual field).
- "Recent work" claimed for refs 7, 5, 2 years old (Reviewer 1).

### D8 — Internal inconsistencies & presentation
- **Two sections titled "Evaluation"** (§3.6 and §5) — Reviewer 2, Reviewer 3.8.
- **Abstract lists only 3 of 5 classes** and contains a duplicated sentence plus a leftover French comment.
- Equations (1)–(2) are never explained (variables not defined; motivation missing) — Reviewers 1 & 4.
- Intro promises handling of "differences between operators, changes in sensor relationships over time, and imbalanced label distributions," but only the imbalance term exists (Eq. 1) — Reviewer 4.

### D9 — No evidence the contribution is a contribution
Reviewer 2: "position paper … difficult to understand what is actually new." The pipeline is preprocessing + concat-fusion + off-the-shelf classifiers; no ablation shows fusion helps; no comparison to the source paper's protocol (Reviewer 3: MultiPhysio-HRC reports Low/Med/High stress & cognitive-load under LOSO with F1 0.34–0.39 / 0.33–0.49 — ours is not comparable). Reviewer 1: no real-world application/impact/latency discussion. Reviewer 3.4: offline study → narrow HRC claims.

---

## 2. Decision (from authors)
- **Label schema:** keep the five-class schema, but make the mapping **explicit, defensible, and reproducible** (task-condition schema). Real questionnaire labels are not available locally (verified) — getting them is a stretch goal that would unlock stress/cognitive-load regression to the dataset's own targets.
- **Scope:** full rebuild of the evaluation (subject-disjoint LOSO, clean features, ablations, statistics).
- **Target:** a full paper for another conference.

---

## 3. Fix plan (maps 1:1 to reviewer comments)

### 3.1 Data & labels
| Action | Files | Reviewers |
|---|---|---|
| Publish the **task-condition label schema**: define each of the 5 labels as a *protocol condition*, give the full `task_name → label` mapping table and the rule function verbatim. | new `paper/` tables; cite rule in Methods | R1.1, R4.2, R3.1 |
| Make classes self-contained: rename/situate `industrial_task` (physical manual + collaborative disassembly conditions, distinct from cognitive/affective states); decide `other` (only `vr-job-sim_0`, 71 windows) — keep and document, or fold into a principled bucket. | Methods, §class definitions | R4.1 |
| **Stretch:** obtain the official `features/` dir (labels.csv + STAI/NASA). Then add a comparability bridge: predict stress & cognitive load as Low/Med/High under the dataset's protocol and compare to Bussolan et al.'s published F1 ranges. Without it, be explicit that a direct comparison is out of scope because those annotations are absent. | new ablations | R3.1, R3.8 |

### 3.2 Evaluation protocol (the core rebuild)
| Action | Files | Reviewers |
|---|---|---|
| **Subject-disjoint LOSO** over all 52 subjects (not the 5-subject subset used in `v1_loso_two_tower`). Imputer + scaler fitted **inside each training fold only**. | new `rebuild/loso_benchmark.py` | R3.2 |
| **Drop the 16 `ID__*`/`Repetition__*` leak columns** and the 96 duplicated `eeg_features_5s__EEG_channel_*` columns → clean feature set (~128), with a machine-readable feature manifest giving `feature → group (EEG/ECG/EDA/EMG/RESP) → kept/reason`. | `rebuild/build_clean_dataset.py` | R3.5 |
| Report per-fold scores for every model, plus mean ± std, and **macro-F1 with 95% CIs** over folds; aggregate the LOSO confusion matrix (sum of fold counts) with raw counts (not only row-normalized). | loso script + tables | R3.5, R3.6, R1.5 |
| **Paired significance:** Wilcoxon signed-rank / paired bootstrap across the 52 folds comparing every model to the best (with effect size). | loso script → significance table/fig | R3.6, R1.5 |
| Keep the **classical row-level split result but relabel it** as a within-subject upper bound / sensitivity analysis — never as the headline. | Results § | R3.2 |
| **Repeated-run stability** for stochastic models (e.g., 3 seeds) to show intervals. | loso script | R3.6 |

### 3.3 Ablations (does fusion help?)
| Action | Files | Reviewers |
|---|---|---|
| **Single-modality baselines:** EEG-only, physio-only (ECG+EDA+EMG+RESP), each under the same LOSO protocol, for the leading models (XGBoost/HGB/RF/MLP). Report a fusion-benefit table and a paired test **fused vs. best single modality**. | `rebuild/ablations.py` | R3.3, R4.5 |
| Report **calibration** (e.g., expected calibration error) for the best detector + the two-tower concat model as the "no-fusion vs fusion-architecture" contrast. | loso script | R3.6 |

### 3.4 Contribution & novelty (Reviewer 2)
Reframe the contribution from "a compact pipeline" to something concrete the evidence supports, e.g.:
1. The **first leakage-free, subject-disjoint benchmark** of task-condition recognition on MultiPhysio-HRC across classical + neural classifiers, with a released feature-cleaning protocol.
2. A **modality-contribution analysis** (EEG vs peripheral physiology vs fusion) under a controlled protocol — the thing the source paper does not provide.
3. Clear **generalization ceiling vs within-subject ceiling** quantification for HRC deployment.

Add an explicit "Contributions" list after the intro. Also describe real-world integration: inference latency, which sensors are wearable, failure/fallback policy (→ narrows the HRC claim the way R3.4 asks), privacy & workplace use of physiological sensing (R3.9).

### 3.5 Reproducibility package
Release the exact scripts used (point to this repo / a DOI), the cleaned feature manifest, hyperparameters of every model, seeds, tuning procedure, missingness stats, per-class/per-fold CSVs, software versions. This directly answers R1.4 / R4.6 / R3.5.

### 3.6 Paper fixes (cosmetic but required)
| Defect | Fix |
|---|---|
| Two "Evaluation" sections | Rename §3.6 → "Evaluation protocol / metrics"; §5 → "Results & error analysis" |
| Fig. 1 shows BiLSTM/sequence path, text doesn't | Regenerate pipeline figure to exactly the models run |
| Abstract lists 3/5 classes; duplicated sentence; leftover French | Rewrite abstract; list all 5 labels; single clean sentence |
| Eq. (1)–(2) unexplained | Define every symbol; motivate class weights; state Eq. (2) argmax rule; say why accuracy alone is insufficient |
| Wrong/moot refs | Replace `brodersen2010` with the true Brodersen et al. 2010; fix `jeni` year; add real physiological task/stress/cognitive-load + multimodal fusion refs from 2022–2026; cut or justify continual-learning refs |
| "recent" wording | Cite actual years |
| MLP identical Balanced-Acc/Macro-F1 | Will resolve in rebuild (new numbers) |
| Table values matching no artifact | All numbers regenerated from one sealed run; state the exact artifact each table comes from |

### 3.7 Results framing
Lead with the **subject-disjoint LOSO** headline (expected honest macro-F1 ≈ 0.5–0.6, i.e., task-condition recognition generalizing across operators is genuinely hard — that *is* the finding), report the within-subject ceiling separately, and state plainly why the source protocol's F1 (0.34–0.49 on stress/cognitive-load) is not directly comparable to a task-condition schema without the questionnaire labels.

---

## 4. Execution order (what I will build/run)
1. `rebuild/build_clean_dataset.py` → cleaned dataset + feature manifest + split blueprint (subject/session/task/window keys preserved).
2. `rebuild/loso_benchmark.py` → full 52-subject LOSO, 11 classifiers, per-fold + aggregate metrics, CIs, paired significance, aggregated confusion (raw + normalized), calibration.
3. `rebuild/ablations.py` → EEG-only / physio-only / fused for the leading models; fusion-benefit table + significance.
4. `rebuild/make_figures.py` → subject-level macro-F1 boxplot, per-class metrics, aggregated confusion, ablation bars, significance heatmap, latency table.
5. Rewrite `paper/task_type_detection_paper.tex` (full paper) incorporating everything above; compile to PDF.
