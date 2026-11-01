# Digital-twin to real transfer for industrial action recognition (WP4)

**Author:** Ameur Gargouri
**Status (2026-10-16):** real-data preprocessing, the three-seed main benchmark, and
the complete few-shot comparison grid are finished.

The current files under `output_v2/benchmark_ctrgcn/` contain four subject-disjoint folds
for seeds 0, 1 and 2 for the main methods (`dt_in_domain`, `source_only`, `dann`, `coral`,
`mmd`, `picda`, `fixmatch`, `oracle`, and `source_plus_target`). All seven few-shot
initialisations, five label budgets, four folds, and three seeds are also complete, giving
12 runs for every few-shot configuration.

The current real-data benchmark does not reproduce the legacy PI-CDA claims. The current
three-seed summary reports source-only Macro-F1 ≈ 40.6%, PI-CDA ≈ 39.3%, and the target
oracle ≈ 75.2%. The legacy 80.82% PI-CDA result remains archived and must not be used in
the paper.

The first version of this work (PI-CDA, archived in `archive/legacy_synthetic_2026-09/`) reported results that were produced on synthetic sine-wave data and random numbers, not on the datasets. None of its numbers, figures or claims may be reused. The archive README lists the problems.

## Question

Can a skeleton action recogniser trained on the InHARD-DT virtual-reality digital twin be deployed on the physical InHARD workstation without (or with very few) real labels, and does the twin add value when real labels exist?

## Data

| Dataset | Role | Capture | Content |
|---|---|---|---|
| InHARD-DT ([Zenodo 7644247](https://zenodo.org/records/7644247)) | source | Perception Neuron suit in a VR replica of the workstation | 12 subjects, 26 sessions, 4,696 segmented clips |
| InHARD ([Zenodo 4003541](https://zenodo.org/records/4003541)) | target | same Perception Neuron suit at the physical workstation | 16 subjects, ~4,800 segmented clips (No action excluded) |

Both share the same 59-joint BVH hierarchy at ~119 Hz, so the gap is behavioural and contextual (VR without physical parts vs. real assembly, different people, different workstation heading), not a sensor gap. Both use the same 13 meso-level classes (Assemble system, Consult sheets, Picking in front, Picking left, Put down/Take component, measuring rod, screwdriver, subsystem, Turn sheets).

## Pipeline

```
picda/                  library (all logic lives here)
  bvh.py                vectorised BVH reader + forward kinematics
  skeleton.py           19-joint body set incl. wrists/knuckles, graph, canonicalisation
  preprocess.py         raw BVH -> data/processed_v2/*.npz + index.csv
  data.py               loading, subject-disjoint splits, augmentation, datasets
  models.py             CTR-GCN (faithful to Chen et al. 2021), ST-GCN
  losses.py             MMD, CORAL, DANN, topology consistency
  train.py              supervised, UDA and FixMatch loops (fixed length, no target-label model selection)
  experiments.py        benchmark runner, one JSON of raw predictions per run
  analysis.py           tables and paired statistics from saved predictions
  gap_analysis.py       kinematic descriptors of the VR-vs-real gap
notebooks/
  01_Preprocess_Real_Data.ipynb   parse, validate, index, splits
  02_Domain_Gap_Analysis.ipynb    what differs between domains, handcrafted baselines
  03_Benchmark.ipynb              smoke test, full benchmark, ablations
  04_Results_Analysis.ipynb       tables, Wilcoxon/bootstrap, figures, LaTeX
scripts/run_benchmark.py          same runner from a terminal
```

Run the notebooks in order. Notebook 03 is resumable.

## Evaluation protocol

InHARD subjects are split into 4 subject-disjoint folds. Per fold, 12 subjects are the target training pool (unlabeled for UDA) and 4 are test subjects never seen in training. The source is all of InHARD-DT. Training length is fixed and the final model is evaluated; target labels are never used for model selection. Results are reported as mean ± std over 4 folds × 3 seeds, with Wilcoxon tests over the 16 target subjects and paired clip bootstraps.

The published InHARD-DT study introduced the digital-twin data-generation setting and
reported an ST-GCN cross-domain baseline: Dallel et al., “Digital twin of an industrial
workstation: A novel method of an auto-labeled data generator using virtual reality for
human action recognition in the context of human--robot collaboration,” *Engineering
Applications of Artificial Intelligence*, 118, 105655 (2023),
doi:10.1016/j.engappai.2022.105655. The current paper cites this work and reports a
strict subject-disjoint CTR-GCN re-evaluation; its numerical results should not be
treated as directly comparable unless the fold, backbone, and training protocol match.
