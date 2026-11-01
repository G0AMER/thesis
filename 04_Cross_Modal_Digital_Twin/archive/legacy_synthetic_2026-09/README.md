# Legacy artefacts. Do not cite, reuse or present any number or figure from this folder.

Archived on 2026-10-11 after an audit found that:

* Notebooks 03 and 04 trained and evaluated on a sine-wave generator (40 samples per class), because `data/inhard/processed` contained no clips. Notebook 04 calls the generator directly. Every classification number in `pi_cda_paper.pdf` (80.82% macro-F1, oracle 71.69%, ablation, few-shot) comes from that toy data.
* Notebook 06 draws the per-subject LOSO scores, the Wilcoxon test inputs, the t-SNE embeddings and the confusion matrices from `np.random`, and hardcodes the anticipation curve.
* The class names in the paper (Hammering, Cleaning surface, ...) do not exist in InHARD.
* `latex_tables.tex` and the old README report a third set of numbers that no code produces.
* The 17-joint BVH mapping put the shoulder in the elbow slot and the elbow in the wrist slot, so hands were never part of the input.
* The only real-data experiment (anticipation, notebook 05) reached about 17-20% top-1. Its horizons are in clip steps, not seconds.

The replacement pipeline lives in `picda/` and `notebooks/01-04`.
