# Graph Report - thesis  (2026-10-12)

## Corpus Check
- Large corpus: 573 files · ~2,320,192 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1413 nodes · 2518 edges · 249 communities (50 shown, 199 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 155 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- FLAIR Notebook Pipeline
- DASIG Style Modeling
- Thesis Architecture & Literature
- DASIG Data Loader Tests
- Task Pipeline Visualization
- Continual Learner Algorithms
- Task Recognition LOSO Ablations
- FLAIR Reports & Presentations
- Incremental Learning Surveys
- PI-CDA Experiments & Losses
- CTR-GCN Model Architecture
- HARMONIC Preprocessing Pipeline
- Cobot Safety Paper Figures
- DASIG Data Loading
- ContinualLearner Base Class
- IL Benchmark Runner
- PI-CDA Benchmark Evaluation
- Cobot Safety Feature Extraction
- Reference Report Builder
- LOSO Detector Evaluation
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122
- Community 125
- Community 126
- Community 128
- Community 129
- Community 133
- Community 135
- Community 136
- Community 137
- Community 138
- Community 139
- Community 140
- Community 141
- Community 142
- Community 143
- Community 144
- Community 145
- Community 146
- Community 147
- Community 148
- Community 149
- Community 150
- Community 151
- Community 153
- Community 154
- Community 155
- Community 156
- Community 157
- Community 159
- Community 160
- Community 161
- Community 162
- Community 163
- Community 164
- Community 165
- Community 166
- Community 208
- Community 209
- Community 210
- Community 211
- Community 212
- Community 213
- Community 214
- Community 215
- Community 216
- Community 217
- Community 218
- Community 219
- Community 220
- Community 221
- Community 222
- Community 223
- Community 224
- Community 226
- Community 227
- Community 228
- Community 229
- Community 230
- Community 231
- Community 232
- Community 233
- Community 234
- Community 235
- Community 236
- Community 237
- Community 238
- Community 239
- Community 240
- Community 241
- Community 242
- Community 243
- Community 244
- Community 245
- Community 246
- Community 248

## God Nodes (most connected - your core abstractions)
1. `ContinualLearner` - 25 edges
2. `Benchmark` - 23 edges
3. `PreprocessConfig` - 17 edges
4. `run_dasig_pipeline()` - 16 edges
5. `discover_trials()` - 16 edges
6. `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels` - 16 edges
7. `main()` - 15 edges
8. `add_paper_details()` - 15 edges
9. `DASIGPreprocessConfig` - 15 edges
10. `TrainConfig` - 14 edges

## Surprising Connections (you probably didn't know these)
- `GPU-Accelerated Kinematic Feature Expansion (65 to 195 channels via derivatives)` --conceptually_related_to--> `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `FLAIR: A Combined Incremental Learning Algorithm for Human-Robot Collaboration` --references--> `HARMONIC Dataset: 24 subjects, EMG+IMU+gaze+joystick, shared autonomy`  [EXTRACTED]
  03_Incremental_Learning_FLAIR/paper/flair_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `Temporal Debouncing: median filter (k=3) + threshold 0.77 → 87.5% FP reduction` --conceptually_related_to--> `False-Alarm vs. Productivity Trade-off in Cobot Safety`  [EXTRACTED]
  00_Thesis_Dissertation_and_Reports/assets/Literature_Contributions.md → 02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf
- `Fig 2: DASIG Data Pipeline (60 subjects, 5 MIMUs, kinematic expansion 65->195 channels)` --references--> `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/assets/fig2_data_pipeline.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `Deep Learning for Safety-State Detection in HRC (KES 2026, Procedia CS)` --references--> `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md

## Import Cycles
- None detected.

## Communities (249 total, 199 thin omitted)

### Community 0 - "FLAIR Notebook Pipeline"
Cohesion: 0.05
Nodes (38): clean_source(), main(), make_cell_id(), all_joint_names(), augment(), AugParams, Canon, few_shot_indices() (+30 more)

### Community 1 - "DASIG Style Modeling"
Cohesion: 0.06
Nodes (26): main(), preprocess_recording(), run_dasig_pipeline(), save_preprocessed_recording(), butterworth_lowpass(), compute_frequency_features(), compute_jerk_features(), compute_subject_stats() (+18 more)

### Community 2 - "Thesis Architecture & Literature"
Cohesion: 0.05
Nodes (61): Five Novel Contributions: IMU-based Safety Pipeline, State-of-the-Art Cobot Safety Systems 2024-2025, PhD Thesis Codebase Analysis, PhD Thesis Architecture and Data Flows, Sensor Modalities for Shared Autonomy and HRC, Thesis Defense Presentation Slides (Marp), Novel Datasets and Research Proposals for HRC, Rapport de Synthèse Scientifique des Travaux de Thèse (+53 more)

### Community 3 - "DASIG Data Loader Tests"
Cohesion: 0.05
Nodes (20): dasig_tree(), _make_arduino_csv(), _make_mimu_csv(), TestConstants, TestDiscovery, TestLoading, TestSubjectsInfo, discover_recordings() (+12 more)

### Community 4 - "Task Pipeline Visualization"
Cohesion: 0.09
Nodes (30): arrow(), build_figure(), rbox(), best_model(), fig_ablation(), fig_confusion(), fig_gap(), fig_perclass() (+22 more)

### Community 5 - "Continual Learner Algorithms"
Cohesion: 0.05
Nodes (3): EWC, SI, ReplayBuffer

### Community 6 - "Task Recognition LOSO Ablations"
Cohesion: 0.07
Nodes (22): cols_of(), indices_of(), load(), loso_sklearn(), loso_twotower(), main(), TwoTower, compare() (+14 more)

### Community 7 - "FLAIR Reports & Presentations"
Cohesion: 0.07
Nodes (41): FLAIR KES 2026 Presentation, Dataset Exploration Report (HARMONIC and DASIG), DASIG Dataset, Preprocessing Pipeline Report (HARMONIC and DASIG), Incremental Learning Benchmark Report (Exp 1), Catastrophic Forgetting, Shared Autonomy Benchmark Report (Exp 01), FLAIR — 03 Incremental Learning README (+33 more)

### Community 8 - "Incremental Learning Surveys"
Cohesion: 0.06
Nodes (38): DER/DER++: Dark Experience Replay for incremental learning, Incremental Learning Algorithms in Autonomous Driving and Robot Lifelong Skill Acquisition, iCaRL: Incremental Classifier and Representation Learning, Real-World Robotics Datasets for Incremental Learning and Lifelong Learning, LIBERO: Lifelong Robot Learning Benchmark Suite, OpenLORIS-Object Dataset for lifelong robotic vision, MultiPhysio-HRC: Multimodal Physiological Signals Dataset for Industrial HRC, MultiPhysio-HRC Dataset (EEG, ECG, EDA, RESP, EMG, Voice, Facial features) (+30 more)

### Community 9 - "PI-CDA Experiments & Losses"
Cohesion: 0.11
Nodes (16): worker_init_fn(), coral_loss(), grad_reverse(), mmd_loss(), topology_consistency_loss(), build_model(), get_device(), infinite() (+8 more)

### Community 10 - "CTR-GCN Model Architecture"
Cohesion: 0.09
Nodes (10): bn_init(), conv_init(), CTRGC, CTRGCN, MultiScaleTemporalConv, STGCN, STGCNBlock, TCNGCNUnit (+2 more)

### Community 11 - "HARMONIC Preprocessing Pipeline"
Cohesion: 0.13
Nodes (12): main(), preprocess_trial(), align_timestamps(), compute_participant_stats(), normalize_dataframe(), preprocess_emg(), preprocess_gaze(), preprocess_imu() (+4 more)

### Community 12 - "Cobot Safety Paper Figures"
Cohesion: 0.10
Nodes (27): Fig 4: Neural Network Architectures for 1D Time-Series Classification (TCN, ConvNeXt 1D, MLP-Mixer 1D, Transformer 1D, InceptionTime), Fig 5: Evaluation Post-Processing Pipeline (Raw Probabilities -> Temporal Debouncing -> Threshold Decision SAFE/DANGER), Fig 5: Evaluation Post-Processing (copy), Fig 6: Model Performance Comparison Table (Threshold=0.77, Median Smoothed; TCN ranked 1st with Macro F1 97.88%), Fig 7: 5-Fold Stratified Cross-Validation Strategy (~55,275 train / ~13,819 validation windows per fold), Pipeline Fig 1: Overview of 4-Phase Deep Learning Pipeline (Data, Training, Evaluation, Deployment), Pipeline Fig 2: Data Pipeline showing Preprocessing & Windowing and Dynamic Kinematic Feature Expansion (65 -> 195 channels), Pipeline Fig 4: Neural Network Architectures for 1D Time-Series Classification (TCN, ConvNeXt 1D, MLP-Mixer 1D, Transformer 1D, InceptionTime) (+19 more)

### Community 13 - "DASIG Data Loading"
Cohesion: 0.11
Nodes (9): build_column_names(), generate_safety_labels(), load_all_subject_info(), load_all_trials(), load_arduino_csv(), load_mimu_csv(), load_subject_info(), load_trial() (+1 more)

### Community 14 - "ContinualLearner Base Class"
Cohesion: 0.13
Nodes (3): ContinualLearner, run_all_strategies(), run_benchmark()

### Community 15 - "IL Benchmark Runner"
Cohesion: 0.12
Nodes (6): BenchmarkResult, build_task_sequence(), load_participant_data(), results_to_dataframe(), save_benchmark_results(), TaskData

### Community 16 - "PI-CDA Benchmark Evaluation"
Cohesion: 0.28
Nodes (3): Benchmark, predict(), set_seed()

### Community 17 - "Cobot Safety Feature Extraction"
Cohesion: 0.16
Nodes (9): split_by_subject(), extract_features_bulk(), evaluate_model(), plot_feature_importance(), to_binary_labels(), train_gradient_boosting(), train_random_forest(), main() (+1 more)

### Community 18 - "Reference Report Builder"
Cohesion: 0.24
Nodes (16): build_seed_record(), clean_text(), extract_doi_from_url(), find_openalex_work(), format_authors(), get_crossref_metadata(), get_hrc_citing_papers(), get_json() (+8 more)

### Community 19 - "LOSO Detector Evaluation"
Cohesion: 0.15
Nodes (4): fold_dir(), fold_labels(), make_models(), run_loso_model()

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (8): _axis_rotation(), BVHClip, forward_kinematics(), read_bvh(), resample(), build_all(), parse_clip_name(), process_domain()

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (7): TrialData, extract_sensor_features(), extract_window_features(), normalize_features(), segment_all_trials(), sliding_window_segment(), _vector_norm()

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (3): extract_key_contribution(), get_binary(), get_pdf_text_from_url()

### Community 24 - "Community 24"
Cohesion: 0.19
Nodes (8): ece_from_1hot(), fold_metrics(), load_clean(), main(), per_class_recall(), t_ci95(), estimate_latency_us(), make_models()

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (5): get_data_columns(), get_timestamp_column(), harmonic_tree(), TestConstants, TestHelpers

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (8): add_hyperlink(), add_paper_details(), build_key_contribution(), build_method_summary(), build_results_summary(), build_work_summary(), extract_approach_details(), main()

### Community 27 - "Community 27"
Cohesion: 0.15
Nodes (5): discover_participants(), load_participant_trials(), load_trial(), TrialData, TrialInfo

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (5): BenchConfig, compute_class_weights(), reshape_to_sequences(), TCNNet, train_torch_model()

### Community 30 - "Community 30"
Cohesion: 0.15
Nodes (3): SequenceDataset, TabularDataset, TwoTowerDataset

### Community 33 - "Community 33"
Cohesion: 0.31
Nodes (3): discover_trials(), TestDiscovery, TestLoading

### Community 34 - "Community 34"
Cohesion: 0.19
Nodes (13): ATISP Laboratory, University of Sfax / ESME Research Lab, Danger State Detection, Embedded AI, FLAIR Continual Learning Method, Human-Robot Collaboration Research, Incremental Learning, Multimodal Sensor Fusion, Robot Safety (+5 more)

### Community 35 - "Community 35"
Cohesion: 0.21
Nodes (6): discover_pdf_url(), extract_data_details(), extract_full_paper_text(), extract_results_details(), get_arxiv_pdf_text(), get_text()

### Community 38 - "Community 38"
Cohesion: 0.36
Nodes (5): build_classifiers(), get_feature_columns(), main(), run_loso(), save_dict_csv()

### Community 39 - "Community 39"
Cohesion: 0.31
Nodes (9): PI-CDA Adjacency Matrix Figure, PI-CDA APA Transition Matrix Figure, PI-CDA Augmentation Demo Figure, PI-CDA: Physics-Informed Cross-Modal Domain Adaptation for Skeleton Action Recognition in Industrial Digital Twins, Contrastive Skeleton Topology Alignment (CSTA), Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN), InHARD-DT Dataset (Industrial Human Action Recognition with Digital Twin), PI-CDA (Physics-Informed Cross-Modal Domain Adaptation) (+1 more)

### Community 40 - "Community 40"
Cohesion: 0.25
Nodes (8): Exp03 Definitive Benchmark: Average MSE evolution over training sequence for Naive Fine-Tune, Joint Training, EWC, Online EWC, LwF, DER++, A-GEM across 10 tasks, Exp03 Definitive Benchmark: Comparison bar charts showing Avg MSE (lower=better) and Backward Transfer BWT for all methods — DER++ achieves 0.4341 MSE, Joint Training 0.3853, Exp03 Definitive Benchmark: Forgetting curve for Task 0 (p100) — MSE on Task 0 over training steps for all continual learning methods; DER++ shows least forgetting, Exp03 Definitive Benchmark: Joint forgetting correlation heatmap for DER++ — 6x6 correlation matrix across joints showing high cross-joint forgetting correlation (0.67–1.00), Exp03 Definitive Benchmark: Per-joint velocity MSE averaged across all tasks and per-joint backward transfer (forgetting) for all methods across 6 joints, Exp03 Definitive Benchmark: Per-joint forgetting curves for Task 0 (p100) — 6 subplots for J1 shoulder, J2 shoulder, J3 elbow, J4 wrist, J5 wrist, J6 wrist showing MSE evolution, Exp03 Definitive Benchmark: Per-participant per-joint MSE heatmaps comparing DER++ vs Naive Fine-Tune final performance across participants p100–p109 and 6 joints, Exp03 Definitive Benchmark: Final MSE per task after all training — grouped bar chart for all methods across participants p100–p109 including A-GEM

### Community 41 - "Community 41"
Cohesion: 0.29
Nodes (4): build_dataset_summary(), extract_dataset_details(), extract_dataset_profile(), extract_detail_phrases()

### Community 42 - "Community 42"
Cohesion: 0.43
Nodes (3): is_id_repetition_leak(), main(), modality_of()

### Community 44 - "Community 44"
Cohesion: 0.33
Nodes (6): IMU Sensor Placement Diagram (cobot_safety_cas): 5 IMU sensors on human body — Sternum (STR), Right/Left Upper Arm (RUA/LUA), Right/Left Front Arm (RFA/LFA) with coordinate frames, IMU Sensor Placement Diagram (cobot_safety_lra): 5 IMU sensors on human body — Sternum, RUA, LUA, RFA, LFA with global coordinate frame annotation, MIMU Placement Diagram (cobot_safety_lra): detailed IMU sensor body placement figure showing STR, RUA, LUA, RFA, LFA positions with 3-axis coordinate frame per sensor, IMU Sensor Placement Diagram (cobot_safety_robotica): 5 IMU sensors on human body — same layout as other paper versions, IMU Sensor Placement Diagram (cobot_safety_springer): 5 IMU sensors on human body — same layout as other paper versions, IMU Sensor Placement Diagram (cobot_safety_submission): 5 IMU sensors on human body — same layout as other paper versions

### Community 47 - "Community 47"
Cohesion: 0.40
Nodes (5): Confusion matrix for ConvNeXt 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 97.13%, danger false negative rate 2.87%, Confusion matrix for InceptionTime model on cobot danger state detection: SAFE recall 98.41%, DANGER recall 97.53%, danger false negative rate 2.47%, Confusion matrix for MLP-Mixer 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 95.88%, danger false negative rate 4.12%, Confusion matrix for TCN model on cobot danger state detection: SAFE recall 99.73%, DANGER recall 97.14%, danger false negative rate 2.86%, Confusion matrix for Transformer 1D model on cobot danger state detection: SAFE recall 96.56%, DANGER recall 95.85%, danger false negative rate 4.15%

### Community 48 - "Community 48"
Cohesion: 0.40
Nodes (5): ConvNeXt 1D Confusion Matrix (cobot_safety_cas): SAFE 99.76%, DANGER 97.13% recall — high-performance deep learning classifier, ConvNeXt 1D Confusion Matrix (cobot_safety_lra): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_robotica): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_springer): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_submission): SAFE 99.76%, DANGER 97.13% recall

### Community 50 - "Community 50"
Cohesion: 0.50
Nodes (4): XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load, high_stress, industrial_task, low_load, other) — assets copy, XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load 0.89, high_stress 0.91, industrial_task 0.99, low_load 0.75, other 0.93) — paper copy, XGBoost confusion matrix for 5-class task-condition recognition using 80/20 stratified split (cognitive_load 0.91, high_stress 0.89, industrial_task 1.00, low_load 0.82, other 0.79), Pooled LOSO confusion matrix for GradientBoosting classifier on 5-class task-condition recognition with raw window counts

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (4): R2 evolution chart (presentation version): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, no standalone EWC), R2 evolution chart (original): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, matches presentation version), R2 evolution chart (copy): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training, EWC (6 methods, different data range from original), R2 evolution chart (copy 2): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training, EWC (6 methods including standalone EWC)

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (4): Exp06 DER-SA: Ablation study bar chart comparing DER++ baseline with ARW-only, FKD-only, JAC-only, pairwise and full DER-SA variants on accuracy and forgetting, Exp06 DER-SA: All 6 metrics comparison — DER-SA vs baselines for ACC, Forgetting F, BWT, FWT, Memory Usage, Training Time showing DER-SA competitive with DER++ at much lower cost, Exp06 DER-SA: Per-joint MSE comparison DER++ vs DER-SA across 6 joints — DER-SA improves J1–J5 by 0.2–4.0%, small regression on J6, Exp06 DER-SA: R² evolution — forgetting trajectory on Task 0 (p100) and average accuracy evolution across tasks for Naive Fine-Tune, DER++, DER-SA, Joint Training

### Community 53 - "Community 53"
Cohesion: 0.50
Nodes (4): Exp07 CPG-Net: Ablation bar chart — No gating (EWC only), Random gates, No gate freezing, No importance reg., No gate sparsity vs full CPG-Net on accuracy and forgetting, Exp07 CPG-Net: All 6 metrics comparison — CPG-Net vs Joint Training, DER++, A-GEM, Online EWC, Naive Fine-Tune on ACC, Forgetting F, BWT, FWT, Memory Usage, Training Time, Exp07 CPG-Net: Per-feature ablation progression — Base, +Retro Boost, +Warm Start, +Adapt Replay, Full CPG-Net on ACC (0.6642→0.6895), Forgetting (0.0387→0.0168), BWT (-0.039→-0.017), Exp07 CPG-Net: Per-task R² after all training — Base vs Full CPG-Net for 10 tasks; Full CPG-Net improves most tasks by +0.001 to +0.056

### Community 55 - "Community 55"
Cohesion: 0.50
Nodes (4): PI-CDA Per-Class F1 Bar Chart (Legacy Synthetic) — InHARD-DT target, 13 action classes, macro-F1 80.8%, CT-R-GCN Confusion Matrices — side-by-side comparison of source_only, picda, and oracle conditions on InHARD 13-class action recognition, CT-R-GCN Few-Shot Learning Curve — macro-F1 vs labeled InHARD clips per class (1/5/10-shot) for fine-tune from picda, scratch, and source_only, CT-R-GCN t-SNE Domain Alignment — 2x3 grid showing by-domain and by-class embeddings for source_only, picda, and oracle conditions

### Community 57 - "Community 57"
Cohesion: 1.00
Nodes (3): FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — original, FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — variant copy, FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — variant copy 2

### Community 59 - "Community 59"
Cohesion: 0.67
Nodes (3): Full task-condition recognition pipeline diagram: MultiPhysio-HRC dataset -> Preprocessing -> EEG features (96-d) + Peripheral physio (32-d) -> Fused representation (128-d) -> Detectors -> Task-condition label, Simplified task-type detection pipeline diagram: Physiological Signals -> Preprocessing -> Feature/Sequence Encoding -> Classifier (XGBoost/BiLSTM/Fusion), Full task-condition recognition pipeline diagram (paper version): MultiPhysio-HRC -> Preprocessing -> EEG/Peripheral features -> Detectors -> Evaluation

### Community 65 - "Community 65"
Cohesion: 1.00
Nodes (3): 5-Fold CV bar chart comparing post-processed performance of TCN, CRNN+Attention, InceptionTime, U-Net 1D, Transformer on Macro F1 Score, Danger Recall, and Danger Precision; TCN leads Macro F1 (0.647), Transformer leads Danger Recall (0.789), 5-Fold CV bar chart for window-level classification with feature engineering comparing TCN, InceptionTime, ResNet1D, CRNN+Attention, LSTM-FCN on Accuracy, Macro F1, Danger Recall, and Danger Precision; TCN leads on Accuracy (0.898) and Macro F1 (0.719), Subject-Dependent Evaluation (Factory-Calibrated Robots) bar chart comparing TCN, ConvNeXt 1D, MLP-Mixer 1D, InceptionTime, Transformer 1D on Accuracy, Macro F1, Danger Recall, and Danger Precision; TCN leads (Accuracy 95.9%, Macro F1 90.5%, Danger Recall 82.8%)

### Community 66 - "Community 66"
Cohesion: 0.67
Nodes (3): Confusion matrix for Gradient Boosting classifier on cobot safety SAFE vs ABRUPT classification: 984 TP-safe, 4 FP-abrupt, 105 FN-safe, 59 TP-abrupt; poor abrupt movement recall, Normalized confusion matrix for Gradient Boosting on cobot safety SAFE vs ABRUPT: SAFE precision 99.60%, ABRUPT recall only 35.98%, showing severe class imbalance problem, Confusion matrix for Random Forest classifier on cobot safety SAFE vs ABRUPT classification: 987 TP-safe, 1 FP-abrupt, 121 FN-safe, 43 TP-abrupt; even lower abrupt recall than gradient boosting

### Community 68 - "Community 68"
Cohesion: 0.67
Nodes (3): CPG-Net R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks, FLAIR R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks vs baselines, FLAIR Paper Copy: R2 Evolution showing Forgetting Trajectory and Average Accuracy Evolution vs baselines

## Knowledge Gaps
- **257 isolated node(s):** `insert`, `update`, `insert`, `update`, `hrc-incremental` (+252 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 679 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **199 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ContinualLearner` connect `ContinualLearner Base Class` to `Community 37`, `Continual Learner Algorithms`, `Community 45`, `IL Benchmark Runner`, `Community 20`, `Community 29`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `ContinualLearner` (e.g. with `run_all_strategies()` and `run_benchmark()`) actually correct?**
  _`ContinualLearner` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `insert`, `update`, `insert` to the rest of the system?**
  _257 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `FLAIR Notebook Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.0532724505327245 - nodes in this community are weakly interconnected._
- **Why does `EWC` connect `Continual Learner Algorithms` to `Community 20`, `ContinualLearner Base Class`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `DASIG Style Modeling` be split into smaller, more focused modules?**
  _Cohesion score 0.06095481670929241 - nodes in this community are weakly interconnected._
- **Why does `LwF` connect `Community 37` to `Community 20`, `Continual Learner Algorithms`, `ContinualLearner Base Class`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._