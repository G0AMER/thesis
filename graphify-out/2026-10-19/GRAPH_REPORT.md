# Graph Report - thesis  (2026-10-18)

## Corpus Check
- 779 files · ~5,018,674 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 459 file(s) not represented in the graph (top: (none) 57, .ipynb 48, .tex 40)

## Summary
- 1443 nodes · 2599 edges · 254 communities (56 shown, 198 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 161 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d0075c78`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preprocess.py
- test_preprocessing.py
- DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels
- test_dasig_loader.py
- pandas
- DERPlusPlus
- pandas
- FLAIR 8-Layer Architecture
- Task Recognition from Physiological Data Using Multimodal Sensor Fusion for Human-Robot Collaboration (revised)
- train.py
- picda/models.py
- harmonic_pipeline.py
- Danger-State Detection in Human-Robot Collaboration Using Deep Learning (IEEE RA-L submission)
- data_loader.py
- ContinualLearner
- ablations.py
- Benchmark
- run_pipeline.py
- build_reference_report.py
- final_detector_loso_10models.py
- numpy
- features.py
- build_detailed_reference_report.py
- JointTraining
- harmonic_loader.py
- add_paper_details
- diagnostics.py
- 08_classic_split_benchmark.py
- EWC
- dasig_loader.py
- test_v4.py
- ContinualMetrics
- LwF
- Ameur Gargouri
- re
- json
- RecordingData
- train_final_loso.py
- PI-CDA: Physics-Informed Cross-Modal Domain Adaptation for Skeleton Action Recognition in Industrial Digital Twins
- Exp03 Definitive Benchmark: Average MSE evolution over training sequence for Naive Fine-Tune, Joint Training, EWC, Online EWC, LwF, DER++, A-GEM across 10 tasks
- extract_dataset_details
- IMU Sensor Placement Diagram (cobot_safety_cas): 5 IMU sensors on human body — Sternum (STR), Right/Left Upper Arm (RUA/LUA), Right/Left Front Arm (RFA/LFA) with coordinate frames
- DASIGPreprocessConfig
- load_trial
- Confusion matrix for ConvNeXt 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 97.13%, danger false negative rate 2.87%
- ConvNeXt 1D Confusion Matrix (cobot_safety_cas): SAFE 99.76%, DANGER 97.13% recall — high-performance deep learning classifier
- XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load 0.89, high_stress 0.91, industrial_task 0.99, low_load 0.75, other 0.93) — paper copy
- R2 evolution chart (original): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, matches presentation version)
- Exp06 DER-SA: Ablation study bar chart comparing DER++ baseline with ARW-only, FKD-only, JAC-only, pairwise and full DER-SA variants on accuracy and forgetting
- Exp07 CPG-Net: Ablation bar chart — No gating (EWC only), Random gates, No gate freezing, No importance reg., No gate sparsity vs full CPG-Net on accuracy and forgetting
- TestAlignTimestamps
- PI-CDA Per-Class F1 Bar Chart (Legacy Synthetic) — InHARD-DT target, 13 action classes, macro-F1 80.8%
- dasig_pipeline.py
- FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — original
- IEEE-conference-template/IEEEtran.cls
- Full task-condition recognition pipeline diagram: MultiPhysio-HRC dataset -> Preprocessing -> EEG features (96-d) + Peripheral physio (32-d) -> Fused representation (128-d) -> Detectors -> Task-condition label
- il_benchmark.py
- SequenceDataset
- 5-Fold CV bar chart comparing post-processed performance of TCN, CRNN+Attention, InceptionTime, U-Net 1D, Transformer on Macro F1 Score, Danger Recall, and Danger Precision; TCN leads Macro F1 (0.647), Transformer leads Danger Recall (0.789)
- Confusion matrix for Gradient Boosting classifier on cobot safety SAFE vs ABRUPT classification: 984 TP-safe, 4 FP-abrupt, 105 FN-safe, 59 TP-abrupt; poor abrupt movement recall
- cobot_safety_lra/IEEEtran.cls
- FLAIR R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks vs baselines
- _rms_envelope
- make_paper_v3.sh
- Dual boxplot comparing 5 deep learning models (TCN, U-Net 1D, CRNN+Attention, Transformer, InceptionTime) on Danger Recall and Macro F1 over 5 folds; Transformer shows high danger recall (~0.9) but very unstable Macro F1
- Confusion matrix for best TCN model (Factory Calibrated) on SAFE vs DANGER classification showing raw counts: 59235 TP-safe, 1324 FP-danger, 142 FN-safe, 8393 TP-danger
- exp01_il_comparison: R[i,j] accuracy matrices (MSE on task j after training up to task i) for 7 continual learning methods: Naive Fine-Tune, Joint Training, EWC, Online EWC, SI, LwF, DER++
- exp01_il_comparison: Average MSE on Seen Tasks Over Training Sequence line chart for 7 baselines (Naive Fine-Tune, Joint Training, EWC, Online EWC, SI, LwF, DER++) across 10 incremental tasks
- exp01_il_comparison: Average MSE After All Tasks and Backward Transfer (BWT) bar charts for 7 baselines; Joint Training achieves lowest MSE (0.5554), Naive Fine-Tune has highest BWT (+0.2379)
- exp01_il_comparison: Final MSE Per Task (per participant p100-p109) after training on all tasks, grouped bar chart for 7 baselines; participant p102 shows highest MSE across most methods
- Exp02 HP Sweep: Hyperparameter sweep plots for DER++ (buffer_size), EWC (ewc_lambda), Online EWC (gamma), and SI (si_c) showing Avg MSE and BWT trade-offs
- CPG-Net Per-Task Learning Curves (Train vs Val) for 10 participants, showing convergence flags
- CPG-Net Overfitting and Convergence Analysis: Train vs Val Loss per Task and Convergence Speed per Task
- FLAIR vs All Baselines: All 6 Metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time)
- paper_build.sh
- install_deps.sh
- t-SNE Domain Alignment: Before and After PI-CDA Adaptation (output/figures)
- Confusion Matrix Comparison: Zero-Shot vs PI-CDA vs Oracle (13 assembly action classes, InHARD-DT to InHARD)
- Few-Shot Domain Adaptation Performance Curve (PI-CDA vs DANN vs Source-Only vs Oracle, 0 to all shots)
- Action Anticipation Accuracy vs Prediction Horizon (Top-1 and Top-5 with/without APA, 0.5-3.0s)
- Cross-Subject Generalization LOSO Boxplot: Macro-F1 across 16 workers (Source-Only vs PI-CDA vs Oracle)
- Qualitative Action Anticipation Timeline for Cobot Lead-Time Planning (Bin/Pick sequence, observed vs ground truth)
- Learned APA Transition Matrix P(next|current) for 13 assembly actions (blue heatmap, max ~0.7)
- Sensor Research Comprehensive PDF
- ConvNeXt 1D confusion matrix for binary SAFE/DANGER classification (99.76% safe, 97.13% danger)
- Elsevier Logo 3p PDF
- IEEEtran HOWTO PDF
- IEEE Procs Template PDF
- SDlogo 3p PDF
- Test Font PDF
- 1st Research Thesis PDF
- Example document (template)
- Per-class precision, recall and F1-score bar chart for pooled LOSO GradientBoosting on 5-class task-condition recognition
- Box plot of subject-disjoint LOSO macro-F1 across 52 held-out operators for 11 classifiers (Tree, GaussNB, kNN, LogReg, MLP, ExtraTrees, SVM-RBF, RF, HistGB, XGBoost, GB)
- Modality ablation bar chart comparing fused, EEG-only, and physio-only LOSO macro-F1 for XGBoost, HistGB, RF, and MLP under subject-disjoint LOSO
- Generalization gap chart comparing within-subject (80/20) ceiling vs subject-disjoint LOSO macro-F1 for HistGB, XGBoost, RF, and MLP
- Statistical significance chart: delta macro-F1 vs best (GB) with per-fold Wilcoxon test over 52 folds for 10 classifiers
- LLNCS documentation (Springer LNCS template doc)
- ORCID logo icon
- ORCID logo (template asset)
- Horizontal bar chart comparing Accuracy, Balanced Accuracy, and Macro-F1 for XGBoost, Random Forest, MLP, Two-Tower, and LogReg detectors on task-condition recognition
- Lollipop chart showing macro-F1 gain from fused input vs best single modality for LogReg, Random Forest, XGBoost, and MLP classifiers
- Lollipop chart of inference latency (microseconds per window, log scale) for Random Forest, XGBoost, MLP, Two-Tower, and LogReg classifiers
- SCITEPRESS logo (template asset)
- Task Recognition paper v3
- IMU sensor placement diagram showing 5 body locations (Sternum STR, Right/Left Upper Arm RUA/LUA, Right/Left Front Arm RFA/LFA) with 3-axis coordinate frames for cobot safety motion capture setup
- Scatter plot of magnetization (kA/m) vs applied field (10^4 A/m) showing nonlinear saturation curve — appears to be an unrelated physics figure placed in the danger detection assets folder
- Photo of human-robot collaboration experiment: participant seated at table with Tnova cobot arm during DASIG corpus data collection; participant wears eye-tracking glasses
- Data pipeline diagram showing two stages: (1) Preprocessing and Windowing of DASIG Corpus (60 subjects x 3 trials, 5 wireless MIMUs at 200 Hz, 0.5s sliding window, 69094 windows, 87.6% SAFE / 12.4% DANGER), (2) Dynamic Kinematic Feature Expansion producing expanded tensor [X;V;A] of shape (B, 195, 100)
- Pipeline Fig 3: Training Loop with 30 Epochs, AdamW, OneCycleLR, Focal Loss, Mixup Augmentation, Kinematic Derivative Expansion
- Top 25 feature importances for Gradient Boosting on cobot safety classification; feature_77 (~0.157) and feature_73 (~0.153) dominate; remaining 23 features each below 0.04 importance
- Random Forest Normalized Confusion Matrix (cobot_safety): SAFE 99.90%, ABRUPT 26.22% recall — binary danger state detection baseline
- Random Forest Top 25 Feature Importances (cobot_safety): STR accelerometer range/peak/std dominate; includes LFA, RFA, LUA, RUA sensor channels for acc, gyr, jerk statistics
- CrossRef Logo
- Cambridge University Press Logo
- Paper Abstract Screenshot (cobot_safety_springer technical check): abstract of 'Danger-State Detection in Human-Robot Collaboration Using Deep Learning' discussing cobot physical hazards and false-alarm trade-off
- Paper Authors Page Screenshot (cobot_safety_springer technical check): title 'Danger-State Detection in Human-Robot Collaboration Using Deep Learning', authors Ameur Gargouri, Mohamed Karray, Mohamed Ksantini, ATISP Lab ENET'Com University of Sfax and ESME Research Lab
- Springer Nature Article Template User Manual
- FLAIR vs All Baselines — All 6 Metrics bar chart: ACC, Forgetting, BWT, FWT, Memory Usage, Training Time comparing FLAIR (ours), Joint Training, DER++, A-GEM, Online EWC, EWC
- FLAIR HRC setup photo (fig1): human participant wearing eye-tracking glasses working alongside Inova collaborative robot arm at a table with objects and AR markers
- FLAIR vs All Baselines — All 6 Metrics bar chart comparing FLAIR, Joint Training, DER++, A-GEM, Online EWC across ACC, Forgetting, BWT, FWT, Memory Usage, Training Time
- Contact QR code for presentation
- Human-robot collaboration setup: person wearing eye-tracking glasses interacting with Kinova robotic arm at a table, InHARD dataset context
- KES International conference logo
- ESME Research Lab institution logo
- Task processing pipeline diagram: Physiological Signals -> Preprocessing Imputation + Scaling -> Feature/Sequence Encoding -> Classifier XGBoost/BiLSTM/Fusion
- University of Sfax Logo (Tunisian institution branding)
- Incubation Pitch 2-Minute Presentation
- IMU sensor placement diagram on human body: STR (sternum), RUA (right upper arm), LUA (left upper arm), RFA (right forearm), LFA (left forearm) with XYZ axes shown, from robotics journal figure
- exp01_il_comparison: Forgetting Curve — MSE on Task 0 over training sequence for 7 baselines; Joint Training and DER++ show best retention, SI spikes sharply
- Exp05 Scalability: 6-panel scalability analysis showing R², threshold accuracy, per-task R², MSE, forgetting BWT, and per-task threshold accuracy at 10 participants for Naive, Joint, DER++, Online EWC, A-GEM
- CPG-Net: Learned Gating Patterns Across Tasks (Layer 1 & 2 Gate Values Heatmap)
- CPG-Net Per-Joint MSE Comparison: DER++ vs DER-SA vs CPG-Net (ours) across 6 joints
- CPG-Net Per-Task R2 After All Training: v2 baseline vs v3 (all improvements) for 10 tasks
- CPG-Net v3 Per-Improvement Ablation: ACC, Forgetting F, and BWT across improvement increments (Retro, Warm, Adapt, All)
- Elsevier LaTeX Documentation (paper folder)
- Elsevier Logo 3p (paper folder)
- Paper Fig 1: Photo of human-robot collaboration setup with Inova cobot arm and participant wearing eye-tracking glasses
- KES 2026 Template Sample Figure (gr1, paper folder)
- KES 2026 Procedia Style Guide (Procs, paper folder)
- Elsevier LaTeX Documentation
- Elsevier Logo 3p
- KES 2026 Template Sample Figure (gr1)
- KES 2026 Procedia Style Guide (Procs)
- KES 2026 Procedia LaTeX Template
- ScienceDirect Logo 3p
- ScienceDirect Logo 3p (paper folder)
- FLAIR Requirements
- EDA Domain Gap Analysis: InHARD-DT Joint Velocity Magnitude density, Per-Joint Mean Velocity bar chart, and Right Wrist Trajectories scatter plot
- PI-CDA Component Ablation Study: Target Macro-F1 scores for Full PI-CDA, w/o Velocity MMD, w/o Topology Regularization, w/o APA Inference Filter, and Source-Only Baseline
- PI-CDA System Overview: Cross-Modal Digital Twin Domain Adaptation Framework with InHARD-DT source, InHARD Real target, Shared CTR-GCN, Action Recognition, Action Anticipation, and CSTA Alignment
- Unified 17-Joint Topology Mapping: Skeleton diagrams showing joint subsampling and normalization from InHARD-DT BVH MoCap (59J) and InHARD Kinect v2 Depth (25J) to canonical 17-joint model
- Action Precedence Automaton (APA) Empirical Transition Matrix: 13x13 heatmap of assembly action transition probabilities (Pick, Position, Align, Screw, Hammer, Inspect, Turn, Aside, Bin, Idle, Adjust, Clean, Discard)
- Zero-Shot Sim-to-Real Confusion Matrix E1 (InHARD-DT to InHARD, 13 classes with numerical values)
- PI-CDA Per-Class F1 Performance on InHARD-DT Target (13 classes, Macro-F1 80.8%, Pick=100%)
- Component Ablation Study of PI-CDA Framework (Full=74.2%, w/o Velocity MMD=68.5%, w/o Topo Reg=69.1%, w/o APA Filter=70.8%, Source-Only=44.1%)
- PI-CDA Cross-Modal Digital Twin Domain Adaptation Framework Overview (InHARD-DT MoCap + InHARD Kinect to Shared CTR-GCN for recognition, anticipation, CSTA alignment)
- Unified 17-Joint Topology Mapping across MoCap (59J) and Kinect (25J) to Canonical Model (17J)
- Action Precedence Automaton (APA) Empirical Transition Matrix P(a_t+1|a_t) for 13 assembly actions (red/orange heatmap, values annotated)
- CT-R-GCN Per-Subject F1 Box Plot — 16 InHARD subjects, seed-averaged, comparing source_only/mmd/picda/fixmatch/oracle/source_plus_target
- InHARD-DT Dataset Statistics — clips per class and duration per class (DT vs real domain), 13 action classes
- Domain Gap Duration Histogram — annotated clip duration distribution (0–8s) for DT vs real InHARD, showing DT has discrete peaks at fixed durations
- Domain Gap Facing Yaw — first-frame yaw distribution (DT around -60 deg, real around +100 deg) and per-class median yaw comparison
- Domain Gap Key Descriptors Per Class — 6-panel box plot comparing DT vs real on duration, wrist speed, hand reach, trunk flexion, root path, and wrist lateral displacement across 13 classes
- Domain Shift Heatmap — Cliff's delta (DT vs real) per class and feature, 13 classes x 17 biomechanical features, BH-corrected significance markers
- Elsevier Logo 3p Template
- Procs Template PDF
- SD Logo 3p Template
- Digital Twin Requirements
- CLAUDE.md — Graphify Instructions
- CV Avatar Photo — circular portrait photo of the author (young man with glasses, patterned shirt, outdoor background)
- CV Certificates Icon — black diploma/scroll icon for certificates section
- CV Clubs Icon — black group/team silhouette icon for clubs/associations section
- CV Codeforces / Competitive Programming Icon — bar chart outline icon representing competitive programming rankings
- CV Contact Icon — phone handset outline icon for contact section
- CV Document Icon — outline icon of a CV/resume document with profile picture placeholder
- CV Education Icon — graduation cap outline icon for education section
- CV Hobbies Icon — outline icon of person surrounded by book, music note, and video player representing hobbies
- CV Languages Icon — two overlapping cards with 'A' and a Chinese character representing multilingual section
- CV Person Icon — generic user/person outline icon
- Projects Section Icon - Light bulb with gear overlay representing innovative projects and engineering ideas
- Publications Section Icon - Person at podium with microphones representing public speaking and academic publications
- Skills Section Icon - Lightning bolt symbol representing energy, capability and technical skills
- Summary Section Icon - Document with pen representing written profile or personal summary
- Target/Goal Section Icon - Bullseye with arrow representing objectives and career goals
- Work Experience Section Icon - Briefcase representing professional work experience
- hrc-incremental
- discover_trials
- copilot-instructions.md
- claude-mem-context.md
- TrialData
- test_harmonic_loader.py
- MLPPolicy
- GRUNet
- LSTMNet
- MLPNet
- TransformerNet
- TwoTowerNet

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
10. `run_benchmark()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `FLAIR: A Combined Incremental Learning Algorithm for Human-Robot Collaboration` --references--> `HARMONIC Dataset: 24 subjects, EMG+IMU+gaze+joystick, shared autonomy`  [EXTRACTED]
  03_Incremental_Learning_FLAIR/paper/flair_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `GPU-Accelerated Kinematic Feature Expansion (65 to 195 channels via derivatives)` --conceptually_related_to--> `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `Temporal Debouncing: median filter (k=3) + threshold 0.77 → 87.5% FP reduction` --conceptually_related_to--> `False-Alarm vs. Productivity Trade-off in Cobot Safety`  [EXTRACTED]
  00_Thesis_Dissertation_and_Reports/assets/Literature_Contributions.md → 02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf
- `Deep Learning for Safety-State Detection in HRC (KES 2026, Procedia CS)` --references--> `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/project_info__1.md
- `Deep Learning for Safety-State Detection in HRC (KES 2026, Procedia CS)` --references--> `Temporal Debouncing: median filter (k=3) + threshold 0.77 → 87.5% FP reduction`  [EXTRACTED]
  02_Danger_State_Detection_Deep_Learning/paper/cobot_safety_paper.pdf → 00_Thesis_Dissertation_and_Reports/assets/Literature_Contributions.md

## Import Cycles
- None detected.

## Communities (254 total, 198 thin omitted)

### Community 0 - "preprocess.py"
Cohesion: 0.19
Nodes (8): _axis_rotation(), BVHClip, forward_kinematics(), read_bvh(), resample(), build_all(), parse_clip_name(), process_domain()

### Community 1 - "test_preprocessing.py"
Cohesion: 0.16
Nodes (9): butterworth_lowpass(), compute_frequency_features(), compute_jerk_features(), compute_time_features(), extract_features_windowed(), _safe_kurtosis(), _safe_skew(), TestButterworthLowpass (+1 more)

### Community 2 - "DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels"
Cohesion: 0.05
Nodes (61): Five Novel Contributions: IMU-based Safety Pipeline, State-of-the-Art Cobot Safety Systems 2024-2025, PhD Thesis Codebase Analysis, PhD Thesis Architecture and Data Flows, Sensor Modalities for Shared Autonomy and HRC, Thesis Defense Presentation Slides (Marp), Novel Datasets and Research Proposals for HRC, Rapport de Synthèse Scientifique des Travaux de Thèse (+53 more)

### Community 3 - "test_dasig_loader.py"
Cohesion: 0.11
Nodes (8): dasig_tree(), _make_arduino_csv(), _make_mimu_csv(), TestConstants, TestDiscovery, TestLoading, TestSubjectsInfo, discover_recordings()

### Community 4 - "pandas"
Cohesion: 0.11
Nodes (20): arrow(), build_figure(), rbox(), draw_arrow(), draw_box(), fig1_pipeline_overview(), fig2_data_pipeline(), fig3_training_loop() (+12 more)

### Community 6 - "pandas"
Cohesion: 0.08
Nodes (30): best_model(), fig_ablation(), fig_confusion(), fig_gap(), fig_perclass(), fig_significance(), fig_subject_box(), load_summary() (+22 more)

### Community 7 - "FLAIR 8-Layer Architecture"
Cohesion: 0.07
Nodes (41): FLAIR KES 2026 Presentation, Dataset Exploration Report (HARMONIC and DASIG), DASIG Dataset, Preprocessing Pipeline Report (HARMONIC and DASIG), Incremental Learning Benchmark Report (Exp 1), Catastrophic Forgetting, Shared Autonomy Benchmark Report (Exp 01), FLAIR — 03 Incremental Learning README (+33 more)

### Community 8 - "Task Recognition from Physiological Data Using Multimodal Sensor Fusion for Human-Robot Collaboration (revised)"
Cohesion: 0.06
Nodes (38): DER/DER++: Dark Experience Replay for incremental learning, Incremental Learning Algorithms in Autonomous Driving and Robot Lifelong Skill Acquisition, iCaRL: Incremental Classifier and Representation Learning, Real-World Robotics Datasets for Incremental Learning and Lifelong Learning, LIBERO: Lifelong Robot Learning Benchmark Suite, OpenLORIS-Object Dataset for lifelong robotic vision, MultiPhysio-HRC: Multimodal Physiological Signals Dataset for Industrial HRC, MultiPhysio-HRC Dataset (EEG, ECG, EDA, RESP, EMG, Voice, Facial features) (+30 more)

### Community 9 - "train.py"
Cohesion: 0.13
Nodes (7): worker_init_fn(), coral_loss(), DomainDiscriminator, grad_reverse(), _GradReverse, mmd_loss(), topology_consistency_loss()

### Community 10 - "picda/models.py"
Cohesion: 0.09
Nodes (11): bn_init(), build_model(), conv_init(), CTRGC, CTRGCN, MultiScaleTemporalConv, STGCN, STGCNBlock (+3 more)

### Community 11 - "harmonic_pipeline.py"
Cohesion: 0.13
Nodes (12): main(), preprocess_trial(), align_timestamps(), compute_participant_stats(), normalize_dataframe(), preprocess_emg(), preprocess_gaze(), preprocess_imu() (+4 more)

### Community 12 - "Danger-State Detection in Human-Robot Collaboration Using Deep Learning (IEEE RA-L submission)"
Cohesion: 0.10
Nodes (27): Fig 4: Neural Network Architectures for 1D Time-Series Classification (TCN, ConvNeXt 1D, MLP-Mixer 1D, Transformer 1D, InceptionTime), Fig 5: Evaluation Post-Processing Pipeline (Raw Probabilities -> Temporal Debouncing -> Threshold Decision SAFE/DANGER), Fig 5: Evaluation Post-Processing (copy), Fig 6: Model Performance Comparison Table (Threshold=0.77, Median Smoothed; TCN ranked 1st with Macro F1 97.88%), Fig 7: 5-Fold Stratified Cross-Validation Strategy (~55,275 train / ~13,819 validation windows per fold), Pipeline Fig 1: Overview of 4-Phase Deep Learning Pipeline (Data, Training, Evaluation, Deployment), Pipeline Fig 2: Data Pipeline showing Preprocessing & Windowing and Dynamic Kinematic Feature Expansion (65 -> 195 channels), Pipeline Fig 4: Neural Network Architectures for 1D Time-Series Classification (TCN, ConvNeXt 1D, MLP-Mixer 1D, Transformer 1D, InceptionTime) (+19 more)

### Community 13 - "data_loader.py"
Cohesion: 0.14
Nodes (8): build_column_names(), generate_safety_labels(), load_all_subject_info(), load_arduino_csv(), load_mimu_csv(), load_subject_info(), load_trial(), SubjectInfo

### Community 15 - "ablations.py"
Cohesion: 0.15
Nodes (8): cols_of(), indices_of(), load(), loso_sklearn(), loso_twotower(), main(), TwoTower, make_models()

### Community 16 - "Benchmark"
Cohesion: 0.13
Nodes (13): Benchmark, infinite(), _log(), make_loader(), make_optimizer(), predict(), set_seed(), _Stepper (+5 more)

### Community 17 - "run_pipeline.py"
Cohesion: 0.20
Nodes (7): evaluate_model(), plot_feature_importance(), to_binary_labels(), train_gradient_boosting(), train_random_forest(), main(), parse_args()

### Community 18 - "build_reference_report.py"
Cohesion: 0.24
Nodes (16): build_seed_record(), clean_text(), extract_doi_from_url(), find_openalex_work(), format_authors(), get_crossref_metadata(), get_hrc_citing_papers(), get_json() (+8 more)

### Community 19 - "final_detector_loso_10models.py"
Cohesion: 0.15
Nodes (4): fold_dir(), fold_labels(), make_models(), run_loso_model()

### Community 21 - "numpy"
Cohesion: 0.05
Nodes (45): ece_from_1hot(), fold_metrics(), load_clean(), main(), per_class_recall(), t_ci95(), estimate_latency_us(), all_joint_names() (+37 more)

### Community 22 - "features.py"
Cohesion: 0.20
Nodes (7): extract_features_bulk(), extract_sensor_features(), extract_window_features(), normalize_features(), segment_all_trials(), sliding_window_segment(), _vector_norm()

### Community 23 - "build_detailed_reference_report.py"
Cohesion: 0.11
Nodes (3): extract_key_contribution(), get_binary(), get_pdf_text_from_url()

### Community 25 - "harmonic_loader.py"
Cohesion: 0.22
Nodes (3): discover_participants(), load_participant_trials(), TrialData

### Community 26 - "add_paper_details"
Cohesion: 0.12
Nodes (8): add_hyperlink(), add_paper_details(), build_key_contribution(), build_method_summary(), build_results_summary(), build_work_summary(), extract_approach_details(), main()

### Community 27 - "diagnostics.py"
Cohesion: 0.23
Nodes (6): adjust(), em_prior(), fold_table(), real_probs(), _scores(), summarize()

### Community 28 - "08_classic_split_benchmark.py"
Cohesion: 0.13
Nodes (5): BenchConfig, compute_class_weights(), reshape_to_sequences(), TCNNet, train_torch_model()

### Community 29 - "EWC"
Cohesion: 0.05
Nodes (3): EWC, SI, ReplayBuffer

### Community 30 - "dasig_loader.py"
Cohesion: 0.21
Nodes (4): load_arduino(), load_mimu(), load_subjects_info(), SubjectInfo

### Community 32 - "ContinualMetrics"
Cohesion: 0.20
Nodes (3): ContinualMetrics, TaskResult, run_benchmark()

### Community 34 - "Ameur Gargouri"
Cohesion: 0.19
Nodes (13): ATISP Laboratory, University of Sfax / ESME Research Lab, Danger State Detection, Embedded AI, FLAIR Continual Learning Method, Human-Robot Collaboration Research, Incremental Learning, Multimodal Sensor Fusion, Robot Safety (+5 more)

### Community 35 - "re"
Cohesion: 0.21
Nodes (6): discover_pdf_url(), extract_data_details(), extract_full_paper_text(), extract_results_details(), get_arxiv_pdf_text(), get_text()

### Community 36 - "json"
Cohesion: 0.08
Nodes (16): is_id_repetition_leak(), main(), modality_of(), clean_source(), main(), make_cell_id(), leave_n_subjects_out(), load_split() (+8 more)

### Community 38 - "train_final_loso.py"
Cohesion: 0.36
Nodes (5): build_classifiers(), get_feature_columns(), main(), run_loso(), save_dict_csv()

### Community 39 - "PI-CDA: Physics-Informed Cross-Modal Domain Adaptation for Skeleton Action Recognition in Industrial Digital Twins"
Cohesion: 0.31
Nodes (9): PI-CDA Adjacency Matrix Figure, PI-CDA APA Transition Matrix Figure, PI-CDA Augmentation Demo Figure, PI-CDA: Physics-Informed Cross-Modal Domain Adaptation for Skeleton Action Recognition in Industrial Digital Twins, Contrastive Skeleton Topology Alignment (CSTA), Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN), InHARD-DT Dataset (Industrial Human Action Recognition with Digital Twin), PI-CDA (Physics-Informed Cross-Modal Domain Adaptation) (+1 more)

### Community 40 - "Exp03 Definitive Benchmark: Average MSE evolution over training sequence for Naive Fine-Tune, Joint Training, EWC, Online EWC, LwF, DER++, A-GEM across 10 tasks"
Cohesion: 0.25
Nodes (8): Exp03 Definitive Benchmark: Average MSE evolution over training sequence for Naive Fine-Tune, Joint Training, EWC, Online EWC, LwF, DER++, A-GEM across 10 tasks, Exp03 Definitive Benchmark: Comparison bar charts showing Avg MSE (lower=better) and Backward Transfer BWT for all methods — DER++ achieves 0.4341 MSE, Joint Training 0.3853, Exp03 Definitive Benchmark: Forgetting curve for Task 0 (p100) — MSE on Task 0 over training steps for all continual learning methods; DER++ shows least forgetting, Exp03 Definitive Benchmark: Joint forgetting correlation heatmap for DER++ — 6x6 correlation matrix across joints showing high cross-joint forgetting correlation (0.67–1.00), Exp03 Definitive Benchmark: Per-joint velocity MSE averaged across all tasks and per-joint backward transfer (forgetting) for all methods across 6 joints, Exp03 Definitive Benchmark: Per-joint forgetting curves for Task 0 (p100) — 6 subplots for J1 shoulder, J2 shoulder, J3 elbow, J4 wrist, J5 wrist, J6 wrist showing MSE evolution, Exp03 Definitive Benchmark: Per-participant per-joint MSE heatmaps comparing DER++ vs Naive Fine-Tune final performance across participants p100–p109 and 6 joints, Exp03 Definitive Benchmark: Final MSE per task after all training — grouped bar chart for all methods across participants p100–p109 including A-GEM

### Community 41 - "extract_dataset_details"
Cohesion: 0.29
Nodes (4): build_dataset_summary(), extract_dataset_details(), extract_dataset_profile(), extract_detail_phrases()

### Community 44 - "IMU Sensor Placement Diagram (cobot_safety_cas): 5 IMU sensors on human body — Sternum (STR), Right/Left Upper Arm (RUA/LUA), Right/Left Front Arm (RFA/LFA) with coordinate frames"
Cohesion: 0.33
Nodes (6): IMU Sensor Placement Diagram (cobot_safety_cas): 5 IMU sensors on human body — Sternum (STR), Right/Left Upper Arm (RUA/LUA), Right/Left Front Arm (RFA/LFA) with coordinate frames, IMU Sensor Placement Diagram (cobot_safety_lra): 5 IMU sensors on human body — Sternum, RUA, LUA, RFA, LFA with global coordinate frame annotation, MIMU Placement Diagram (cobot_safety_lra): detailed IMU sensor body placement figure showing STR, RUA, LUA, RFA, LFA positions with 3-axis coordinate frame per sensor, IMU Sensor Placement Diagram (cobot_safety_robotica): 5 IMU sensors on human body — same layout as other paper versions, IMU Sensor Placement Diagram (cobot_safety_springer): 5 IMU sensors on human body — same layout as other paper versions, IMU Sensor Placement Diagram (cobot_safety_submission): 5 IMU sensors on human body — same layout as other paper versions

### Community 45 - "DASIGPreprocessConfig"
Cohesion: 0.14
Nodes (9): preprocess_recording(), DASIGPreprocessConfig, detect_abrupt_motions(), preprocess_mimu(), segment_around_alarms(), _make_mimu_df(), TestAbruptDetection, TestAlarmSegmentation (+1 more)

### Community 46 - "load_trial"
Cohesion: 0.23
Nodes (4): run_harmonic_pipeline(), save_preprocessed_trial(), load_trial(), TrialInfo

### Community 47 - "Confusion matrix for ConvNeXt 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 97.13%, danger false negative rate 2.87%"
Cohesion: 0.40
Nodes (5): Confusion matrix for ConvNeXt 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 97.13%, danger false negative rate 2.87%, Confusion matrix for InceptionTime model on cobot danger state detection: SAFE recall 98.41%, DANGER recall 97.53%, danger false negative rate 2.47%, Confusion matrix for MLP-Mixer 1D model on cobot danger state detection: SAFE recall 99.76%, DANGER recall 95.88%, danger false negative rate 4.12%, Confusion matrix for TCN model on cobot danger state detection: SAFE recall 99.73%, DANGER recall 97.14%, danger false negative rate 2.86%, Confusion matrix for Transformer 1D model on cobot danger state detection: SAFE recall 96.56%, DANGER recall 95.85%, danger false negative rate 4.15%

### Community 48 - "ConvNeXt 1D Confusion Matrix (cobot_safety_cas): SAFE 99.76%, DANGER 97.13% recall — high-performance deep learning classifier"
Cohesion: 0.40
Nodes (5): ConvNeXt 1D Confusion Matrix (cobot_safety_cas): SAFE 99.76%, DANGER 97.13% recall — high-performance deep learning classifier, ConvNeXt 1D Confusion Matrix (cobot_safety_lra): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_robotica): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_springer): SAFE 99.76%, DANGER 97.13% recall, ConvNeXt 1D Confusion Matrix (cobot_safety_submission): SAFE 99.76%, DANGER 97.13% recall

### Community 50 - "XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load 0.89, high_stress 0.91, industrial_task 0.99, low_load 0.75, other 0.93) — paper copy"
Cohesion: 0.50
Nodes (4): XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load, high_stress, industrial_task, low_load, other) — assets copy, XGBoost confusion matrix for 5-class task-condition recognition (cognitive_load 0.89, high_stress 0.91, industrial_task 0.99, low_load 0.75, other 0.93) — paper copy, XGBoost confusion matrix for 5-class task-condition recognition using 80/20 stratified split (cognitive_load 0.91, high_stress 0.89, industrial_task 1.00, low_load 0.82, other 0.79), Pooled LOSO confusion matrix for GradientBoosting classifier on 5-class task-condition recognition with raw window counts

### Community 51 - "R2 evolution chart (original): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, matches presentation version)"
Cohesion: 0.67
Nodes (4): R2 evolution chart (presentation version): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, no standalone EWC), R2 evolution chart (original): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training (5 methods, matches presentation version), R2 evolution chart (copy): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training, EWC (6 methods, different data range from original), R2 evolution chart (copy 2): Forgetting Trajectory on Task 0 and Average Accuracy Evolution across 10 incremental tasks, comparing FLAIR, DER++, Online EWC, A-GEM, Joint Training, EWC (6 methods including standalone EWC)

### Community 52 - "Exp06 DER-SA: Ablation study bar chart comparing DER++ baseline with ARW-only, FKD-only, JAC-only, pairwise and full DER-SA variants on accuracy and forgetting"
Cohesion: 0.50
Nodes (4): Exp06 DER-SA: Ablation study bar chart comparing DER++ baseline with ARW-only, FKD-only, JAC-only, pairwise and full DER-SA variants on accuracy and forgetting, Exp06 DER-SA: All 6 metrics comparison — DER-SA vs baselines for ACC, Forgetting F, BWT, FWT, Memory Usage, Training Time showing DER-SA competitive with DER++ at much lower cost, Exp06 DER-SA: Per-joint MSE comparison DER++ vs DER-SA across 6 joints — DER-SA improves J1–J5 by 0.2–4.0%, small regression on J6, Exp06 DER-SA: R² evolution — forgetting trajectory on Task 0 (p100) and average accuracy evolution across tasks for Naive Fine-Tune, DER++, DER-SA, Joint Training

### Community 53 - "Exp07 CPG-Net: Ablation bar chart — No gating (EWC only), Random gates, No gate freezing, No importance reg., No gate sparsity vs full CPG-Net on accuracy and forgetting"
Cohesion: 0.50
Nodes (4): Exp07 CPG-Net: Ablation bar chart — No gating (EWC only), Random gates, No gate freezing, No importance reg., No gate sparsity vs full CPG-Net on accuracy and forgetting, Exp07 CPG-Net: All 6 metrics comparison — CPG-Net vs Joint Training, DER++, A-GEM, Online EWC, Naive Fine-Tune on ACC, Forgetting F, BWT, FWT, Memory Usage, Training Time, Exp07 CPG-Net: Per-feature ablation progression — Base, +Retro Boost, +Warm Start, +Adapt Replay, Full CPG-Net on ACC (0.6642→0.6895), Forgetting (0.0387→0.0168), BWT (-0.039→-0.017), Exp07 CPG-Net: Per-task R² after all training — Base vs Full CPG-Net for 10 tasks; Full CPG-Net improves most tasks by +0.001 to +0.056

### Community 55 - "PI-CDA Per-Class F1 Bar Chart (Legacy Synthetic) — InHARD-DT target, 13 action classes, macro-F1 80.8%"
Cohesion: 0.50
Nodes (4): PI-CDA Per-Class F1 Bar Chart (Legacy Synthetic) — InHARD-DT target, 13 action classes, macro-F1 80.8%, CT-R-GCN Confusion Matrices — side-by-side comparison of source_only, picda, and oracle conditions on InHARD 13-class action recognition, CT-R-GCN Few-Shot Learning Curve — macro-F1 vs labeled InHARD clips per class (1/5/10-shot) for fine-tune from picda, scratch, and source_only, CT-R-GCN t-SNE Domain Alignment — 2x3 grid showing by-domain and by-class embeddings for source_only, picda, and oracle conditions

### Community 56 - "dasig_pipeline.py"
Cohesion: 0.18
Nodes (7): main(), run_dasig_pipeline(), save_preprocessed_recording(), compute_subject_stats(), normalize_features(), load_recording(), RecordingInfo

### Community 57 - "FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — original"
Cohesion: 1.00
Nodes (3): FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — original, FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — variant copy, FLAIR vs all baselines across 6 continual learning metrics (ACC, Forgetting, BWT, FWT, Memory Usage, Training Time) — variant copy 2

### Community 59 - "Full task-condition recognition pipeline diagram: MultiPhysio-HRC dataset -> Preprocessing -> EEG features (96-d) + Peripheral physio (32-d) -> Fused representation (128-d) -> Detectors -> Task-condition label"
Cohesion: 0.67
Nodes (3): Full task-condition recognition pipeline diagram: MultiPhysio-HRC dataset -> Preprocessing -> EEG features (96-d) + Peripheral physio (32-d) -> Fused representation (128-d) -> Detectors -> Task-condition label, Simplified task-type detection pipeline diagram: Physiological Signals -> Preprocessing -> Feature/Sequence Encoding -> Classifier (XGBoost/BiLSTM/Fusion), Full task-condition recognition pipeline diagram (paper version): MultiPhysio-HRC -> Preprocessing -> EEG/Peripheral features -> Detectors -> Evaluation

### Community 60 - "il_benchmark.py"
Cohesion: 0.10
Nodes (7): BenchmarkResult, build_task_sequence(), load_participant_data(), results_to_dataframe(), run_all_strategies(), save_benchmark_results(), TaskData

### Community 64 - "SequenceDataset"
Cohesion: 0.15
Nodes (3): SequenceDataset, TabularDataset, TwoTowerDataset

### Community 65 - "5-Fold CV bar chart comparing post-processed performance of TCN, CRNN+Attention, InceptionTime, U-Net 1D, Transformer on Macro F1 Score, Danger Recall, and Danger Precision; TCN leads Macro F1 (0.647), Transformer leads Danger Recall (0.789)"
Cohesion: 1.00
Nodes (3): 5-Fold CV bar chart comparing post-processed performance of TCN, CRNN+Attention, InceptionTime, U-Net 1D, Transformer on Macro F1 Score, Danger Recall, and Danger Precision; TCN leads Macro F1 (0.647), Transformer leads Danger Recall (0.789), 5-Fold CV bar chart for window-level classification with feature engineering comparing TCN, InceptionTime, ResNet1D, CRNN+Attention, LSTM-FCN on Accuracy, Macro F1, Danger Recall, and Danger Precision; TCN leads on Accuracy (0.898) and Macro F1 (0.719), Subject-Dependent Evaluation (Factory-Calibrated Robots) bar chart comparing TCN, ConvNeXt 1D, MLP-Mixer 1D, InceptionTime, Transformer 1D on Accuracy, Macro F1, Danger Recall, and Danger Precision; TCN leads (Accuracy 95.9%, Macro F1 90.5%, Danger Recall 82.8%)

### Community 66 - "Confusion matrix for Gradient Boosting classifier on cobot safety SAFE vs ABRUPT classification: 984 TP-safe, 4 FP-abrupt, 105 FN-safe, 59 TP-abrupt; poor abrupt movement recall"
Cohesion: 0.67
Nodes (3): Confusion matrix for Gradient Boosting classifier on cobot safety SAFE vs ABRUPT classification: 984 TP-safe, 4 FP-abrupt, 105 FN-safe, 59 TP-abrupt; poor abrupt movement recall, Normalized confusion matrix for Gradient Boosting on cobot safety SAFE vs ABRUPT: SAFE precision 99.60%, ABRUPT recall only 35.98%, showing severe class imbalance problem, Confusion matrix for Random Forest classifier on cobot safety SAFE vs ABRUPT classification: 987 TP-safe, 1 FP-abrupt, 121 FN-safe, 43 TP-abrupt; even lower abrupt recall than gradient boosting

### Community 68 - "FLAIR R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks vs baselines"
Cohesion: 0.67
Nodes (3): CPG-Net R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks, FLAIR R2 Evolution: Forgetting Trajectory Task 0 and Average Accuracy Evolution across 10 tasks vs baselines, FLAIR Paper Copy: R2 Evolution showing Forgetting Trajectory and Average Accuracy Evolution vs baselines

### Community 249 - "discover_trials"
Cohesion: 0.31
Nodes (3): discover_trials(), TestDiscovery, TestLoading

### Community 254 - "TrialData"
Cohesion: 0.22
Nodes (3): load_all_trials(), split_by_subject(), TrialData

### Community 255 - "test_harmonic_loader.py"
Cohesion: 0.13
Nodes (5): get_data_columns(), get_timestamp_column(), harmonic_tree(), TestConstants, TestHelpers

## Knowledge Gaps
- **257 isolated node(s):** `insert`, `update`, `insert`, `update`, `hrc-incremental` (+252 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 686 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **198 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ContinualLearner` connect `ContinualLearner` to `ContinualMetrics`, `LwF`, `DERPlusPlus`, `typing`, `JointTraining`, `il_benchmark.py`, `EWC`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `insert`, `update`, `insert` to the rest of the system?**
  _257 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DASIG Dataset: 60 subjects, 65 MIMU channels, danger/safe labels` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Why does `EWC` connect `EWC` to `typing`, `ContinualLearner`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `test_dasig_loader.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10837438423645321 - nodes in this community are weakly interconnected._
- **Why does `LwF` connect `LwF` to `typing`, `EWC`, `ContinualLearner`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.10810810810810811 - nodes in this community are weakly interconnected._