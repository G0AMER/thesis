# 01 - Task Recognition from Physiological Data Using Multimodal Sensor Fusion

**Paper Title:** *Task Recognition from Physiological Data Using Multimodal Sensor Fusion For Human-Robot Collaboration*  
**Conference:** CoopIS (International Conference on Cooperative Information Systems)  
**Authors:** Ameur Gargouri, Mohamed Karray, Mohamed Ksantini  
**Dataset:** MultiPhysio-HRC (55 subjects, multimodal physiological signals: EEG, ECG, EDA, EMG, Respiration)

---

## Directory Structure

```
01_Task_Recognition_Multimodal_Fusion/
├── paper/               # Manuscript & LaTeX sources
│   ├── task_type_detection_paper.tex
│   ├── task_type_detection_paper.pdf
│   ├── task_type_detection_references.bib
│   ├── llncs.cls / splncs04.bst (Springer LNCS class & style)
│   └── confusion_matrix_tuned_xgboost.png
├── src/                 # Source code, Jupyter notebooks, benchmark scripts
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_fusion_dataset_builder.ipynb
│   ├── 04_loso_tabular_baselines.ipynb
│   ├── 05_loso_two_tower_fusion.ipynb
│   ├── 06_fusion_results_review.ipynb
│   ├── 07_full_benchmark.ipynb
│   ├── 08_classic_split_benchmark.py / .ipynb
│   ├── final_detector_loso.ipynb
│   ├── final_detector_loso_10models.py / .ipynb
│   └── scripts/
│       ├── train_final_loso.py
│       └── recompute_cis.py
├── assets/              # Pipeline figures and data
│   ├── task_pipeline.png
│   ├── generate_task_pipeline.py
│   ├── README_RESEARCH.md
│   └── data/
│       └── physiological_data.zip
└── output/              # Experimental outputs & benchmarks
    └── research_outputs/
        ├── fusion/               # fusion_dataset.csv (standardized fused tabular data)
        ├── preprocessing/        # intermediate preprocessing outputs
        └── fusion_training/      # model checkpoints, per-fold results, confusion matrices
```

## How to Run

### Run Classic Split Benchmark
```bash
cd src
python3 08_classic_split_benchmark.py
```

### Run Leave-One-Subject-Out (LOSO) Benchmark
```bash
cd src
python3 final_detector_loso_10models.py
```

### Compile Paper
```bash
cd paper
pdflatex task_type_detection_paper.tex
bibtex task_type_detection_paper
pdflatex task_type_detection_paper.tex
```
