# 02 - Danger-State Detection in Human-Robot Collaboration Using Deep Learning

**Paper Title:** *Danger-State Detection in Human-Robot Collaboration Using Deep Learning*  
**Target:** Elsevier / Springer / Robotica / JINT  
**Authors:** Ameur Gargouri, Mohamed Karray, Mohamed Ksantini  
**Dataset:** DASIG (60 subjects, 65 MIMU inertial channels, standard vs. abrupt/danger movements)

---

## Directory Structure

```
02_Danger_State_Detection_Deep_Learning/
├── paper/               # Journal submission packages & LaTeX templates
│   ├── cobot_safety_cas/       # Elsevier CAS template & compiled manuscript
│   ├── cobot_safety_submission/# Final submission bundle
│   ├── cobot_safety_robotica/  # Cambridge Robotica format
│   ├── cobot_safety_springer/  # Springer format
│   ├── sn-article-template/
│   └── cover_letter_JINT.*
├── src/                 # Pipeline and model training code
│   ├── cobot_safety_model/     # Feature extraction, data loaders, ML models
│   │   ├── data_loader.py      # MIMU loader with European decimal format
│   │   ├── features.py         # Kinematic derivation, sliding windows
│   │   └── models.py           # Random Forest, Gradient Boosting
│   ├── run_pipeline.py         # Classical ML end-to-end pipeline
│   ├── test_v4.py              # 1D Deep learning pipeline (TCN, ConvNeXt, 5-fold CV)
│   ├── cobot_safety_dl_pipeline.ipynb
│   ├── cobot_safety_full_pipeline.ipynb
│   └── configs/ & tests/
├── assets/              # Figures, generation scripts, and dataset
│   ├── pipeline_figures/       # 7 publication-quality PDF figures
│   ├── generate_pipeline_pdfs.py
│   ├── fig1.jpg, fig1.png, etc.
│   └── data/
│       └── DASIG/              # 60 subjects MIMU & Arduino CSV files
└── output/              # Trained models, confusion matrices, metrics
    ├── models/                 # PyTorch checkpoints (*.pth):
    │   ├── ConvNeXt_1D_best.pth
    │   ├── TCN_best.pth
    │   ├── Transformer_1D_best.pth
    │   ├── InceptionTime_best.pth
    │   └── ...
    ├── figures/                # Confusion matrices (*_cm.png), boxplots, comparisons
    ├── metrics/cobot_safety/   # Performance logs and JSON metrics
    ├── Final_Models_Summary.md
    └── Pipeline_Architecture_Workflow.md / .pdf
```

## How to Run

### Run Classical ML Pipeline
```bash
cd src
python3 run_pipeline.py --quick
```

### Run 1D Deep Learning Pipeline
```bash
cd src
python3 test_v4.py
```

### Re-generate Figures
```bash
cd assets
python3 generate_pipeline_pdfs.py
```

### Compile Paper
```bash
cd paper/cobot_safety_cas
pdflatex cobot_safety_paper.tex
bibtex cobot_safety_paper
pdflatex cobot_safety_paper.tex
```
