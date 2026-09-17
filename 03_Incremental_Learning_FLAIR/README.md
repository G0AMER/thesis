# 03 - Combined Incremental Learning for Human-Robot Collaboration (FLAIR)

**Paper Title:** *A Combined Incremental Learning Algorithm for Human-Robot Collaboration*  
**Conference:** 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)  
**Authors:** Ameur Gargouri, Mohamed Karray, Mohamed Ksantini  
**Dataset:** HARMONIC (24 subjects, multi-modal shared autonomy trajectories)

---

## Directory Structure

```
03_Incremental_Learning_FLAIR/
├── paper/               # KES 2026 manuscript & LaTeX source
│   ├── flair_paper.tex
│   ├── flair_paper.pdf
│   ├── references.bib
│   ├── PROCS_KES2026_LATEX_Template/
│   └── paper_build.sh
├── src/                 # FLAIR algorithm, 8-layer architecture, experimental notebooks
│   ├── src/             # Incremental learning core modules:
│   │   ├── incremental_learning/   # Base ContinualLearner, DER++, EWC, Adapters, FLAIR
│   │   ├── irl/                    # Inverse Reinforcement Learning
│   │   ├── behavior_generation/    # CVAE, DMP, temporal scheduling
│   │   ├── style_modeling/         # Operator style clustering
│   │   ├── perception/             # Intent recognition
│   │   ├── safety/                 # ISO/TS 15066 safety filters
│   │   └── metrics/                # Fluidity, alignment, R^2 metrics
│   ├── notebooks/       # 01-10 continual learning & shared autonomy notebooks
│   ├── scripts/         # Report generation and notebook creation scripts
│   ├── configs/         # Preprocessing YAML configs
│   ├── tests/           # Unit tests
│   ├── data/            # Preprocessed HARMONIC sequential data
│   ├── pyproject.toml / requirements.txt
│   └── ARCHITECTURE.md  # 8-layer architecture documentation
├── assets/              # Architecture diagrams & presentations
│   ├── flair_architecture.pdf / .png
│   ├── generate_architecture.py
│   ├── r2_evolution.png
│   └── presentation/    # KES 2026 presentation slides & assets
└── output/              # Experimental outputs & reports
    ├── experiments/     # exp01 to exp07 (FLAIR benchmarks, ablation studies)
    ├── reports/         # Exploration and benchmark markdown/docx reports
    └── reference_report_hrc.*
```

## How to Run

### Run Experiment Notebooks
```bash
cd src/notebooks
# Open Jupyter and run 08_der_sa_experiment.ipynb or 10_final_benchmarking.ipynb
jupyter notebook
```

### Compile Paper
```bash
cd paper
bash paper_build.sh
# or:
pdflatex flair_paper.tex
bibtex flair_paper
pdflatex flair_paper.tex
```
