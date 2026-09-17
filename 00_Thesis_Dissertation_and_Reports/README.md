# 00 - Thesis Dissertation and Progress Reports

**PhD Title:** *Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale*  
**Author:** Ameur Gargouri  
**Laboratories:** ATISP Laboratory (ENET'Com, University of Sfax, Tunisia) & ESME Research Lab (France)  
**Supervisors:** Mohamed Karray (ESME) & Mohamed Ksantini (University of Sfax)

---

## Directory Structure

```
00_Thesis_Dissertation_and_Reports/
├── paper/               # Thesis dissertation drafts, progress reports, LaTeX project
│   ├── latex_project/   # Complete LaTeX report (chapters 01 to 07, compiled to main.pdf)
│   ├── 1st_research_thesis_31Jan2026.docx / .pdf
│   ├── 2nd_research_thesis_02Fév2026.docx / .pdf
│   ├── 3rd_research_thesis_18Fév2026.docx / .pdf
│   ├── 4th_research_thesis_20Avr2026 .docx / .pdf
│   ├── 5th_research_thesis_20Avr2026.docx
│   ├── 6th_research_thesis_23Avr2026.docx
│   ├── Avancement.docx / avancement.pdf
│   ├── these.docx / thesis.pdf
│   └── rapport_these_complet.md / rapport_these_complet.pdf
├── assets/              # Presentation slides, literature surveys, templates
│   ├── thesis_presentation.md / .html  # Defense & review slide deck
│   ├── Literature_Contributions.md
│   ├── SENSOR_RESEARCH_COMPREHENSIVE.md / .pdf
│   ├── project_info__1.md / project_info__2.md
│   └── templates/       # IEEE and conference template baselines
├── output/              # Compilation logs and synthesis outputs
│   └── report_output.log
└── src/                 # Literature extraction scripts & preliminary explorations
    ├── get_papers.py
    ├── openalex_*.json
    ├── shoplifting/     # Side computer vision exploration
    └── dataset-ninja/   # CORe50 exploratory dataset
```

## How to Build the LaTeX Progress Report

```bash
cd paper/latex_project
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
