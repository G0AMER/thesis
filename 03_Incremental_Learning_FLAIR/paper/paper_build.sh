#!/usr/bin/env bash
set -euo pipefail
pdflatex -interaction=nonstopmode -halt-on-error task_type_detection_paper.tex
bibtex task_type_detection_paper
pdflatex -interaction=nonstopmode -halt-on-error task_type_detection_paper.tex
pdflatex -interaction=nonstopmode -halt-on-error task_type_detection_paper.tex
