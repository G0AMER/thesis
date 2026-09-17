#!/usr/bin/env bash
# Finalize + compile the rebuilt paper (task_type_detection_paper_v3.tex).
# 1) regenerate LaTeX tables from the sealed result CSVs
# 2) copy the publication figures into paper/figures/
# 3) latexmk the paper
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/3] generating tables from v3_rebuild results..."
python3 src/rebuild/make_latex_tables.py

echo "[2/3] copying figures..."
FIGSRC="$ROOT/output/research_outputs/fusion_training/v3_rebuild/figures"
FIGDST="$ROOT/paper/figures"
mkdir -p "$FIGDST"
cp -f "$FIGSRC"/fig*.png "$FIGSRC"/fig*.pdf "$FIGDST"/ 2>/dev/null || true
cp -f "$FIGSRC/task_pipeline.png" "$FIGDST"/ 2>/dev/null || true
ls -1 "$FIGDST" | sed 's/^/    /'

echo "[3/3] compiling..."
cd "$ROOT/paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error task_type_detection_paper_v3.tex
echo "OK -> paper/task_type_detection_paper_v3.pdf"
