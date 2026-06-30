#!/bin/bash
# =============================================================================
# PhD Project — Install Dependencies
# Run from: /home/g0amer/Desktop/thesis/phd_project
# Usage:    bash scripts/install_deps.sh
# =============================================================================

set -e

VENV_DIR=".venv"
PYTHON="python3"

# ---------- 1. Create virtual environment (if not exists) ----------
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

echo ">>> Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

# ---------- 2. Upgrade pip, setuptools, wheel ----------
echo ""
echo "=========================================="
echo "  Step 1/7 — Upgrading pip & build tools"
echo "=========================================="
pip install --upgrade pip setuptools wheel

# ---------- 3. PyTorch (with CUDA) ----------
echo ""
echo "=========================================="
echo "  Step 2/7 — Installing PyTorch (CUDA 12.4)"
echo "=========================================="
echo "  This is the largest download (~2.5 GB)"
echo ""
pip install torch torchvision torchaudio 

# ---------- 4. Scientific computing & data ----------
echo ""
echo "=========================================="
echo "  Step 3/7 — Scientific computing & data"
echo "=========================================="
pip install \
    numpy>=1.26.0 \
    scipy>=1.11.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.4.0 \
    h5py>=3.10.0 \
    PyYAML>=6.0.0 \
    jsonlines>=4.0.0

# ---------- 5. Vision, audio, signal processing ----------
echo ""
echo "=========================================="
echo "  Step 4/7 — Vision, audio, signals"
echo "=========================================="
pip install \
    opencv-python>=4.9.0 \
    mediapipe>=0.10.9 \
    Pillow>=10.0.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    mne>=1.6.0 \
    neurokit2>=0.2.7 \
    tslearn>=0.6.3

# ---------- 6. Deep learning utilities & clustering ----------
echo ""
echo "=========================================="
echo "  Step 5/7 — DL utilities & clustering"
echo "=========================================="
pip install \
    einops>=0.7.0 \
    timm>=0.9.0 \
    hdbscan>=0.8.33 \
    movement-primitives>=0.8.0

# ---------- 7. Visualization & experiment tracking ----------
echo ""
echo "=========================================="
echo "  Step 6/7 — Visualization & tracking"
echo "=========================================="
pip install \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    plotly>=5.18.0 \
    wandb>=0.16.0 \
    tensorboard>=2.15.0

# ---------- 8. Dev tools, Jupyter, utilities ----------
echo ""
echo "=========================================="
echo "  Step 7/7 — Dev tools & Jupyter"
echo "=========================================="
pip install \
    jupyter>=1.0.0 \
    ipykernel>=6.28.0 \
    ipywidgets>=8.1.0 \
    nbformat>=5.9.0 \
    tqdm>=4.66.0 \
    rich>=13.7.0 \
    loguru>=0.7.0 \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    python-dotenv>=1.0.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0

# ---------- 9. Register Jupyter kernel ----------
echo ""
echo "=========================================="
echo "  Registering Jupyter kernel"
echo "=========================================="
python -m ipykernel install --user --name phd_hrc --display-name "PhD HRC"

# ---------- 10. Verify ----------
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import numpy; print(f'NumPy {numpy.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
import cv2; print(f'OpenCV {cv2.__version__}')
import matplotlib; print(f'Matplotlib {matplotlib.__version__}')
print()
print('All packages installed successfully!')
"

echo ""
echo "=========================================="
echo "  DONE — Activate with:"
echo "  source .venv/bin/activate"
echo "=========================================="
