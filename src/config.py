import os
import numpy as np
import random

# Paths
COLAB_ROOT = '/content'
DRIVE_ROOT = '/content/drive/MyDrive/NoiseClassification'
DATASET_ROOT = os.path.join(DRIVE_ROOT, 'datasets')
FEATURES_ROOT = os.path.join(DATASET_ROOT, 'features')
MODELS_ROOT = os.path.join(DRIVE_ROOT, 'models')
REPORTS_ROOT = os.path.join(DRIVE_ROOT, 'reports')
FIGURES_ROOT = os.path.join(DRIVE_ROOT, 'figures')

# Audio params
SR = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

# Export for scripts/notebooks
print(f"SR={SR}, Duration={DURATION}, N_MELS={N_MELS}")