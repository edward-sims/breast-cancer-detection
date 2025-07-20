import torch
import datetime
import os

# Set img dimension constants for medical imaging
ROWS = 224  # ViT standard input size
COLS = 224
CHANNELS = 3

# Set seed for reproducibility
SEED = 42

# Device constants - optimized for your hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WORKERS = min(4, os.cpu_count() or 1)  # Conservative for your 6-core system

# Set test size
TEST_SIZE = 0.15

# Set number of splits for cross-validation
N_SPLITS = 5

# Batch sizes optimized for your memory
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16

# Verbosity
VERBOSE = True

# Model configuration
MODEL_CONFIG = {
    "backbone": "vit_base_patch16_224",  # Vision Transformer
    "pretrained": True,
    "num_classes": 1,  # Binary classification
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "stochastic_depth": 0.1,
}

# Multi-view fusion settings
MULTI_VIEW_CONFIG = {
    "fusion_method": "attention",  # attention, concat, weighted_avg
    "attention_heads": 8,
    "hidden_dim": 768,
}

# Training configuration
TRAINING_CONFIG = {
    "patience": 15,
    "min_epochs": 50,
    "max_epochs": 200,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "scheduler": "cosine_with_warmup",
}

# Optimizer & learning rate constants
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPSILON = 1e-8
AMSGRAD = True
BETAS = (0.9, 0.999)
ETA_MIN = 1e-6
T_MAX = 20
T_MULT = 2

# Mixed precision training
PRECISION = 16
GRADIENT_CLIP_VAL = 1.0

# Logging
LOG_DIR = "logs"
LOG_NAME = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_")[:19]

# Data augmentation
AUGMENTATION_CONFIG = {
    "rotation_limit": 15,
    "shift_limit": 0.1,
    "scale_limit": 0.1,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "p": 0.5,
}

# Evaluation metrics
METRICS = ["pf1", "auc", "precision", "recall", "f1"]

# Paths
DATA_DIR = "data"
MODEL_DIR = "models/saved"
RESULTS_DIR = "results"
