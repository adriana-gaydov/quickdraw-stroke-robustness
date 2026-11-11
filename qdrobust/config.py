"""Shared constants and default hyper-parameters (values from the proposal)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# QuickDraw simplified drawings (CC BY 4.0).
SOURCE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"

CATEGORIES = [
    "airplane", "apple", "bicycle", "bird", "car",
    "cat", "chair", "dog", "fish", "flower",
]
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
NUM_CLASSES = len(CATEGORIES)

# Originals per category and split.
PER_CATEGORY = {"train": 4000, "val": 500, "test": 500}
MAX_POINTS = 512          # originals longer than this are excluded
COORD_MAX = 255

# Identifier-based split: bucket = MD5(key_id) mod 10.
SPLIT_BUCKETS = {"train": range(0, 8), "val": range(8, 9), "test": range(9, 10)}

# Training-time augmentation.
TRAIN_Q = 0.25            # per-eligible-point selection probability
MIX_PROB = 0.5            # probability of using the transformed version

# Evaluation.
EVAL_QS = [0.10, 0.25, 0.50, 1.00]
EVAL_REPLICATES = 5
EVAL_MASK_SEED = 413
POINT_BINS = [(1, 32), (33, 64), (65, 128), (129, 512)]
STROKE_BINS = [(1, 3), (4, 7), (8, None)]

# Model / optimisation defaults (pilot tuning may override lr and dropout for the LSTM).
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
GRAD_CLIP = 1.0
MAX_EPOCHS = 50
PATIENCE = 8
SEEDS = [16, 18, 20]

PILOT_LRS = [3e-4, 1e-3]
PILOT_DROPOUTS = [0.10, 0.20]

CONDITIONS = ["ordinary", "duplication", "boundary", "canonical", "cnn"]
