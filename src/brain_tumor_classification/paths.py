"""Shared project paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRAINING_DIR = DATA_DIR / "Training"
TESTING_DIR = DATA_DIR / "Testing"
DATASET_PATH = DATA_DIR / "brain_tumor_dataset.csv"
