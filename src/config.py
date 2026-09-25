"""
Central configuration for the PSL (Pakistan Sign Language) translator project.

Defines all filesystem paths and shared constants in one place, built with
pathlib relative to the project root so the code works on any machine
regardless of where the repo is checked out (including paths containing
spaces or parentheses).
"""

from pathlib import Path

# Project root (two levels up from this file: src/config.py -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CUSTOM_TEST_DIR = DATA_DIR / "custom_test"

# Processed landmarks file
LANDMARKS_CSV = PROCESSED_DATA_DIR / "landmarks.csv"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Reports and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# App
APP_DIR = PROJECT_ROOT / "app"

# Reproducibility
RANDOM_SEED = 42
