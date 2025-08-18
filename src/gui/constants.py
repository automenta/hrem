import os

# --- Constants ---

# Experiment and Search Statuses
STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
STATUS_UNKNOWN = "Unknown"

# --- UI Constants ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
REFRESH_INTERVAL_MS = 2000
LAUNCH_DELAY_MS = 500
INITIAL_SPLITTER_SIZES = [300, 900]
CONFIGS_DIR = "configs"
BASE_MODELS_DIR = os.path.join(CONFIGS_DIR, "base", "models")
BASE_DATASETS_DIR = os.path.join(CONFIGS_DIR, "base", "datasets")
TRAINING_PROFILES_DIR = os.path.join(CONFIGS_DIR, "training_profiles")
RESULTS_DIR = "results"
RACES_DIR = os.path.join(RESULTS_DIR, ".races")
ARCHIVE_DIR = os.path.join(RESULTS_DIR, "archive")
