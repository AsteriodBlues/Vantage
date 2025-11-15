"""
Configuration settings for the Vantage F1 analysis project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
START_YEAR = 2018
END_YEAR = 2024

# Collection settings
INCLUDE_SPRINT_RACES = False  # Focus on main races only
RETRY_FAILED_SESSIONS = True
MAX_RETRIES = 3
