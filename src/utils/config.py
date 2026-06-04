"""
Project-wide configuration and path management.

Usage:
    from src.utils.config import PATHS, DB_PATH
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Thesis/

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    # Raw data (never mutate these files)
    "raw_london_crime":      DATA_DIR / "raw" / "london" / "crime",
    "raw_london_boundaries": DATA_DIR / "raw" / "london" / "boundaries",
    "raw_london_imd":        DATA_DIR / "raw" / "london" / "imd",
    "raw_london_census":     DATA_DIR / "raw" / "london" / "census",
    "raw_london_weather":    DATA_DIR / "raw" / "london" / "weather",

    "raw_vancouver_crime":      DATA_DIR / "raw" / "vancouver" / "crime",
    "raw_vancouver_boundaries": DATA_DIR / "raw" / "vancouver" / "boundaries",

    # Processed data
    "processed_london":    DATA_DIR / "processed" / "london",
    "processed_vancouver": DATA_DIR / "processed" / "vancouver",

    # External data
    "external": DATA_DIR / "external",
}

# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "processed" / "london" / "thesis.duckdb"

# ---------------------------------------------------------------------------
# Visualization outputs
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Notebook directories
# ---------------------------------------------------------------------------
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Ensure key directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all project directories if they don't exist."""
    for p in PATHS.values():
        p.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ All project directories created.")
    for name, path in PATHS.items():
        print(f"  {name}: {path}")
