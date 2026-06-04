"""
Data ingestion utilities.

Functions for loading raw data files into the project pipeline.
"""

import zipfile
import shutil
from pathlib import Path
from src.utils.config import PATHS


def unzip_crime_data(zip_path: str | Path, force: str = "metropolitan") -> Path:
    """
    Extract crime data from the data.police.uk ZIP download.

    The ZIP contains folders like:
        2024-01/metropolitan-street.csv
        2024-01/city-of-london-street.csv
        ...

    This function extracts all CSVs for the specified force(s) into
    data/raw/london/crime/.

    Args:
        zip_path: Path to the downloaded ZIP file.
        force: Force identifier prefix (e.g., 'metropolitan', 'city-of-london').
               Use 'london' to extract both Met + City of London.

    Returns:
        Path to the output directory.
    """
    zip_path = Path(zip_path)
    output_dir = PATHS["raw_london_crime"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define which force prefixes to include
    if force == "london":
        prefixes = ("metropolitan", "city-of-london")
    else:
        prefixes = (force,)

    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Skip directories and non-CSV files
            if member.endswith("/") or not member.endswith(".csv"):
                continue

            filename = Path(member).name

            # Check if this file belongs to our target force(s)
            # Filenames are like: 2023-02-metropolitan-street.csv
            if any(prefix in filename for prefix in prefixes):
                # Extract directly into output_dir (flat structure)
                target_path = output_dir / filename
                with zf.open(member) as source, \
                     open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_count += 1

    print(f"✅ Extracted {extracted_count} CSV files to {output_dir}")
    return output_dir


def list_crime_csvs(crime_dir: Path = None) -> list[Path]:
    """List all crime CSVs in the raw data directory."""
    crime_dir = crime_dir or PATHS["raw_london_crime"]
    return sorted(crime_dir.glob("*.csv"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.ingest <path_to_zip>")
        sys.exit(1)

    zip_file = sys.argv[1]
    print(f"📦 Extracting crime data from: {zip_file}")
    output = unzip_crime_data(zip_file, force="london")
    csvs = list_crime_csvs(output)
    print(f"\n📁 Files extracted:")
    for csv in csvs[:10]:
        print(f"   {csv.name}")
    if len(csvs) > 10:
        print(f"   ... and {len(csvs) - 10} more")
