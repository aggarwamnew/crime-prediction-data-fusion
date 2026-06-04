"""Download Santander Cycles trip data from TfL for thesis period (Jan 2023 - Dec 2025).

Since TfL doesn't expose a directory listing, we construct URLs from known patterns.
The browser found that files transition from weekly (2023) to semi-monthly (mid-2023 onward).

Strategy: try numbered files 350-434, inferring filenames from a known index page.
Since we can't scrape the listing, we'll use the TfL open data API instead.
"""
import os
import json
import re
import requests
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_DIR = Path("data/raw/london/transport/santander_cycles/trips")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# From browser research, we know the URL pattern and some key files.
# Let's try to get the cycling-load.json manifest first.
MANIFEST_URLS = [
    "https://cycling.data.tfl.gov.uk/cycling-load.json",
    "https://cycling.data.tfl.gov.uk/usage-stats/cycling-load.json",
]

BASE_URL = "https://cycling.data.tfl.gov.uk/usage-stats/"


def try_manifest():
    """Try to get file listing from manifest JSON."""
    for url in MANIFEST_URLS:
        print(f"Trying manifest: {url}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  Found manifest with {len(data)} entries")
                return data
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def generate_candidate_urls():
    """Generate candidate filenames based on known patterns from browser research.
    
    Known patterns:
    - Weekly (early 2023): {NUM}JourneyDataExtract{DDMonYYYY}-{DDMonYYYY}.csv
    - Semi-monthly (mid-2023+): {NUM}JourneyDataExtract{DDMonYYYY}-{DDMonYYYY}.csv
    
    We know file 350 = ~26Dec2022-01Jan2023 and file 434 = 16Dec2025-31Dec2025.
    """
    # Generate all dates in our range and try various filename patterns
    candidates = []
    
    # We know some specific files from browser:
    known_files = [
        "350JourneyDataExtract26Dec2022-01Jan2023.csv",
        "351JourneyDataExtract02Jan2023-08Jan2023.csv",
        "352JourneyDataExtract09Jan2023-15Jan2023.csv",
        "353JourneyDataExtract16Jan2023-22Jan2023.csv",
        "354JourneyDataExtract23Jan2023-29Jan2023.csv",
        "355JourneyDataExtract30Jan2023-05Feb2023.csv",
        "411JourneyDataExtract01Jan2025-14Jan2025.csv",
        "412JourneyDataExtract15Jan2025-31Jan2025.csv",
        "434JourneyDataExtract16Dec2025-31Dec2025.csv",
    ]
    
    return known_files


def probe_and_download():
    """Since we can't get a full listing, probe URLs with numbered patterns."""
    
    # The most robust approach: for each number 350-434, try a few URL patterns
    # and download whatever succeeds.
    
    # First, let's check if we can access the known files
    test_url = BASE_URL + "351JourneyDataExtract02Jan2023-08Jan2023.csv"
    print(f"Testing access: {test_url}")
    try:
        resp = requests.head(test_url, timeout=10)
        print(f"  Status: {resp.status_code}")
        if resp.status_code != 200:
            print("  TfL may be blocking requests. Try browser download.")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Try manifest first
    manifest = try_manifest()
    
    if manifest:
        # Filter and download from manifest
        print("Using manifest for download...")
    else:
        print("\nNo manifest found. Testing direct URL access...")
        can_access = probe_and_download()
        
        if not can_access:
            print("\n=== MANUAL DOWNLOAD REQUIRED ===")
            print("TfL blocks automated downloads for Santander Cycles data.")
            print("Please download manually from:")
            print("  https://cycling.data.tfl.gov.uk/usage-stats/")
            print(f"  Save files to: {OUTPUT_DIR.absolute()}")
            print("  Files needed: numbered 350-434 (JourneyDataExtract*.csv)")
            print("\nAlternatively, use the browser to download.")
