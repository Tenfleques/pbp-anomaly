#!/usr/bin/env python3
"""Download public datasets for the Nature environmental anomaly paper.

Small datasets already shipped in data/:
  - AirQualityUCI.csv
  - open_meteo_cape_town.csv
  - open_meteo_sao_paulo.csv

This script downloads the larger ones:
  - Beijing Multi-Site Air Quality (UCI, ~62MB ZIP)
  - 4 NOAA ISD weather stations (ORD/MIA/SFO/FAI, 2022-2023)
  - Optionally: EPA AQS hourly data (--include-epa, ~2GB per parameter)

Usage:
    python experiments/download_data.py              # Beijing + NOAA
    python experiments/download_data.py --include-epa  # Also EPA AQS
    python experiments/download_data.py --quick        # Verify shipped data only
"""

import argparse
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ---------------------------------------------------------------------------
# Beijing Multi-Site
# ---------------------------------------------------------------------------
BEIJING_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)
BEIJING_DIR = os.path.join(DATA_DIR, "beijing_multisite")

# ---------------------------------------------------------------------------
# NOAA ISD stations (2022-2023)
# ---------------------------------------------------------------------------
NOAA_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access"
NOAA_STATIONS = {
    "ord": "72530014819",  # Chicago O'Hare
    "mia": "72202012839",  # Miami
    "sfo": "72494023234",  # San Francisco
    "fai": "70261026411",  # Fairbanks
}
NOAA_YEARS = [2022, 2023]

# ---------------------------------------------------------------------------
# EPA AQS (optional)
# ---------------------------------------------------------------------------
EPA_BASE = "https://aqs.epa.gov/aqsweb/airdata"
EPA_PARAMS = {
    "44201": "O3",
    "42401": "SO2",
    "42101": "CO",
    "42602": "NO2",
    "88101": "PM2.5",
}
EPA_DEFAULT_YEAR = 2023


def _progress_hook(block_num, block_size, total_size):
    """Simple progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
    else:
        mb = downloaded / 1e6
        sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
    sys.stdout.flush()


def _download(url, dest):
    """Download url to dest with progress, skipping if dest exists."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  SKIP (exists, {size_mb:.1f} MB): {os.path.basename(dest)}")
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"\n  Done: {os.path.basename(dest)} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_beijing():
    """Download and extract Beijing Multi-Site Air Quality dataset."""
    print("\n=== Beijing Multi-Site Air Quality (UCI) ===")

    # Check if already extracted
    if os.path.isdir(BEIJING_DIR) and any(
        f.endswith(".csv") for f in os.listdir(BEIJING_DIR)
    ):
        n_csv = sum(1 for f in os.listdir(BEIJING_DIR) if f.endswith(".csv"))
        print(f"  SKIP (already extracted, {n_csv} CSVs): {BEIJING_DIR}")
        return

    zip_path = os.path.join(DATA_DIR, "beijing_multisite.zip")
    if not _download(BEIJING_URL, zip_path):
        if not os.path.exists(zip_path):
            return

    # Extract
    print(f"  Extracting to {BEIJING_DIR} ...")
    os.makedirs(BEIJING_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Flatten directory structure: extract CSVs to beijing_multisite/
            basename = os.path.basename(member)
            if basename and basename.endswith(".csv"):
                target = os.path.join(BEIJING_DIR, basename)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"    {basename}")

    n_csv = sum(1 for f in os.listdir(BEIJING_DIR) if f.endswith(".csv"))
    print(f"  Extracted {n_csv} CSV files")

    # Clean up ZIP
    os.remove(zip_path)
    print("  Removed ZIP archive")


def download_noaa():
    """Download NOAA ISD hourly data for 4 stations, 2022-2023."""
    print("\n=== NOAA ISD Hourly Weather Data ===")

    for prefix, station_id in NOAA_STATIONS.items():
        station_dir = os.path.join(DATA_DIR, f"noaa_isd_{prefix}")
        os.makedirs(station_dir, exist_ok=True)
        print(f"\n  Station: {prefix.upper()} ({station_id})")

        for year in NOAA_YEARS:
            url = f"{NOAA_BASE}/{year}/{station_id}.csv"
            dest = os.path.join(station_dir, f"{prefix}_{year}.csv")
            _download(url, dest)


def download_epa(year=EPA_DEFAULT_YEAR):
    """Download EPA AQS hourly data for 5 pollutants."""
    print(f"\n=== EPA AQS Hourly Data ({year}) ===")

    epa_dir = os.path.join(DATA_DIR, f"epa_aqs_{year}")
    os.makedirs(epa_dir, exist_ok=True)

    for code, name in EPA_PARAMS.items():
        zip_name = f"hourly_{code}_{year}.zip"
        url = f"{EPA_BASE}/{zip_name}"
        zip_path = os.path.join(epa_dir, zip_name)
        csv_name = f"hourly_{code}_{year}.csv"
        csv_path = os.path.join(epa_dir, csv_name)

        # Skip if CSV already extracted
        if os.path.exists(csv_path):
            size_mb = os.path.getsize(csv_path) / 1e6
            print(f"\n  SKIP {name} ({code}): CSV exists ({size_mb:.1f} MB)")
            continue

        print(f"\n  Parameter: {name} ({code})")
        if not _download(url, zip_path):
            if not os.path.exists(zip_path):
                continue

        # Extract
        print(f"  Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(epa_dir)

        if os.path.exists(csv_path):
            size_mb = os.path.getsize(csv_path) / 1e6
            print(f"  Extracted: {csv_name} ({size_mb:.1f} MB)")

        os.remove(zip_path)
        print("  Removed ZIP archive")


def verify_shipped():
    """Verify that shipped (small) datasets exist."""
    print("\n=== Verifying shipped datasets ===")
    shipped = [
        "AirQualityUCI.csv",
        "open_meteo_cape_town.csv",
        "open_meteo_sao_paulo.csv",
    ]
    all_ok = True
    for name in shipped:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1e3
            print(f"  OK   {name} ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING  {name}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for the Nature environmental anomaly paper."
    )
    parser.add_argument(
        "--include-epa",
        action="store_true",
        help="Also download EPA AQS hourly data (~2GB per parameter)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only verify that shipped datasets exist (no downloads)",
    )
    parser.add_argument(
        "--epa-year",
        type=int,
        default=EPA_DEFAULT_YEAR,
        help=f"Year for EPA AQS data (default: {EPA_DEFAULT_YEAR})",
    )
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    ok = verify_shipped()

    if args.quick:
        sys.exit(0 if ok else 1)

    download_beijing()
    download_noaa()

    if args.include_epa:
        download_epa(year=args.epa_year)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
