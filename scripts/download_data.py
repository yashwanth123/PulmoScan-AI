#!/usr/bin/env python3
"""Download COVID-19 X-ray dataset for training."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"
REPO = "https://github.com/education454/datasets.git"

if __name__ == "__main__":
    if (DATA_DIR / "Data").exists():
        print(f"Dataset already exists at {DATA_DIR}")
        sys.exit(0)

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning dataset to {DATA_DIR} (~1.3 GB)...")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(DATA_DIR)], check=True)
    print("Done.")
