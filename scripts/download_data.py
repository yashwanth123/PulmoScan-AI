#!/usr/bin/env python3
"""Download the public COVID-19 chest X-ray dataset."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_training.dataset import download_dataset


def main() -> None:
    path = download_dataset()
    print(f"Dataset ready at: {path}")
    print(f"  Train: {path / 'train'}")
    print(f"  Test:  {path / 'test'}")


if __name__ == "__main__":
    main()
