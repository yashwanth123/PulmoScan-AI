"""Copy sample X-ray/CT images from the local dataset for offline demo use."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_training.dataset import download_dataset, get_dataset_root

OUT = ROOT / "frontend" / "assets" / "samples"

# Map sample id → (source subfolder, filename pattern)
SAMPLES = {
    "chest_xray/xray_normal": ("test/NORMAL", "NORMAL(1).jpg"),
    "chest_xray/xray_pneumonia": ("test/NORMAL", "NORMAL(2).jpg"),  # fallback; COVID for pneumonia demo
    "chest_xray/xray_covid": ("test/COVID19", "COVID19(1).jpg"),
    "ct_scan/ct_chest": ("test/NORMAL", "NORMAL(10).jpg"),
    "ct_scan/ct_lung": ("test/NORMAL", "NORMAL(20).jpg"),
}


def find_file(base: Path, subpath: str, preferred: str) -> Path | None:
    folder = base / subpath
    if not folder.is_dir():
        return None
    preferred_path = folder / preferred
    if preferred_path.exists():
        return preferred_path
    files = sorted(folder.glob("*.jpg"))
    return files[0] if files else None


def main() -> None:
    data_root = download_dataset()
    copied = 0

    for sample_key, (subpath, fname) in SAMPLES.items():
        scan_type, sample_id = sample_key.split("/")
        src = find_file(data_root, subpath, fname)
        if src is None:
            print(f"Skip {sample_key}: source not found")
            continue
        dest_dir = OUT / scan_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sample_id}.jpg"
        shutil.copy2(src, dest)
        print(f"Copied {src.name} → {dest}")
        copied += 1

    # Fix pneumonia sample — use COVID image labeled for demo if available
    covid_src = find_file(data_root, "test/COVID19", "COVID19(1).jpg")
    if covid_src:
        dest = OUT / "chest_xray" / "xray_covid.jpg"
        shutil.copy2(covid_src, dest)
        # Also use as pneumonia placeholder for label-check demos
        shutil.copy2(covid_src, OUT / "chest_xray" / "xray_pneumonia.jpg")
        print(f"COVID sample → xray_covid.jpg & xray_pneumonia.jpg")

    print(f"\nDone — {copied} samples in {OUT}")
    print("Restart server: python3 run.py")


if __name__ == "__main__":
    main()
