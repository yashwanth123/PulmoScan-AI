#!/usr/bin/env python3
"""Run inference on a lung scan from the command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.ml.predictor import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a lung scan image")
    parser.add_argument("image", type=Path, help="Path to X-ray/scan image")
    parser.add_argument(
        "--scan-type",
        default="chest_xray",
        choices=["chest_xray", "ct_scan"],
        help="Imaging modality",
    )
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    image_bytes = args.image.read_bytes()
    result = predict(
        image_bytes,
        scan_type=args.scan_type,
        include_gradcam=not args.no_gradcam,
    )

    payload = result.to_dict()
    if payload.get("gradcam_image"):
        payload["gradcam_image"] = "<base64 omitted>"

    print(json.dumps(payload, indent=2))

    if args.output:
        args.output.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull result saved to {args.output}")


if __name__ == "__main__":
    main()
