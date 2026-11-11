"""Download QuickDraw simplified files and build the filtered, de-duplicated splits.

Usage:
    python scripts/prepare_data.py --download
Outputs (git-ignored): data/raw/*.ndjson, data/processed/{train,val,test}.ndjson
Outputs (shareable):  splits/{train,val,test}_ids.csv, splits/filter_counts.json
"""

import argparse
from pathlib import Path

import _path  # noqa: F401
from qdrobust import config
from qdrobust.data import build_dataset, download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="download raw files first")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR,
                    help="folder with <category>.ndjson files (default: data/raw)")
    args = ap.parse_args()
    if args.download or args.force_download:
        download(raw_dir=args.raw_dir, force=args.force_download)
    selected, _ = build_dataset(raw_dir=args.raw_dir)
    for split, recs in selected.items():
        print(f"{split}: {len(recs)} drawings -> {config.PROCESSED_DIR / (split + '.ndjson')}")


if __name__ == "__main__":
    main()
