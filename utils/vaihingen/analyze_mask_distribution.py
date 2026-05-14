#!/usr/bin/env python3
"""Analyze class distribution from Vaihingen mask tiles.

This script scans mask files recursively and reports:
- pixel count per class token (indexed id or RGB triplet),
- percentage of total pixels,
- how many images contain each class token.

Usage examples:
  python analyze_mask_distribution.py
  python analyze_mask_distribution.py --output-json dist.json --output-csv dist.csv
  python analyze_mask_distribution.py --max-files 100
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_MASKS_ROOT = "/scratch/matheuspimenta/Vaihingen/tiles/masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class distribution in Vaihingen masks")
    parser.add_argument(
        "--masks-root",
        type=str,
        default=DEFAULT_MASKS_ROOT,
        help="Root folder containing mask tiles (searched recursively).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Filename pattern for masks (default: *.png).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 means all).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def find_mask_files(root: Path, pattern: str, max_files: int) -> List[Path]:
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files


def unique_counts_indexed(arr: np.ndarray) -> Tuple[List[str], np.ndarray]:
    values, counts = np.unique(arr, return_counts=True)
    keys = [f"id:{int(v)}" for v in values]
    return keys, counts


def unique_counts_rgb(arr: np.ndarray) -> Tuple[List[str], np.ndarray]:
    pixels = arr[..., :3].reshape(-1, 3)
    values, counts = np.unique(pixels, axis=0, return_counts=True)
    keys = [f"rgb:{int(r)},{int(g)},{int(b)}" for r, g, b in values]
    return keys, counts


def extract_class_counts(mask_path: Path) -> Tuple[List[str], np.ndarray, Tuple[int, int]]:
    arr = np.array(Image.open(mask_path))
    height, width = int(arr.shape[0]), int(arr.shape[1])

    if arr.ndim == 2:
        return unique_counts_indexed(arr), (height, width)

    if arr.ndim == 3 and arr.shape[2] == 1:
        return unique_counts_indexed(arr[..., 0]), (height, width)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        return unique_counts_rgb(arr), (height, width)

    raise ValueError(f"Unsupported mask shape {arr.shape} at {mask_path}")


def normalize_rows(
    pixel_counts: Dict[str, int],
    image_counts: Dict[str, int],
    total_pixels: int,
    total_images: int,
) -> List[dict]:
    rows = []
    for key, pixels in sorted(pixel_counts.items(), key=lambda kv: kv[1], reverse=True):
        imgs = image_counts.get(key, 0)
        rows.append(
            {
                "class_token": key,
                "pixel_count": int(pixels),
                "pixel_pct": (100.0 * pixels / total_pixels) if total_pixels > 0 else 0.0,
                "image_count": int(imgs),
                "image_pct": (100.0 * imgs / total_images) if total_images > 0 else 0.0,
            }
        )
    return rows


def print_report(rows: Iterable[dict], total_images: int, total_pixels: int) -> None:
    print(f"Total images: {total_images}")
    print(f"Total pixels: {total_pixels}")
    print("\nClass distribution:")
    print("class_token,pixel_count,pixel_pct,image_count,image_pct")
    for row in rows:
        print(
            f"{row['class_token']},{row['pixel_count']},"
            f"{row['pixel_pct']:.4f},{row['image_count']},{row['image_pct']:.4f}"
        )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_token", "pixel_count", "pixel_pct", "image_count", "image_pct"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    masks_root = Path(args.masks_root)
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks root not found: {masks_root}")

    files = find_mask_files(masks_root, args.pattern, args.max_files)
    if not files:
        raise RuntimeError(f"No mask files found in {masks_root} with pattern {args.pattern}")

    pixel_counts: Dict[str, int] = defaultdict(int)
    image_counts: Dict[str, int] = defaultdict(int)
    total_pixels = 0

    for idx, mask_path in enumerate(files, start=1):
        (keys, counts), (height, width) = extract_class_counts(mask_path)
        total_pixels += height * width

        present = set()
        for key, count in zip(keys, counts):
            pixel_counts[key] += int(count)
            present.add(key)

        for key in present:
            image_counts[key] += 1

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(files)} files...")

    rows = normalize_rows(pixel_counts, image_counts, total_pixels, len(files))
    print_report(rows, len(files), total_pixels)

    payload = {
        "masks_root": str(masks_root),
        "pattern": args.pattern,
        "processed_files": len(files),
        "total_pixels": total_pixels,
        "distribution": rows,
    }

    if args.output_json:
        write_json(Path(args.output_json), payload)
        print(f"\nSaved JSON: {args.output_json}")

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"Saved CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
