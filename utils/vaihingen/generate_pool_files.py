#!/usr/bin/env python3
"""Generate Vaihingen pool files with 10% class presence threshold.

RGB colors are mapped to class IDs 1-6 (0 reserved for background).
For each mask, a class is included if that color comprises ≥10% of the image.

Output: pool_files/ directory with 1.txt, 2.txt, ..., 6.txt, and querys.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
from PIL import Image


DEFAULT_MASKS_ROOT = "/scratch/matheuspimenta/Vaihingen/tiles/masks"
DEFAULT_OUTPUT_DIR = "/home/matheuspimenta/Jobs/SR/ifsl/utils/vaihingen/pool_files"

# RGB to class ID mapping (1-6, excluding 0 which is background)
RGB_TO_CLASS = {
    (255, 255, 255): 1,  # White
    (0, 0, 255): 2,      # Blue
    (0, 255, 0): 3,      # Green
    (0, 255, 255): 4,    # Cyan
    (255, 255, 0): 5,    # Yellow
    (255, 0, 0): 6,      # Red
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vaihingen pool files with 10% threshold")
    parser.add_argument(
        "--masks-root",
        type=str,
        default=DEFAULT_MASKS_ROOT,
        help="Root folder containing mask tiles (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for pool files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Percentage threshold for class inclusion (default: 10%).",
    )
    return parser.parse_args()


def find_mask_files(root: Path) -> list[Path]:
    """Find all PNG mask files recursively."""
    files = sorted(p for p in root.rglob("*.png") if p.is_file())
    return files


def extract_relative_path(mask_path: Path, masks_root: Path) -> str:
    """Get relative path from masks_root, e.g. 'top_mosaic_09cm_area16/part14.png'"""
    return str(mask_path.relative_to(masks_root))


def classify_mask(mask_path: Path, threshold: float) -> Set[int]:
    """
    Load mask and return set of class IDs present with ≥threshold% of pixels.
    
    Args:
        mask_path: path to PNG mask file
        threshold: percentage threshold (0-100)
    
    Returns:
        Set of class IDs (1-6) that meet the threshold
    """
    arr = np.array(Image.open(mask_path))
    
    # Handle different mask formats
    if arr.ndim == 2:
        # Indexed mask (not expected for Vaihingen, but handle it)
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.tile(arr, (1, 1, 3))
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported mask shape {arr.shape} at {mask_path}")
    
    total_pixels = arr.shape[0] * arr.shape[1]
    threshold_pixels = (threshold / 100.0) * total_pixels
    
    # Extract unique RGB triplets and their counts
    pixels = arr.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    classes = set()
    for color, count in zip(unique_colors, counts):
        rgb_tuple = tuple(int(c) for c in color)
        if rgb_tuple in RGB_TO_CLASS and count >= threshold_pixels:
            classes.add(RGB_TO_CLASS[rgb_tuple])
    
    return classes


def main() -> int:
    args = parse_args()
    masks_root = Path(args.masks_root)
    output_dir = Path(args.output_dir)
    
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks root not found: {masks_root}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all masks
    files = find_mask_files(masks_root)
    if not files:
        raise RuntimeError(f"No mask files found in {masks_root}")
    
    print(f"Found {len(files)} mask files")
    print(f"Using threshold: {args.threshold}%\n")
    
    # Build class pools
    class_pools: Dict[int, list[str]] = defaultdict(list)
    all_masks = []
    
    for idx, mask_path in enumerate(files, start=1):
        rel_path = extract_relative_path(mask_path, masks_root)
        all_masks.append(rel_path)
        
        classes = classify_mask(mask_path, args.threshold)
        for class_id in classes:
            class_pools[class_id].append(rel_path)
        
        if idx % 200 == 0:
            print(f"Processed {idx}/{len(files)} files...")
    
    # Write pool files
    for class_id in range(1, 7):
        if class_id in class_pools and class_pools[class_id]:
            pool_file = output_dir / f"{class_id}.txt"
            with pool_file.open("w") as f:
                for line in class_pools[class_id]:
                    f.write(line + "\n")
            count = len(class_pools[class_id])
            pct = 100.0 * count / len(files)
            print(f"Class {class_id}: {count} files ({pct:.1f}%)")
        else:
            print(f"Class {class_id}: 0 files")
    
    # Write querys.txt (all images)
    querys_file = output_dir / "querys.txt"
    with querys_file.open("w") as f:
        for line in all_masks:
            f.write(line + "\n")
    
    print(f"\nWrote pool files to {output_dir}")
    print(f"Query pool: {len(all_masks)} images")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
