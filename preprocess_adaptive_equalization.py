"""
Dataset preprocessing utility using adaptive histogram equalization (CLAHE).

This script applies a Roboflow-like adaptive equalization contrast adjustment
to all images in a dataset folder while preserving the folder structure.

Default behavior targets:
    input : blinkblink-6/
    output: blinkblink-6-preprocessed/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image files under root recursively."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def adaptive_equalize_bgr(image_bgr, clip_limit: float, tile_grid_size: int):
    """Apply CLAHE on luminance channel in LAB color space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_channel_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    clip_limit: float,
    tile_grid_size: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Preprocess all images and preserve relative directory structure."""
    processed = 0
    skipped = 0
    failed = 0

    for src_path in iter_images(input_dir):
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            print(f"[FAILED] Could not read: {src_path}")
            continue

        preprocessed = adaptive_equalize_bgr(
            image_bgr=image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

        ok = cv2.imwrite(str(dst_path), preprocessed)
        if not ok:
            failed += 1
            print(f"[FAILED] Could not write: {dst_path}")
            continue

        processed += 1

    return processed, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Roboflow-like adaptive equalization (CLAHE) to a dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("blinkblink-6"),
        help="Input dataset directory (default: blinkblink-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blinkblink-6-preprocessed"),
        help="Output dataset directory (default: blinkblink-6-preprocessed)",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0)",
    )
    parser.add_argument(
        "--tile-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size N for N x N (default: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.tile_grid_size <= 0:
        raise SystemExit("--tile-grid-size must be > 0")
    if args.clip_limit <= 0:
        raise SystemExit("--clip-limit must be > 0")

    print("=" * 70)
    print("Adaptive Equalization Preprocessing (CLAHE)")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"CLAHE : clip_limit={args.clip_limit}, tile_grid_size={args.tile_grid_size}x{args.tile_grid_size}")
    print("=" * 70)

    processed, skipped, failed = preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid_size,
        overwrite=args.overwrite,
    )

    print("\nDone")
    print(f"Processed: {processed}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")


if __name__ == "__main__":
    main()
