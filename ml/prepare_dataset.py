"""Rebuild the pest image dataset with no duplicates and no split leakage.

The 1.x dataset shipped 3,001 training files and 500 test files, but only 661
and 443 of those were distinct images -- and 390 of the 443 distinct test
images (88%) were byte-identical copies of training images. Any accuracy
measured against that test set was measuring memorisation.

This script pools every image, drops exact duplicates by content hash, drops
images that appear under more than one class label, and writes a fresh
stratified train/validation/test split under ``Data/clean/``.

Usage::

    python -m ml.prepare_dataset [--seed 42] [--val-frac 0.15] [--test-frac 0.15]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (PROJECT_ROOT / "Data" / "train", PROJECT_ROOT / "Data" / "test")
OUTPUT_ROOT = PROJECT_ROOT / "Data" / "clean"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _hash_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_unique_images() -> tuple[dict[str, dict[str, Path]], dict[str, set[str]]]:
    """Pool every source image and index it by content hash.

    Returns a ``{label: {hash: path}}`` mapping of images with an unambiguous
    label, plus a ``{hash: {labels}}`` mapping of every hash seen, so the
    caller can report how much was thrown away.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, Path] = {}

    for source in SOURCE_DIRS:
        if not source.is_dir():
            continue
        for image in sorted(source.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                continue
            label = image.parent.name.replace(" ", "_").lower()
            file_hash = _hash_file(image)
            seen[file_hash].add(label)
            paths.setdefault(file_hash, image)

    by_label: dict[str, dict[str, Path]] = defaultdict(dict)
    for file_hash, labels in seen.items():
        if len(labels) == 1:  # ambiguous images are label noise -- drop them
            by_label[next(iter(labels))][file_hash] = paths[file_hash]

    return by_label, seen


def split_label(
    hashes: list[str], val_frac: float, test_frac: float, rng: random.Random
) -> dict[str, list[str]]:
    """Split one class's hashes into train/val/test, guaranteeing >=1 test image."""
    shuffled = hashes[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_test = max(1, round(total * test_frac))
    n_val = max(1, round(total * val_frac))
    if n_test + n_val >= total:  # tiny class -- keep at least one training image
        n_test, n_val = 1, 1
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_label, seen = collect_unique_images()

    source_files = sum(
        1
        for source in SOURCE_DIRS
        if source.is_dir()
        for p in source.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    )
    ambiguous = sum(1 for labels in seen.values() if len(labels) > 1)

    print(f"source files            : {source_files}")
    print(f"distinct images         : {len(seen)}")
    print(f"dropped as exact dupes  : {source_files - len(seen)}")
    print(f"dropped as label-clashes: {ambiguous}")
    print(f"usable images           : {sum(len(v) for v in by_label.values())}")
    print()

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    manifest: dict[str, dict[str, list[str]]] = {}
    totals: dict[str, int] = defaultdict(int)

    for label in sorted(by_label):
        splits = split_label(sorted(by_label[label]), args.val_frac, args.test_frac, rng)
        manifest[label] = splits
        for split_name, hashes in splits.items():
            destination = OUTPUT_ROOT / split_name / label
            destination.mkdir(parents=True, exist_ok=True)
            for file_hash in hashes:
                source_path = by_label[label][file_hash]
                shutil.copy2(
                    source_path, destination / f"{file_hash[:16]}{source_path.suffix.lower()}"
                )
            totals[split_name] += len(hashes)
        print(
            f"{label:14s} train={len(splits['train']):4d} "
            f"val={len(splits['val']):3d} test={len(splits['test']):3d}"
        )

    print()
    print(f"TOTAL train={totals['train']} val={totals['val']} test={totals['test']}")

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {"seed": args.seed, "labels": sorted(manifest), "splits": manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUTPUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
