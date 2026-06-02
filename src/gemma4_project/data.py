#!/usr/bin/env python3
"""Download and prepare project datasets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


DEFAULT_ROOT = Path(os.environ.get("GEMMA4_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_REPO_ID = "mario-rc/aif-emotional-generation"
SOURCE_FILES = (
    "dialogues/train.json",
    "dialogues/test.json",
    "aif_annotations/train.json",
)
TARGET_FILES = (
    "sft_demonstration_dataset.json",
    "sft_demonstration_dataset_test.json",
    "dpo_preference_dataset.json",
    "ppo_unlabeled_prompts_dataset_test.json",
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}")
    return data


def write_json(path: Path, data: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path} exists. Use --force to overwrite.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[WRITE] {path} ({len(data)} rows)")


def copy_json(path: Path, source_path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path} exists. Use --force to overwrite.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, path)
    print(f"[WRITE] {path}")


def download_source(repo_id: str, filename: str, revision: str | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
    )


def filter_by_set(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("set") == split_name]


def write_manifest(root: Path, repo_id: str, revision: str | None, counts: dict[str, int]) -> None:
    manifest = {
        "repo_id": repo_id,
        "revision": revision or "default",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": list(SOURCE_FILES),
        "targets": counts,
        "derivations": {
            "sft_demonstration_dataset.json": "dialogues/train.json filtered where set == sft-demonstration",
            "sft_demonstration_dataset_test.json": "dialogues/test.json filtered where set == sft-demonstration",
            "dpo_preference_dataset.json": "aif_annotations/train.json",
            "ppo_unlabeled_prompts_dataset_test.json": "dialogues/test.json",
        },
    }
    path = root / "datasets" / "source_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[WRITE] {path}")


def prepare_datasets(root: Path, repo_id: str, revision: str | None, force: bool) -> dict[str, int]:
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dialogues_train_path = download_source(repo_id, "dialogues/train.json", revision)
    dialogues_test_path = download_source(repo_id, "dialogues/test.json", revision)
    dpo_train_path = download_source(repo_id, "aif_annotations/train.json", revision)

    dialogues_train = load_json(dialogues_train_path)
    dialogues_test = load_json(dialogues_test_path)

    sft_train = filter_by_set(dialogues_train, "sft-demonstration")
    sft_test = filter_by_set(dialogues_test, "sft-demonstration")

    targets = {
        "sft_demonstration_dataset.json": len(sft_train),
        "sft_demonstration_dataset_test.json": len(sft_test),
        "dpo_preference_dataset.json": len(load_json(dpo_train_path)),
        "ppo_unlabeled_prompts_dataset_test.json": len(dialogues_test),
    }

    write_json(datasets_dir / "sft_demonstration_dataset.json", sft_train, force=force)
    write_json(datasets_dir / "sft_demonstration_dataset_test.json", sft_test, force=force)
    copy_json(datasets_dir / "dpo_preference_dataset.json", dpo_train_path, force=force)
    copy_json(datasets_dir / "ppo_unlabeled_prompts_dataset_test.json", dialogues_test_path, force=force)
    write_manifest(root, repo_id, revision, targets)

    return targets


def verify_datasets(root: Path) -> int:
    missing = []
    for filename in TARGET_FILES:
        path = root / "datasets" / filename
        if not path.exists():
            missing.append(str(path))
            continue
        print(f"[OK] {path}: {len(load_json(path))} rows")

    if missing:
        for path in missing:
            print(f"[MISSING] {path}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset JSON files.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify expected local dataset files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return verify_datasets(args.root)

    counts = prepare_datasets(args.root, args.repo_id, args.revision, args.force)
    print("[DONE] Prepared datasets from Hugging Face:")
    for filename, count in counts.items():
        print(f"  {filename}: {count} rows")
    return verify_datasets(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
