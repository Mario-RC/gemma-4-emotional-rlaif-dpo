#!/usr/bin/env python3
"""Compare analyzed Gemma4 DPO results against the generated Gemma2 DPO reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .analysis import (
        DEFAULT_MODELS,
        DEFAULT_ROOT,
        REFERENCE_MODEL,
        default_dataset_for_run,
        result_filename,
        score_records,
        write_json,
    )
except ImportError:  # pragma: no cover - keeps direct file execution usable.
    from analysis import (
        DEFAULT_MODELS,
        DEFAULT_ROOT,
        REFERENCE_MODEL,
        default_dataset_for_run,
        result_filename,
        score_records,
        write_json,
    )


SUPPORTED_RUN = "dpo_3ep"


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}")
    return data


def prediction_candidates(preferred: str) -> list[str]:
    return [preferred, "predict_dpo_3", "predict_dpo_3ep"]


def pick_prediction_key(records: list[dict[str, Any]], preferred: str) -> str:
    if not records:
        return preferred
    first = records[0]
    for key in prediction_candidates(preferred):
        if key in first:
            return key
    raise KeyError(f"None of these prediction keys exist: {', '.join(prediction_candidates(preferred))}")


def result_path(root: Path, model: str, dataset: str, run_name: str) -> Path:
    return root / "saves" / model / "emotional_balanced" / result_filename(dataset, run_name)


def summarize(path: Path, model: str, preferred_key: str, run_name: str, dataset: str, source: str) -> dict[str, Any]:
    records = load_json_array(path)
    predict_key = pick_prediction_key(records, preferred_key)
    summary = score_records(records, predict_key)
    summary.update(
        {
            "source": source,
            "model": model,
            "run_name": run_name,
            "dataset": dataset,
            "prediction_key": predict_key,
            "results_file": str(path),
        }
    )
    return summary


def add_deltas(summary: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "valid_three_tag_rate",
        "response_1_accuracy",
        "response_2_accuracy",
        "response_3_accuracy",
        "all_three_accuracy",
        "third_response_neutral_rate",
    ]
    return {metric: round(float(summary[metric]) - float(reference[metric]), 6) for metric in metrics}


def write_tables(root: Path, rows: list[dict[str, Any]]) -> None:
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "source",
        "model",
        "run_name",
        "dataset",
        "total_examples",
        "valid_three_tag_rate",
        "response_1_accuracy",
        "response_2_accuracy",
        "response_3_accuracy",
        "all_three_accuracy",
        "delta_all_three_accuracy",
        "third_response_neutral_rate",
        "results_file",
    ]

    tsv_path = analysis_dir / "gemma4_vs_gemma2_dpo.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    md_path = analysis_dir / "gemma4_vs_gemma2_dpo.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| source | model | run | valid tags | r1 acc | r2 acc | r3 acc | all 3 acc | delta all 3 | neutral 3 |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                "| {source} | {model} | {run_name} | {valid_three_tag_rate:.3f} | "
                "{response_1_accuracy:.3f} | {response_2_accuracy:.3f} | "
                "{response_3_accuracy:.3f} | {all_three_accuracy:.3f} | "
                "{delta_all_three_accuracy:.3f} | {third_response_neutral_rate:.3f} |\n".format(
                    **row
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", choices=[SUPPORTED_RUN], default=SUPPORTED_RUN)
    parser.add_argument("--strict", action="store_true", help="Fail when any expected Gemma4 result file is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset or default_dataset_for_run(args.run_name)
    preferred_key = f"predict_{args.run_name}"

    reference_file = result_path(args.root, REFERENCE_MODEL, dataset, args.run_name)
    if not reference_file.exists():
        print(
            f"[ERROR] Missing generated Gemma2 reference: {reference_file}\n"
            "Run: scripts/gemma4.py reference --stage all",
            file=sys.stderr,
        )
        return 1

    reference = summarize(reference_file, REFERENCE_MODEL, preferred_key, args.run_name, dataset, "gemma2_reference")
    reference["delta_all_three_accuracy"] = 0.0
    rows = [reference]
    gemma4_rows: list[dict[str, Any]] = []

    for model in args.models:
        path = result_path(args.root, model, dataset, args.run_name)
        if not path.exists():
            message = f"[WARN] Missing analyzed Gemma4 results: {path}"
            if args.strict:
                print(message, file=sys.stderr)
                return 1
            print(message, file=sys.stderr)
            continue
        row = summarize(path, model, preferred_key, args.run_name, dataset, "gemma4")
        deltas = add_deltas(row, reference)
        row["deltas_vs_gemma2_reference"] = deltas
        row["delta_all_three_accuracy"] = deltas["all_three_accuracy"]
        rows.append(row)
        gemma4_rows.append(row)

    output = {
        "reference_model": "https://huggingface.co/mario-rc/gemma-2-9b-it-emotional-rlaif-dpo",
        "run_name": args.run_name,
        "dataset": dataset,
        "reference": reference,
        "gemma4": gemma4_rows,
    }
    write_json(args.root / "analysis" / "gemma4_vs_gemma2_dpo.json", output)
    write_tables(args.root, rows)
    print(f"Wrote DPO comparison to {args.root / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
