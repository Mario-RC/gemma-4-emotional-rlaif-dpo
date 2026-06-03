#!/usr/bin/env python3
"""Analyze Gemma4 SFT and DPO predictions on their matching test sets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("GEMMA4_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_MODELS = ("gemma-4-E2B-it", "gemma-4-E4B-it")
REFERENCE_MODEL = "gemma-2-9b-it-emotional-rlaif-dpo"
RUN_CONFIGS = {
    "sft_3ep": {
        "dataset": "sft_demonstration_dataset_test",
        "result_file": "demonstration_data_emotional_balanced_test_results.json",
    },
    "dpo_3ep": {
        "dataset": "ppo_unlabeled_prompts_dataset_test",
        "result_file": "ppo_unlabeled_prompts_dataset_test_results.json",
    },
}
DEFAULT_RUNS = tuple(RUN_CONFIGS)
DEFAULT_DATASET = RUN_CONFIGS["dpo_3ep"]["dataset"]
DEFAULT_RUN = "all"
EMOTION_RE = re.compile(r"\(([A-Za-z_ -]+)\)")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_emotions(text: str) -> list[str]:
    return [f"({match.group(1).strip().upper()})" for match in EMOTION_RE.finditer(text or "")]


def default_dataset_for_run(run_name: str) -> str:
    return RUN_CONFIGS.get(run_name, {}).get("dataset", DEFAULT_DATASET)


def result_filename(dataset: str, run_name: str) -> str:
    return RUN_CONFIGS.get(run_name, {}).get("result_file", f"{dataset}_results.json")


def prediction_keys(run_name: str) -> tuple[str, list[str]]:
    primary = f"predict_{run_name}"
    if run_name == "sft_3ep":
        return primary, ["predict_sft"]
    if run_name.endswith("ep"):
        return primary, [f"predict_{run_name[:-2]}"]
    return primary, []


def safe_rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def score_records(records: list[dict[str, Any]], predict_key: str) -> dict[str, Any]:
    total = len(records)
    missing = 0
    valid_three_tags = 0
    correct_by_position = [0, 0, 0]
    target_available = [0, 0, 0]
    all_three_correct = 0
    third_neutral = 0

    for record in records:
        target_tags = extract_emotions(str(record.get("target", "")))[:3]
        predicted = str(record.get(predict_key, ""))
        predicted_tags = extract_emotions(predicted)[:3]

        if not predicted:
            missing += 1
        if len(predicted_tags) >= 3:
            valid_three_tags += 1
        if len(predicted_tags) >= 3 and predicted_tags[2] == "(NEUTRAL)":
            third_neutral += 1

        row_all_correct = len(target_tags) >= 3 and len(predicted_tags) >= 3
        for idx in range(3):
            if len(target_tags) > idx:
                target_available[idx] += 1
            if len(target_tags) > idx and len(predicted_tags) > idx and target_tags[idx] == predicted_tags[idx]:
                correct_by_position[idx] += 1
            else:
                row_all_correct = False
        if row_all_correct:
            all_three_correct += 1

    return {
        "total_examples": total,
        "missing_predictions": missing,
        "valid_three_tag_predictions": valid_three_tags,
        "valid_three_tag_rate": safe_rate(valid_three_tags, total),
        "response_1_accuracy": safe_rate(correct_by_position[0], target_available[0]),
        "response_2_accuracy": safe_rate(correct_by_position[1], target_available[1]),
        "response_3_accuracy": safe_rate(correct_by_position[2], target_available[2]),
        "all_three_accuracy": safe_rate(all_three_correct, total),
        "third_response_neutral_rate": safe_rate(third_neutral, total),
        "correct_by_position": {
            "response_1": correct_by_position[0],
            "response_2": correct_by_position[1],
            "response_3": correct_by_position[2],
            "all_three": all_three_correct,
        },
    }


def merge_dataset_and_predictions(
    dataset: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    model: str,
    run_name: str,
) -> tuple[list[dict[str, Any]], str]:
    predict_key, alias_keys = prediction_keys(run_name)
    merged: list[dict[str, Any]] = []

    for idx, source_row in enumerate(dataset):
        pred_row = predictions[idx] if idx < len(predictions) else {}
        predicted_text = str(pred_row.get("predict", ""))
        record = {
            "model": model,
            "did": source_row.get("dialogue_id", source_row.get("did", str(idx))),
            "set": source_row.get("set"),
            "instruction": source_row.get("system", ""),
            "history": source_row.get("history", []),
            "prompt": source_row.get("instruction", ""),
            "target": source_row.get("output", pred_row.get("label", "")),
            predict_key: predicted_text,
        }
        for alias_key in alias_keys:
            record[alias_key] = predicted_text
        merged.append(record)

    return merged, predict_key


def write_summary_tables(root: Path, summaries: list[dict[str, Any]]) -> None:
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = analysis_dir / "gemma4_summary.tsv"
    columns = [
        "model",
        "run_name",
        "dataset",
        "total_examples",
        "missing_predictions",
        "valid_three_tag_rate",
        "response_1_accuracy",
        "response_2_accuracy",
        "response_3_accuracy",
        "all_three_accuracy",
        "third_response_neutral_rate",
        "results_file",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in summaries:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    md_path = analysis_dir / "gemma4_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| model | run | valid tags | r1 acc | r2 acc | r3 acc | all 3 acc | neutral 3 |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            f.write(
                "| {model} | {run_name} | {valid_three_tag_rate:.3f} | "
                "{response_1_accuracy:.3f} | {response_2_accuracy:.3f} | "
                "{response_3_accuracy:.3f} | {all_three_accuracy:.3f} | "
                "{third_response_neutral_rate:.3f} |\n".format(**row)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", choices=[*DEFAULT_RUNS, "all"], default=DEFAULT_RUN)
    parser.add_argument("--strict", action="store_true", help="Fail when any expected prediction file is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    run_names = list(DEFAULT_RUNS) if args.run_name == "all" else [args.run_name]

    for run_name in run_names:
        dataset_name = args.dataset or default_dataset_for_run(run_name)
        dataset_path = args.root / "datasets" / f"{dataset_name}.json"
        dataset = load_json_array(dataset_path)

        for model in args.models:
            prediction_path = args.root / "saves" / model / "predict" / run_name / "generated_predictions.jsonl"
            if not prediction_path.exists():
                message = f"[WARN] Missing prediction file: {prediction_path}"
                if args.strict:
                    print(message, file=sys.stderr)
                    return 1
                print(message, file=sys.stderr)
                continue

            predictions = load_jsonl(prediction_path)
            if len(predictions) != len(dataset):
                message = (
                    f"[WARN] Prediction row count mismatch for {model} {run_name}: "
                    f"{len(predictions)} predictions vs {len(dataset)} dataset rows"
                )
                if args.strict:
                    print(message, file=sys.stderr)
                    return 1
                print(message, file=sys.stderr)

            merged, predict_key = merge_dataset_and_predictions(dataset, predictions, model, run_name)
            result_path = (
                args.root
                / "saves"
                / model
                / "emotional_balanced"
                / result_filename(dataset_name, run_name)
            )
            write_json(result_path, merged)

            summary = score_records(merged, predict_key)
            summary.update(
                {
                    "model": model,
                    "run_name": run_name,
                    "dataset": dataset_name,
                    "prediction_file": str(prediction_path),
                    "results_file": str(result_path),
                    "prediction_rows": len(predictions),
                }
            )
            summaries.append(summary)

    write_json(args.root / "analysis" / "gemma4_summary.json", summaries)
    write_summary_tables(args.root, summaries)
    print(f"Wrote {len(summaries)} Gemma4 summary rows to {args.root / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
