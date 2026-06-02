#!/usr/bin/env python3
"""Single command-line entrypoint for the Gemma4 training project."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEMMA4_ROOT", DEFAULT_ROOT)).resolve()
MODELS = {
    "e2b": "gemma-4-E2B-it",
    "e4b": "gemma-4-E4B-it",
}
REFERENCE_MODEL = "gemma-2-9b-it-emotional-rlaif-dpo"
FULL_STAGES = ("sft", "predict_sft", "dpo", "predict_dpo")
SMOKE_STAGES = ("sft", "dpo")
PREDICT_RUNS = {
    "sft_3ep": "predict_sft",
    "dpo_3ep": "predict_dpo",
}


def project_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = root / "vgemma4" / "bin"
    cache_root = root / ".cache"
    hf_home = cache_root / "huggingface"

    env["GEMMA4_ROOT"] = str(root)
    env["VIRTUAL_ENV"] = str(root / "vgemma4")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hf_home / "hub")
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["TMPDIR"] = str(root / "tmp")
    env["PYTHONPATH"] = f"{root / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for directory in (
        hf_home,
        hf_home / "hub",
        hf_home / "datasets",
        cache_root / "pip",
        root / "tmp",
        root / "logs",
        root / "analysis",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return env


def selected_models(selector: str) -> list[str]:
    if selector == "all":
        return list(MODELS.values())
    return [MODELS[selector]]


def model_key(model_name: str) -> str:
    for key, value in MODELS.items():
        if value == model_name:
            return key
    raise KeyError(model_name)


def config_for(model_name: str, stage: str, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    if stage == "predict_sft":
        if smoke:
            raise ValueError("No smoke prediction config exists; smoke runs SFT and DPO only.")
        return ROOT / "configs" / f"predict_{model_name}_sft_3ep.yaml"
    if stage == "predict_dpo":
        if smoke:
            raise ValueError("No smoke prediction config exists; smoke runs SFT and DPO only.")
        return ROOT / "configs" / f"predict_{model_name}_dpo_3ep.yaml"
    return ROOT / "configs" / f"{stage}_{model_name}{suffix}.yaml"


def log_for(model_name: str, stage: str, smoke: bool) -> Path:
    prefix = "smoke_" if smoke else ""
    if stage in {"dpo", "predict_dpo"}:
        run_name = "dpo_3ep"
    else:
        run_name = "sft_3ep"
    return ROOT / "logs" / f"{prefix}{stage}_{model_name}_{run_name}.log"


def artifact_for(model_name: str, stage: str, smoke: bool) -> Path:
    if smoke and stage == "sft":
        return ROOT / "saves" / model_name / "lora" / "sft_smoke" / "adapter_model.safetensors"
    if smoke and stage == "dpo":
        return ROOT / "saves" / model_name / "lora" / "smoke" / "adapter_model.safetensors"
    if stage == "sft":
        return ROOT / "saves" / model_name / "lora" / "sft_3ep" / "adapter_model.safetensors"
    if stage == "dpo":
        return ROOT / "saves" / model_name / "lora" / "dpo_3ep" / "adapter_model.safetensors"
    if stage == "predict_sft":
        return ROOT / "saves" / model_name / "predict" / "sft_3ep" / "generated_predictions.jsonl"
    if stage == "predict_dpo":
        return ROOT / "saves" / model_name / "predict" / "dpo_3ep" / "generated_predictions.jsonl"
    raise ValueError(f"Unsupported stage: {stage}")


def reference_artifact(stage: str) -> Path:
    if stage == "predict":
        return ROOT / "saves" / REFERENCE_MODEL / "predict" / "dpo_3ep" / "generated_predictions.jsonl"
    if stage == "analyze":
        return (
            ROOT
            / "saves"
            / REFERENCE_MODEL
            / "emotional_balanced"
            / "ppo_unlabeled_prompts_dataset_test_results.json"
        )
    raise ValueError(f"Unsupported reference stage: {stage}")


def skip_existing(label: str, artifact: Path, force: bool) -> bool:
    if artifact.exists() and not force:
        print(f"[SKIP] {label}: {artifact} already exists. Add --force to rerun.")
        return True
    return False


def stream_command(command: list[str], env: dict[str, str], cwd: Path, log_path: Path | None = None) -> int:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
    else:
        log_file = None

    print(f"$ {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_file:
                log_file.write(line)
        return process.wait()
    finally:
        if log_file:
            log_file.close()


def run_llamafactory(model_name: str, stage: str, smoke: bool, force: bool = False) -> int:
    artifact = artifact_for(model_name, stage, smoke)
    if skip_existing(f"{model_name} {stage}", artifact, force):
        return 0
    env = project_env(ROOT)
    config_path = config_for(model_name, stage, smoke)
    log_path = log_for(model_name, stage, smoke)
    command = [str(ROOT / "vgemma4" / "bin" / "llamafactory-cli"), "train", str(config_path)]
    return stream_command(command, env=env, cwd=ROOT / "LlamaFactory", log_path=log_path)


def run_reference_predict(force: bool = False) -> int:
    artifact = reference_artifact("predict")
    if skip_existing(f"{REFERENCE_MODEL} predict_dpo", artifact, force):
        return 0
    env = project_env(ROOT)
    config_path = ROOT / "configs" / "predict_reference_gemma2_dpo_3ep.yaml"
    log_path = ROOT / "logs" / "predict_reference_gemma2_dpo_3ep.log"
    command = [str(ROOT / "vgemma4" / "bin" / "llamafactory-cli"), "train", str(config_path)]
    return stream_command(command, env=env, cwd=ROOT / "LlamaFactory", log_path=log_path)


def run_module(module: str, args: list[str]) -> int:
    env = project_env(ROOT)
    command = [str(ROOT / "vgemma4" / "bin" / "python"), "-m", module, "--root", str(ROOT), *args]
    return stream_command(command, env=env, cwd=ROOT)


def command_train(args: argparse.Namespace) -> int:
    stages = FULL_STAGES if args.stage == "pipeline" else (args.stage,)
    if args.smoke:
        stages = SMOKE_STAGES if args.stage == "pipeline" else (args.stage,)

    for model_name in selected_models(args.model):
        for stage in stages:
            code = run_llamafactory(model_name, stage, args.smoke, force=args.force)
            if code != 0:
                return code
    return 0


def command_predict(args: argparse.Namespace) -> int:
    run_names = list(PREDICT_RUNS) if args.run_name == "all" else [args.run_name]
    for model_name in selected_models(args.model):
        for run_name in run_names:
            code = run_llamafactory(model_name, PREDICT_RUNS[run_name], smoke=False, force=args.force)
            if code != 0:
                return code
    return 0


def analysis_args(args: argparse.Namespace) -> list[str]:
    models = selected_models(args.model)
    module_args = ["--models", *models]
    if args.strict:
        module_args.append("--strict")
    if args.dataset:
        module_args.extend(["--dataset", args.dataset])
    if args.run_name:
        module_args.extend(["--run-name", args.run_name])
    return module_args


def command_analyze(args: argparse.Namespace) -> int:
    code = run_module("gemma4_project.analysis", analysis_args(args))
    if code != 0 or args.no_compare:
        return code
    if args.run_name == "sft_3ep":
        print("[INFO] Skipping Gemma2 comparison for sft_3ep; only the DPO Gemma2 reference is published.")
        return 0
    compare_args = argparse.Namespace(
        model=args.model,
        strict=args.strict,
        dataset=args.dataset,
        run_name="dpo_3ep",
    )
    return run_module("gemma4_project.compare", analysis_args(compare_args))


def command_compare(args: argparse.Namespace) -> int:
    return run_module("gemma4_project.compare", analysis_args(args))


def command_reference(args: argparse.Namespace) -> int:
    if args.stage in {"predict", "all"}:
        code = run_reference_predict(force=args.force)
        if code != 0:
            return code
    if args.stage in {"analyze", "all"}:
        artifact = reference_artifact("analyze")
        if skip_existing(f"{REFERENCE_MODEL} analyze_dpo", artifact, args.force):
            return 0
        module_args = ["--models", REFERENCE_MODEL, "--run-name", "dpo_3ep"]
        if args.strict:
            module_args.append("--strict")
        return run_module("gemma4_project.analysis", module_args)
    return 0


def command_pipeline(args: argparse.Namespace) -> int:
    train_args = argparse.Namespace(model=args.model, stage="pipeline", smoke=args.smoke, force=args.force)
    code = command_train(train_args)
    if code != 0 or args.smoke or args.skip_analysis:
        return code
    analyze_args = argparse.Namespace(
        model=args.model,
        strict=args.strict,
        dataset=args.dataset,
        run_name=args.run_name,
        no_compare=False,
    )
    return command_analyze(analyze_args)


def command_data(args: argparse.Namespace) -> int:
    module_args = ["--repo-id", args.repo_id]
    if args.revision:
        module_args.extend(["--revision", args.revision])
    if args.force:
        module_args.append("--force")
    if args.verify_only:
        module_args.append("--verify-only")
    return run_module("gemma4_project.data", module_args)


def command_login(args: argparse.Namespace) -> int:
    env = project_env(ROOT)
    command = [str(ROOT / "vgemma4" / "bin" / "hf"), "auth", "login"]
    return stream_command(command, env=env, cwd=ROOT)


def command_status(args: argparse.Namespace) -> int:
    if args.model == "reference":
        paths = {
            "dpo_predictions": ROOT / "saves" / REFERENCE_MODEL / "predict" / "dpo_3ep" / "generated_predictions.jsonl",
            "dpo_analysis": ROOT
            / "saves"
            / REFERENCE_MODEL
            / "emotional_balanced"
            / "ppo_unlabeled_prompts_dataset_test_results.json",
        }
        print(REFERENCE_MODEL)
        for label, path in paths.items():
            marker = "OK" if path.exists() else "--"
            print(f"  {marker} {label}: {path}")
        return 0

    for model_name in selected_models(args.model):
        paths = {
            "sft_adapter": ROOT / "saves" / model_name / "lora" / "sft_3ep" / "adapter_model.safetensors",
            "sft_predictions": ROOT / "saves" / model_name / "predict" / "sft_3ep" / "generated_predictions.jsonl",
            "sft_analysis": ROOT
            / "saves"
            / model_name
            / "emotional_balanced"
            / "demonstration_data_emotional_balanced_test_results.json",
            "dpo_adapter": ROOT / "saves" / model_name / "lora" / "dpo_3ep" / "adapter_model.safetensors",
            "dpo_predictions": ROOT / "saves" / model_name / "predict" / "dpo_3ep" / "generated_predictions.jsonl",
            "dpo_analysis": ROOT
            / "saves"
            / model_name
            / "emotional_balanced"
            / "ppo_unlabeled_prompts_dataset_test_results.json",
        }
        print(model_name)
        for label, path in paths.items():
            marker = "OK" if path.exists() else "--"
            print(f"  {marker} {label}: {path}")
    return 0


def add_common_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    parser.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run one stage or SFT+SFT-predict+DPO+DPO-predict.")
    train.add_argument("model", choices=[*MODELS, "all"])
    train.add_argument("--stage", choices=[*FULL_STAGES, "pipeline"], default="pipeline")
    train.add_argument("--smoke", action="store_true", help="Use smoke configs; pipeline runs SFT and DPO only.")
    train.add_argument("--force", action="store_true", help="Rerun stages even when their expected artifacts exist.")
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser("predict", help="Run SFT and/or DPO prediction on the matching test set.")
    predict.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    predict.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    predict.add_argument("--force", action="store_true", help="Regenerate predictions even when they already exist.")
    predict.set_defaults(func=command_predict)

    analyze = subparsers.add_parser("analyze", help="Analyze Gemma4 predictions and compare to Gemma2.")
    add_common_analysis_args(analyze)
    analyze.add_argument("--no-compare", action="store_true")
    analyze.set_defaults(func=command_analyze)

    compare = subparsers.add_parser("compare", help="Compare analyzed Gemma4 DPO results with the generated Gemma2 DPO reference.")
    compare.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    compare.add_argument("--dataset", default=None)
    compare.add_argument("--run-name", choices=["dpo_3ep"], default="dpo_3ep")
    compare.add_argument("--strict", action="store_true")
    compare.set_defaults(func=command_compare)

    pipeline = subparsers.add_parser("pipeline", help="Run full train pipeline and then analysis.")
    pipeline.add_argument("model", choices=[*MODELS, "all"])
    pipeline.add_argument("--smoke", action="store_true", help="Run smoke SFT+DPO only; analysis is skipped.")
    pipeline.add_argument("--skip-analysis", action="store_true")
    pipeline.add_argument("--dataset", default=None)
    pipeline.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    pipeline.add_argument("--strict", action="store_true")
    pipeline.add_argument("--force", action="store_true", help="Rerun all pipeline stages even when artifacts exist.")
    pipeline.set_defaults(func=command_pipeline)

    data = subparsers.add_parser("data", help="Download and prepare datasets from Hugging Face.")
    data.add_argument("--repo-id", default="mario-rc/aif-emotional-generation")
    data.add_argument("--revision", default=None)
    data.add_argument("--force", action="store_true", help="Overwrite existing dataset JSON files.")
    data.add_argument("--verify-only", action="store_true", help="Only verify local dataset files.")
    data.set_defaults(func=command_data)

    reference = subparsers.add_parser("reference", help="Generate the Gemma2 DPO reference from Hugging Face.")
    reference.add_argument("--stage", choices=["predict", "analyze", "all"], default="all")
    reference.add_argument("--strict", action="store_true")
    reference.add_argument("--force", action="store_true", help="Regenerate reference prediction/analysis outputs.")
    reference.set_defaults(func=command_reference)

    status = subparsers.add_parser("status", help="Show expected artifacts for each model.")
    status.add_argument("model", choices=[*MODELS, "all", "reference"], nargs="?", default="all")
    status.set_defaults(func=command_status)

    login = subparsers.add_parser("login", help="Log in to Hugging Face using the project-local HF cache.")
    login.set_defaults(func=command_login)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
