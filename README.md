# Gemma 4 Emotional RLAIF DPO

Standalone project for training and evaluating Gemma4 models with an emotional
SFT and DPO/RLAIF workflow.

Repository name:

```text
gemma-4-emotional-rlaif-dpo
```

The project is intentionally isolated from the rest of the RLAIF workspace. Its
virtual environment, Hugging Face cache, datasets, configs, logs, adapters,
predictions, and analysis results all live under this directory.

## Scope

Models:

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`

Training stages:

- SFT LoRA, 3 epochs.
- DPO LoRA, 3 epochs, initialized from the corresponding SFT adapter.
- Test-set prediction after SFT and after DPO.
- Metric analysis for SFT and DPO predictions.

## Data Source

Training and prediction datasets mirror the canonical phase2 SFT and phase3
RLAIF LlamaFactory datasets. The large JSON files are kept local and ignored by
Git; `datasets/dataset_info.json` is the trackable index.

```text
mario-rc/aif-emotional-generation
```

Use this check after syncing the large local JSON files:

```bash
scripts/gemma4.py data --verify-only
```

The expected local LlamaFactory dataset files are:

| Local file | Canonical source |
| --- | --- |
| `datasets/sft_demonstration_dataset.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_foundation.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_test.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_test_history.json` | phase2 SFT |
| `datasets/dpo_preference_dataset.json` | phase3 RLAIF |
| `datasets/ppo_unlabeled_prompts_dataset.json` | phase3 RLAIF |
| `datasets/ppo_unlabeled_prompts_dataset_test.json` | phase3 RLAIF |

## Layout

```text
gemma-4/
  README.md
  LlamaFactory/                 # cloned LlamaFactory checkout
  vgemma4/                      # project-local virtual environment
  configs/                      # SFT, DPO, and prediction YAML configs
  datasets/                     # local SFT, DPO, and test datasets
  saves/                        # adapters, predictions, analyzed per-model outputs
  logs/                         # training and prediction logs
  analysis/                     # summary and DPO comparison tables
  src/gemma4_project/           # dataset and analysis Python modules
  scripts/
    gemma4.py                   # main CLI entrypoint
    env_gemma4.sh               # environment helper
```

## Main CLI

Use the single CLI entrypoint:

```bash
scripts/gemma4.py --help
```

Check current artifacts:

```bash
scripts/gemma4.py status
scripts/gemma4.py status e2b
scripts/gemma4.py status e4b
```

When a training stage has checkpoints but no final adapter yet, `status` also
prints the newest `checkpoint-*` directory that can be used with `--resume`.

## Setup After Clone

Create or activate the project environment, then clone LlamaFactory into the
expected local path:

```bash
cd /autofs/thau00a/home/mrodriguez/data/rlaif/gemma-4
python3 -m venv vgemma4
source vgemma4/bin/activate
python -m pip install --upgrade pip setuptools wheel
git clone https://github.com/hiyouga/LlamaFactory.git LlamaFactory
```

Install LlamaFactory and project dependencies inside `vgemma4`:

```bash
python -m pip install -e ./LlamaFactory
```

If the node needs explicit CUDA 12.8 PyTorch wheels, install PyTorch first and
then install LlamaFactory:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./LlamaFactory
```

The local CLI expects the checkout at `LlamaFactory/` and writes all runtime
caches under `gemma-4/`.

Then download/prepare the datasets from Hugging Face:

```bash
scripts/gemma4.py data
```

Use `--force` only when you intentionally want to overwrite existing local
dataset files:

```bash
scripts/gemma4.py data --force
```

## Full Pipeline

Run the complete flow for one model:

```bash
scripts/gemma4.py pipeline e2b --strict
scripts/gemma4.py pipeline e4b --strict
```

Run both models:

```bash
scripts/gemma4.py pipeline all --strict
```

The full pipeline runs:

1. SFT training.
2. SFT test prediction.
3. DPO training from the SFT adapter.
4. DPO test prediction.
5. Analysis.

## Idempotent Execution

Training and prediction commands are safe to re-run. The CLI checks for the
expected final artifact before launching each expensive stage:

| Stage | Skip condition |
| --- | --- |
| Gemma4 SFT train | `saves/<model>/lora/sft_3ep/adapter_model.safetensors` exists |
| Gemma4 SFT predict | `saves/<model>/predict/sft_3ep/generated_predictions.jsonl` exists |
| Gemma4 DPO train | `saves/<model>/lora/dpo_3ep/adapter_model.safetensors` exists |
| Gemma4 DPO predict | `saves/<model>/predict/dpo_3ep/generated_predictions.jsonl` exists |

To deliberately rerun an existing stage, add `--force`:

```bash
scripts/gemma4.py train e2b --stage sft --force
scripts/gemma4.py predict e2b --run-name dpo_3ep --force
scripts/gemma4.py pipeline e2b --strict --force
```

## Resume From Checkpoint

If SFT or DPO stops before the final adapter is written, resume from the latest
`checkpoint-*` directory with `--resume`:

```bash
scripts/gemma4.py train e2b --stage dpo --resume
scripts/gemma4.py pipeline e2b --strict --resume
scripts/gemma4.py pipeline all --strict --resume
```

The CLI keeps the original YAML unchanged, writes a temporary resume config
under `tmp/resume_configs/`, sets `resume_from_checkpoint` to the newest
checkpoint, and disables `overwrite_output_dir` for that resumed run.

Use `--resume` after an interrupted SFT/DPO run. A plain rerun without `--resume`
uses the base YAML and may overwrite an unfinished output directory.

## Running By Stage

SFT:

```bash
scripts/gemma4.py train e2b --stage sft
scripts/gemma4.py predict e2b --run-name sft_3ep
```

DPO:

```bash
scripts/gemma4.py train e2b --stage dpo
scripts/gemma4.py predict e2b --run-name dpo_3ep
```

Analyze existing predictions:

```bash
scripts/gemma4.py analyze e2b --run-name all --strict
```

## Smoke Runs

Smoke runs are available for quick checks of SFT and DPO configs. They do not
run prediction or analysis:

```bash
scripts/gemma4.py train e2b --stage pipeline --smoke
```

## Outputs

Expected artifacts for each model:

```text
saves/<model>/lora/sft_3ep/adapter_model.safetensors
saves/<model>/predict/sft_3ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/demonstration_data_emotional_balanced_test_results.json

saves/<model>/lora/dpo_3ep/adapter_model.safetensors
saves/<model>/predict/dpo_3ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/ppo_unlabeled_prompts_dataset_test_results.json
```

Project-level summaries are written to:

```text
analysis/gemma4_summary.json
analysis/gemma4_summary.tsv
analysis/gemma4_summary.md
```

LlamaFactory stdout/stderr for train and prediction stages is mirrored to
`logs/`. The CLI writes the launched command at the top of each log file and
flushes new lines as they arrive, so long-running jobs can be monitored with:

```bash
tail -f logs/dpo_gemma-4-E2B-it_dpo_3ep.log
```

## Environment

The project-local environment is `vgemma4`.

By default, `scripts/gemma4.py` infers the project root from its own location.
Set `GEMMA4_ROOT` only if you intentionally want to override that path.

The CLI sets these paths inside `gemma-4` before launching LlamaFactory:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `XDG_CACHE_HOME`
- `PIP_CACHE_DIR`
- `TMPDIR`
- `PYTHONPATH`

Use project-local Hugging Face login if needed:

```bash
scripts/gemma4.py login
```

## NFS Stability

The configs use conservative worker settings:

```yaml
preprocessing_num_workers: 1
dataloader_num_workers: 0
```

This avoids common NFS cleanup errors such as `.nfs... Device or resource busy`
during dataset preprocessing or worker shutdown.

## GitHub Notes

This repository tracks code, configs, and documentation. Large generated
artifacts stay out of Git unless intentionally handled with Git LFS:

- `vgemma4/`
- `.cache/`
- `tmp/`
- model adapters in `saves/`
- prediction logs in `logs/`
- large local datasets in `datasets/`
- the third-party `LlamaFactory/` checkout

The `.gitignore` is configured for that policy while keeping
`datasets/dataset_info.json` trackable.

## GitHub Upload

Use the repository name:

```text
gemma-4-emotional-rlaif-dpo
```

Because this directory can live inside a larger RLAIF Git workspace, first check
which directory Git considers the repository root:

```bash
git rev-parse --show-toplevel
```

For an independent GitHub repository, the top level should be this `gemma-4`
directory. If Git reports a parent workspace instead, initialize a new repository
inside `gemma-4`:

```bash
git init -b main
git remote add origin git@github.com:<user>/gemma-4-emotional-rlaif-dpo.git
```

Stage only the reproducible project files:

```bash
git add README.md LICENSE .gitignore configs datasets/dataset_info.json scripts src
git status --short --ignored
git commit -m "Initial Gemma4 emotional RLAIF DPO project"
git push -u origin main
```

Do not add `vgemma4/`, `.cache/`, `tmp/`, `LlamaFactory/`, `saves/`, `logs/`,
`analysis/`, or the generated large dataset JSON files unless there is a
deliberate Git LFS plan for them.

## License

This project is released under the MIT License. See `LICENSE`.

The license applies to this project's code, configs, and documentation. Third
party projects, model weights, datasets, and Hugging Face artifacts remain under
their own licenses and terms.
