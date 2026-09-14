# TinyLibrary training and evaluation

These scripts train nanoVLM models on TinyLibrary under the curricula reported in the paper.

Use the Python 3.11 training/evaluation environment and exact nanoVLM and BabyLM revisions recorded in [`environment/README.md`](../environment/README.md). The corpus-construction environment is incompatible for these and should not be used for these scripts.

## Inputs

`prepare_manifest.py` expects twelve LLaVA-format JSON files named `{task}_{age_band}.json`, where `task` is one of `caption`, `vqa`, or `reasoning`, and `age_band` is one of `3_5`, `6_8`, `9_12`, or `12`. The corresponding page images must be available locally. Images are not part of this release.

Each generated manifest record contains:

```json
{
  "id": "caption_3_5_example",
  "image": "book_page.jpg",
  "texts": [{"user": "Describe the image.", "assistant": "..."}],
  "task": "caption",
  "age_band": "3_5",
  "n_words": 42,
  "fk_grade": 3.2
}
```

Teacher scoring adds `nll` and `n_tokens`. Alternative-teacher scoring can add fields such as `nll_qwen` and `nll_gemma`.

## Preparing and scoring the manifest

```bash
python training/prepare_manifest.py \
  --json_dir "$TINYLIBRARY_BASE/updated_jsons" \
  --image_root "$TINYLIBRARY_BASE/images" \
  --out "$TINYLIBRARY_BASE/manifest.jsonl"

python training/compute_perplexity.py \
  --manifest "$TINYLIBRARY_BASE/manifest.jsonl" \
  --out "$TINYLIBRARY_BASE/manifest_scored.jsonl"
```

The primary teacher is `HuggingFaceTB/SmolLM2-135M`. Pass `--scorer` to use a different causal language model.

## Curriculum conditions

| Flag | Values |
|---|---|
| `--order` | `developmental`, `antidevelopmental`, `readability`, `perplexity`, `random`, and the paper controls |
| `--schedule` | `staged`, `competence` |
| `--lm_init` | `scratch` (default), `pretrained` (ablation) |

For staged and random runs, one epoch is one complete corpus pass, so those conditions have matched sample and token exposure. Competence runs draw with replacement from an expanding prefix.

All runs use the deterministic 2% validation split implemented in `tinylibrary_dataset.py`. Reusing a seed across conditions supports paired comparisons with the random baseline.

## Exporting the dataset artifacts

`export_dataset_artifacts.py` writes difficulty scores, sample mappings, and 63 reconstructed pre-packing curriculum sequences. It needs NumPy, the original scored manifest, and the twelve final annotation files; it does not load models or images. The manifest's annotation text is checked against the JSONs, but is not duplicated in the outputs. Existing outputs are never overwritten: use a new or empty staging directory.

```bash
python training/export_dataset_artifacts.py \
  --manifest /path/to/manifest_scored_multi.jsonl \
  --annotations /path/to/TinyLibrary-dataset \
  --reference /path/to/TinyLibrary-dataset/export_metadata.json \
  --output /path/to/new-artifact-staging
```

The released reference contains hashes/fingerprints produced by the original experiment setup.

To use an exported sample order, follow the sample-mapping example in the dataset card. The original manifest is not required to look up or load the samples referenced by an exported order.

## Training locally

Check out the recorded nanoVLM revision and make its root importable:

```bash
export NANOVLM_DIR=/path/to/nanoVLM
cd "$NANOVLM_DIR"
PYTHONPATH=.:/path/to/TinyLibrary/training \
python /path/to/TinyLibrary/training/train_curriculum.py \
  --manifest /path/to/manifest_scored.jsonl \
  --image_root /path/to/images \
  --checkpoint_path /path/to/checkpoints \
  --order developmental \
  --schedule staged \
  --epochs 10 \
  --seed 0 \
  --no_log_wandb
```

W&B logging is optional. Configure it with `WANDB_PROJECT` and, if needed, `WANDB_ENTITY`; use `--no_log_wandb` to disable it.

## Slurm templates

The templates in `slurm/` contain no account, partition, module, or filesystem assumptions. Before submission, set:

```bash
export TINYLIBRARY_CODE_DIR=/absolute/path/to/TinyLibrary
export TINYLIBRARY_BASE=/path/to/experiment-workspace
export TINYLIBRARY_ACTIVATE=/path/to/venv/bin/activate  # optional
export NANOVLM_DIR=/path/to/nanoVLM                     # optional if under BASE
export BABYLM_EVAL_DIR=/path/to/babylm-eval             # optional if under BASE
```

`TINYLIBRARY_CODE_DIR` must be absolute because Slurm executes a transferred copy of the submitted batch script, not the copy inside the repository.

Supply cluster-specific resources through `sbatch`, for example:

```bash
sbatch --partition=gpu --time=12:00:00 \
  training/slurm/train_curriculum.slurm \
  --order developmental --schedule staged --epochs 10 --seed 0
```

## Main files

- `prepare_manifest.py` — LLaVA JSON files to the flat training manifest
- `compute_perplexity.py` — teacher-model NLL and token counts
- `merge_teacher_scores.py` — merge alternative-teacher scores
- `export_dataset_artifacts.py` — verified scores, sample mappings, and order-file exports
- `curriculum.py` — ordering and schedule construction
- `tinylibrary_dataset.py` — dataset adapters and deterministic split
- `train_curriculum.py` — nanoVLM training wrapper
- `export_decoder_hf.py` — decoder conversion for BabyLM evaluation
- `run_multimodal.py` — nanoVLM wrapper for VQA and Winoground evaluation
- `collect_*.py` — result collation and trajectory analysis

The result collectors keep epoch budget, LM initialization, and modality as separate conditions. If more than one rerun supplies the same task and seed, they stop rather than choosing one silently. Use `--epochs`, `--lm_init`, `--modality`, or `--run_name_regex` to select the intended experiment.

For example, to collate the primary multimodal-model grid only:

```bash
python training/collect_zeroshot.py \
  --results_dir "$BABYLM_EVAL_DIR/strict/results" \
  --epochs 10 --lm_init scratch --modality mm
```