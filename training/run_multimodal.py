"""Multimodal zero-shot eval (VQA / Winoground) for nanoVLM checkpoints, using
the *official* BabyLM-2026 sentence-zero-shot completion-ranking scorer.

nanoVLM is not an HF `AutoModelForCausalLM`, and the official pipeline
(`evaluation_pipeline.sentence_zero_shot.run`) hard-codes AutoModel/AutoProcessor.
So we reuse the official data loader, phrase-mask logic, metric and report format
verbatim, and swap in exactly two things:

  * NanoVLMProcessor  -- builds `input_ids` = [image-placeholder string] + [text],
                         plus `pixel_values` = image tiles, mirroring nanoVLM's own
                         train-time image handling (data/datasets.py + processors.py)
                         so the number of `<|image|>` placeholders always matches the
                         number of projected image embeddings.
  * NanoVLMScorer     -- forward(input_ids, attention_mask, pixel_values) -> (logits,)
                         does the image-token substitution nanoVLM does natively and
                         applies the LM head (nanoVLM.forward skips the head when
                         targets is None), returning full-length B x T x V logits.

The official scorer slices the last len(text) positions and ranks completions by
log-likelihood -- unchanged. Run from inside `babylm-eval/strict`.

Caveat recorded for the paper: candidates are scored as raw completions (the
official protocol), not wrapped in the SmolLM2 chat template the model trained
with. This matches how every other BabyLM multimodal submission is scored.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

import torch

# --- make nanoVLM importable (models.*, data.*) ---
NANOVLM_DIR = os.environ.get("NANOVLM_DIR")
if not NANOVLM_DIR:
    raise RuntimeError("Set NANOVLM_DIR to the root of a nanoVLM checkout.")
if NANOVLM_DIR not in sys.path:
    sys.path.insert(0, NANOVLM_DIR)

from models.vision_language_model import VisionLanguageModel  # noqa: E402
from data.processors import get_image_processor, get_image_string  # noqa: E402

# --- make the babylm-eval package importable (we run with cwd = babylm-eval/strict,
#     but running a script by path puts the script's dir on sys.path, not cwd) ---
sys.path.insert(0, os.getcwd())

# --- official pipeline pieces (we reuse these unchanged) ---
from evaluation_pipeline.sentence_zero_shot.dataset import (  # noqa: E402
    CompletionRankingDataset,
    get_collate_fn,
)
from evaluation_pipeline.sentence_zero_shot.compute_results import compute_results  # noqa: E402
from evaluation_pipeline.sentence_zero_shot import run as official_run  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NanoVLMProcessor:
    """Minimal stand-in for an HF processor, exposing the exact call surface the
    official CompletionRankingDataset uses: __call__(text, images, return_offsets_mapping)
    returning input_ids / attention_mask / offset_mapping / pixel_values, plus a
    `.tokenizer` attribute."""

    def __init__(self, tokenizer, image_processor, mp_image_token_length):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

    def _process_images(self, images):
        """Mirror data/datasets.py::BaseDataset._process_images."""
        from PIL import Image

        processed_images, splitted_image_counts = [], []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL image, got {type(image)}")
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image, split_count = self.image_processor(image)
            if not hasattr(self.tokenizer, "global_image_token") and \
                    split_count[0] * split_count[1] == len(processed_image) - 1:
                processed_image = processed_image[1:]
            processed_images.append(processed_image)
            splitted_image_counts.append(split_count)
        return processed_images, splitted_image_counts

    def __call__(self, text=None, images=None, return_offsets_mapping=False, **kw):
        out = {}
        pixel_values = None
        image_prefix = ""

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            processed_images, split_counts = self._process_images(images)
            # Build the exact placeholder string nanoVLM prepends at train time.
            image_prefix = get_image_string(self.tokenizer, split_counts, self.mp_image_token_length)
            # Flatten tiles across all images -> (n_tiles, 3, H, W)
            tiles = [t for per_image in processed_images for t in per_image]
            pixel_values = torch.stack(tiles, dim=0)

        # Tokenize the image-placeholder prefix (no offsets needed) and the text
        # (with offsets, so the scorer can locate the completion char span).
        if image_prefix:
            prefix_ids = self.tokenizer(image_prefix, add_special_tokens=False)["input_ids"]
        else:
            prefix_ids = []

        text_enc = self.tokenizer(text, add_special_tokens=False,
                                  return_offsets_mapping=return_offsets_mapping)
        text_ids = text_enc["input_ids"]
        text_attn = text_enc["attention_mask"]

        out["input_ids"] = prefix_ids + text_ids
        out["attention_mask"] = [1] * len(prefix_ids) + text_attn
        if return_offsets_mapping:
            text_offsets = text_enc["offset_mapping"]
            out["offset_mapping"] = [(0, 0)] * len(prefix_ids) + list(text_offsets)
        if pixel_values is not None:
            out["pixel_values"] = pixel_values
        return out


class NanoVLMScorer(torch.nn.Module):
    """Wraps a loaded VisionLanguageModel to expose the (input_ids, attention_mask,
    pixel_values) -> (logits,) contract the official causal scorer expects."""

    def __init__(self, vlm: VisionLanguageModel):
        super().__init__()
        self.vlm = vlm
        self._cache_key = None      # id(pixel_values) of the last image
        self._cache_embd = None     # its projected embedding

    def _image_embd(self, pixel_values, device):
        # The official scorer loops candidates of one item, re-passing the SAME
        # pixel_values object each time -> memoize by identity to run the vision
        # encoder once per image instead of once per candidate (~8x for VQA).
        key = id(pixel_values)
        if key != self._cache_key:
            embd = self.vlm.MP(self.vlm.vision_encoder(pixel_values.to(device)))
            self._cache_key, self._cache_embd = key, embd
        return self._cache_embd

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, pixel_values=None):
        vlm = self.vlm
        token_embd = vlm.decoder.token_embedding(input_ids)
        if pixel_values is not None:
            image_embd = self._image_embd(pixel_values, input_ids.device)
            token_embd = vlm._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
        hidden, _ = vlm.decoder(token_embd, attention_mask=attention_mask)
        logits = vlm.decoder.head(hidden)  # nanoVLM.forward only applies head when targets given
        return (logits,)


def _read_vqa_local(args):
    """Build the VQA eval set from vqa.jsonl + local COCO val2014 images (by
    image_id), bypassing the HuggingFaceM4/VQAv2 script (which pulls ~40 GB of
    train+val+test COCO archives from flaky external hosts). Mirrors decode_vqa."""
    import glob
    from PIL import Image

    coco = pathlib.Path(args.local_coco_dir)
    data = []
    for fn in args.data_path.iterdir():
        if fn.suffix != ".jsonl":
            continue
        for line in fn.open():
            r = json.loads(line)
            pair = {
                "sentences": [" ".join([r["question"], r["target_ans"]])]
                             + [" ".join([r["question"], a]) for a in r["distractors"]],
                "prefixes": [r["question"]] * (len(r["distractors"]) + 1),
                "completions": [" " + r["target_ans"]] + [" " + a for a in r["distractors"]],
                "label": 0,
                "UID": "vqa",
            }
            img_path = coco / f"COCO_val2014_{int(r['image_id']):012d}.jpg"
            pair["image"] = Image.open(img_path).convert("RGB")
            data.append(pair)
    return data


class NanoVLMCompletionRankingDataset(CompletionRankingDataset):
    """CompletionRankingDataset that uses NanoVLMProcessor instead of AutoProcessor."""

    def __init__(self, args, processor):
        # deliberately skip the parent __init__ (which builds an AutoProcessor)
        self.backend = args.backend
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.image_token = None  # no image_template: images ride in via pixel_values
        if args.task == "vqa" and getattr(args, "local_coco_dir", None):
            self.data = _read_vqa_local(args)
        else:
            from evaluation_pipeline.sentence_zero_shot.read_files import read_files
            self.data = read_files(args)


def build_dataloader(args, processor):
    dataset = NanoVLMCompletionRankingDataset(args, processor)
    pad_idx = dataset.tokenizer.pad_token_id
    collate_fn = get_collate_fn(args, pad_idx)
    return DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)


def main():
    args = official_run._parse_arguments()
    # Optional: load VQA images from a local COCO val2014 dir (by image_id) instead
    # of the HuggingFaceM4/VQAv2 script. Keep --images_path set (non-None) so the
    # scorer still feeds pixel_values; the local reader ignores its value.
    args.local_coco_dir = os.environ.get("LOCAL_COCO_DIR")
    assert args.backend == "causal", "nanoVLM eval is causal-only"
    if args.images_path is not None:
        assert args.batch_size == 1, "Multimodal only works in batch size 1!"

    dataset_stem = args.data_path.stem
    # Key results by the RUN name, not the checkpoint step dir: model_path is
    # "<run>/step_N", whose .stem is "step_N" (collides across conditions and is
    # unmatchable by the collator). Use the parent run-dir name in that case.
    _mp = pathlib.Path(args.model_path_or_name)
    args.model_name = _mp.parent.name if re.match(r"step_\d+$", _mp.name) else _mp.name
    revision_name = args.revision_name if args.revision_name is not None else "main"
    args.output_path = args.output_dir / args.model_name / revision_name / "zero_shot" / args.backend / args.task / dataset_stem
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Load nanoVLM + build our processor from its own config/tokenizer.
    vlm = VisionLanguageModel.from_pretrained(args.model_path_or_name)
    vlm.to(DEVICE).eval()
    # Match training exactly: get_image_processor(max_img_size, vit_img_size, resize_to_max_side_len)
    image_processor = get_image_processor(
        vlm.cfg.max_img_size, vlm.cfg.vit_img_size, vlm.cfg.resize_to_max_side_len
    )
    processor = NanoVLMProcessor(vlm.tokenizer, image_processor, vlm.cfg.mp_image_token_length)
    model = NanoVLMScorer(vlm)

    dataloader = build_dataloader(args, processor)
    temperatures = official_run.get_temperatures(args)
    results, predictions = compute_results(args, model, dataloader, temperatures)

    accuracies, average_accuracies = official_run.process_results(args, results)
    best_acc, best_temp = -1, -1
    for temperature, acc in average_accuracies.items():
        print(f"{temperature}\t{acc:.2f}")
        if acc > best_acc:
            best_acc, best_temp = acc, temperature
    print()

    official_run.create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task)
    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        official_run.create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task, file=f)

    if args.save_predictions:
        official_run.save_predictions(args, predictions, best_temp)


if __name__ == "__main__":
    main()
