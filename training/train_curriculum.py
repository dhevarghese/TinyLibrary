"""Curriculum training of nanoVLM on TinyLibrary.

Thin wrapper around nanoVLM's train.py: swaps in the TinyLibrary dataset with a
precomputed curriculum order, and (by default) trains the language model from
scratch while keeping the pretrained SigLIP vision encoder.

Must run with nanoVLM repo root on PYTHONPATH, e.g.:
    cd nanoVLM && PYTHONPATH=. python ../training/train_curriculum.py \
        --manifest ../manifest_scored.jsonl --image_root ../images \
        --order developmental --schedule staged --epochs 5 --seed 0
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import train as nanovlm_train  # nanoVLM/train.py
import models.config as config
from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from models.vision_transformer import ViT

from curriculum import build_order, summarize_order
from tinylibrary_dataset import OrderedView, TinyLibraryRaw, TinyLibraryTextRaw, load_manifest, split_train_val

ARGS = None


def estimate_max_steps(records, order_seq, train_cfg, vlm_cfg):
    """Estimate the packed-sequence steps needed to consume ``order_seq``.

    Slightly undershoots (x0.97) so no condition wraps around and re-enters
    the curriculum from the start. Staged and random runs have matched
    exposure; competence sampling can produce a different realized token mass.
    """
    def est_len(rec):
        n_tok = rec.get("n_tokens") or int(rec["n_words"] * 1.35)
        img_tok = 0 if ARGS.text_only else vlm_cfg.mp_image_token_length
        return n_tok + img_tok + 20  # (image tokens if multimodal) + chat template overhead

    total_tokens = sum(est_len(records[i]) for i in order_seq)
    tokens_per_step = (vlm_cfg.lm_max_length * train_cfg.batch_size
                       * train_cfg.gradient_accumulation_steps * nanovlm_train.get_world_size())
    return max(int(0.97 * total_tokens / tokens_per_step), 1)


def get_dataloaders(train_cfg, vlm_cfg):
    from torch.utils.data import DataLoader

    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    records = load_manifest(ARGS.manifest)
    train_recs, val_recs = split_train_val(records, ARGS.val_fraction)  # fixed seed: same split for every condition
    order_seq = build_order(train_recs, ARGS.order, ARGS.schedule, ARGS.epochs, ARGS.seed,
                            competence_c0=ARGS.competence_c0)

    if nanovlm_train.is_master():
        print(f"train {len(train_recs)} / val {len(val_recs)} samples; "
              f"order '{ARGS.order}/{ARGS.schedule}' x{ARGS.epochs} epochs = {len(order_seq)} draws")
        print(f"mean age-band index per decile of run: {summarize_order(train_recs, order_seq)}")

    # Estimate the step limit from the *global* order, before DDP sharding.
    train_cfg.max_training_steps = ARGS.max_steps or estimate_max_steps(train_recs, order_seq, train_cfg, vlm_cfg)
    if nanovlm_train.is_master():
        print(f"max_training_steps = {train_cfg.max_training_steps}")

    if nanovlm_train.is_dist():
        order_seq = order_seq[nanovlm_train.get_rank()::nanovlm_train.get_world_size()]

    if ARGS.text_only:
        raw_train = TinyLibraryTextRaw(train_recs)          # no images: modality control
        raw_val = TinyLibraryTextRaw(val_recs)
    else:
        raw_train = TinyLibraryRaw(train_recs, ARGS.image_root)
        raw_val = TinyLibraryRaw(val_recs, ARGS.image_root)
    train_view = OrderedView(raw_train, order_seq)
    train_dataset = VQADataset(train_view, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = VQADataset(raw_val, tokenizer, image_processor, vlm_cfg.mp_image_token_length)

    train_dataset = ConstantLengthDataset(
        train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size * 4, queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack)
    val_dataset = ConstantLengthDataset(
        val_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size * 4, queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    g = torch.Generator()
    g.manual_seed(ARGS.seed)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, collate_fn=collator,
                              num_workers=3, pin_memory=True, drop_last=True,
                              worker_init_fn=nanovlm_train.seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, collate_fn=collator,
                            num_workers=1, pin_memory=True, drop_last=True,
                            worker_init_fn=nanovlm_train.seed_worker, generator=g)

    iter_train, iter_val = iter(train_loader), iter(val_loader)
    next(iter_train), next(iter_val)  # warm up worker processes
    return train_loader, val_loader, iter_train, iter_val


def make_model(cfg, load_backbone=True):
    if ARGS.lm_init == "pretrained":
        return VisionLanguageModel(cfg, load_backbone=True)
    model = VisionLanguageModel(cfg, load_backbone=False)  # random-init LM + MP
    model.vision_encoder = ViT.from_pretrained(cfg)        # pretrained eyes, newborn brain
    return model


def get_run_name(train_cfg, vlm_cfg):
    date = time.strftime("%m%d-%H%M%S")
    modality = "textonly" if ARGS.text_only else "mm"
    return (f"tl_{ARGS.order}_{ARGS.schedule}_ep{ARGS.epochs}_{ARGS.lm_init}lm_{modality}_seed{ARGS.seed}_{date}")


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="manifest_scored.jsonl (or manifest.jsonl)")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--order", default="developmental",
                        choices=["developmental", "antidevelopmental", "readability", "perplexity",
                                 "random", "randomblock", "qwenppl", "gemmappl"])
    parser.add_argument("--schedule", default="staged", choices=["staged", "competence"])
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="corpus passes for staged/random; equivalent number of draws for competence sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--competence_c0", type=float, default=0.1)
    parser.add_argument("--lm_init", default="scratch", choices=["scratch", "pretrained"],
                        help="scratch = random LM initialization; pretrained = SmolLM2 weights (ablation)")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=2)  # 40GB A100 limit at 4096-token packed seqs
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr_mp", type=float, default=0.00512)
    parser.add_argument("--lr_vision_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_language_backbone", type=float, default=None,
                        help="default: 3e-4 for scratch, 5e-5 for pretrained")
    parser.add_argument("--max_img_size", type=int, default=512,
                        help="512 = single tile, 64 image tokens (no splitting)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="override the estimated packed-sequence step count")
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_path", default="checkpoints")
    parser.add_argument("--text_only", action="store_true",
                        help="modality control: train the decoder on the same text with NO image "
                             "tokens (the step limit is recomputed without image tokens)")
    parser.add_argument("--no_log_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "tinylibrary"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ARGS = parser.parse_args()

    torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARGS.seed)

    vlm_cfg = config.VLMConfig()
    vlm_cfg.max_img_size = ARGS.max_img_size
    vlm_cfg.vlm_checkpoint_path = ARGS.checkpoint_path
    vlm_cfg.hf_repo_name = None  # never auto-push

    train_cfg = config.TrainConfig()
    train_cfg.batch_size = ARGS.batch_size
    train_cfg.gradient_accumulation_steps = ARGS.gradient_accumulation_steps
    train_cfg.lr_mp = ARGS.lr_mp
    train_cfg.lr_vision_backbone = ARGS.lr_vision_backbone
    train_cfg.lr_language_backbone = ARGS.lr_language_backbone or (3e-4 if ARGS.lm_init == "scratch" else 5e-5)
    train_cfg.eval_interval = ARGS.eval_interval
    train_cfg.use_lmms_eval = False  # BabyLM eval runs separately on checkpoints
    train_cfg.log_wandb = not ARGS.no_log_wandb
    train_cfg.wandb_entity = ARGS.wandb_entity

    # Redirect nanoVLM's train() to our data, model factory and run naming.
    nanovlm_train.get_dataloaders = get_dataloaders
    nanovlm_train.VisionLanguageModel = make_model
    nanovlm_train.get_run_name = get_run_name
    if ARGS.text_only:
        # nanoVLM's _is_batch_valid drops any image-less batch (it guards against a
        # DDP deadlock: the vision encoder gets no gradient). We run text-only on a
        # single GPU, so that risk doesn't apply -- allow image-less batches through.
        def _textonly_step(train_loader, is_dist):
            for batch in train_loader:
                if batch and len(batch["input_ids"]) > 0:
                    yield batch
        nanovlm_train.synchronized_dataloader_step = _textonly_step
    if train_cfg.log_wandb:
        import wandb
        _init = wandb.init

        def _wandb_init(**kwargs):
            options = {**kwargs, "project": ARGS.wandb_project}
            if ARGS.wandb_entity:
                options["entity"] = ARGS.wandb_entity
            return _init(**options)

        nanovlm_train.wandb.init = _wandb_init

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        nanovlm_train.init_dist()
        import torch.distributed as dist
        nanovlm_train.PG_CPU = dist.new_group(backend="gloo")

    if nanovlm_train.is_master():
        print("--- Curriculum args ---")
        print(vars(ARGS))

    nanovlm_train.train(train_cfg, vlm_cfg)

    if nanovlm_train.is_dist():
        nanovlm_train.destroy_dist()


if __name__ == "__main__":
    main()
