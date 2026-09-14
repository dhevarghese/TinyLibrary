"""Export a trained nanoVLM checkpoint's language decoder to a standard
HuggingFace LlamaForCausalLM so the BabyLM evaluation pipeline (babylm-eval)
can treat it as a plain causal LM.

Inverts the key mapping used by nanoVLM's LanguageModel.from_pretrained.
The tokenizer (SmolLM2 + nanoVLM's 66 extra image tokens) is saved alongside.

Run with nanoVLM root on PYTHONPATH:
    PYTHONPATH=nanoVLM:training python export_decoder_hf.py \
        --checkpoint checkpoints/tl_developmental_.../step_2250 --out hf_exports/tl_developmental_...
"""
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from models.vision_language_model import VisionLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="nanoVLM save_pretrained directory")
parser.add_argument("--out", required=True, help="output HF model directory")
args = parser.parse_args()

vlm = VisionLanguageModel.from_pretrained(args.checkpoint)
decoder_sd = vlm.decoder.state_dict()
cfg = vlm.cfg

hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
hf_config.vocab_size = cfg.lm_vocab_size  # includes the 66 extra VLM tokens
hf_config.tie_word_embeddings = cfg.lm_tie_weights
hf_model = LlamaForCausalLM(hf_config)

mapping = {
    "token_embedding.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
}
for i in range(cfg.lm_n_blocks):
    ours, hf = f"blocks.{i}.", f"model.layers.{i}."
    mapping.update({
        f"{ours}attn.q_proj.weight": f"{hf}self_attn.q_proj.weight",
        f"{ours}attn.k_proj.weight": f"{hf}self_attn.k_proj.weight",
        f"{ours}attn.v_proj.weight": f"{hf}self_attn.v_proj.weight",
        f"{ours}attn.out_proj.weight": f"{hf}self_attn.o_proj.weight",
        f"{ours}mlp.gate_proj.weight": f"{hf}mlp.gate_proj.weight",
        f"{ours}mlp.up_proj.weight": f"{hf}mlp.up_proj.weight",
        f"{ours}mlp.down_proj.weight": f"{hf}mlp.down_proj.weight",
        f"{ours}norm1.weight": f"{hf}input_layernorm.weight",
        f"{ours}norm2.weight": f"{hf}post_attention_layernorm.weight",
    })
if not cfg.lm_tie_weights:
    mapping["head.weight"] = "lm_head.weight"

hf_sd = hf_model.state_dict()
missing = []
with torch.no_grad():
    for our_key, hf_key in mapping.items():
        if our_key not in decoder_sd or hf_key not in hf_sd:
            missing.append((our_key, hf_key))
            continue
        assert decoder_sd[our_key].shape == hf_sd[hf_key].shape, (our_key, decoder_sd[our_key].shape, hf_sd[hf_key].shape)
        hf_sd[hf_key].copy_(decoder_sd[our_key])
hf_model.load_state_dict(hf_sd)
if cfg.lm_tie_weights:
    hf_model.tie_weights()
assert not missing, f"unmapped keys: {missing}"

tokenizer = AutoTokenizer.from_pretrained(cfg.lm_tokenizer,
                                          extra_special_tokens=cfg.vlm_extra_tokens)
hf_model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)

# Round-trip sanity check: same next-token logits from both implementations.
prompt = "The little bear looked at the"
ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
vlm.decoder.lm_use_tokens = True  # ids in, logits out
with torch.no_grad():
    ours = vlm.decoder(ids)[0][:, -1]
    theirs = hf_model(ids).logits[:, -1]
diff = (ours - theirs).abs().max().item()
print(f"exported to {args.out}; max |logit diff| = {diff:.2e}")
assert diff < 1e-3, "decoder export does not match nanoVLM forward pass"
