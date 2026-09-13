"""
Two environments:
  --device cpu    laptop, no GPU  -> fp32, small batches, code-check only
  --device cuda   3050 6GB        -> bf16, real runs
  --device auto   pick whichever is available (default)
Usage:
    python load_models.py --device cpu
    python load_models.py --device cuda --layer 7
"""

import argparse
import torch

HOOK_TMPL = "blocks.{}.hook_resid_pre"


# --------------------------------------------------------------------------
def resolve_device(choice="auto"):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")
    return choice


def dtype_for(device):
    # bf16 on the 3050 (Ampere, native support); fp32 on CPU - bf16 on CPU is
    # emulated and slower, not a saving.
    return torch.bfloat16 if device == "cuda" else torch.float32


def budget(device):
    """Batch sizes that fit. CPU numbers are for smoke-testing, not throughput."""
    if device == "cuda":
        return dict(tlens_batch=8, sae_batch=4096, n_seqs=64)   # 6GB
    return dict(tlens_batch=2, sae_batch=256, n_seqs=4)         # no-GPU laptop


# --------------------------------------------------------------------------
# 1. HF form - for DPO training
# --------------------------------------------------------------------------
def load_hf(model_name="gpt2", device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token        # GPT-2 ships no pad token; TRL needs one
    tok.padding_side = "left"            # correct side for decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype_for(device)
    ).to(device)
    return model, tok


# --------------------------------------------------------------------------
# 2. TransformerLens form - for activations, patching, ablation
# --------------------------------------------------------------------------
def load_tlens(model_name="gpt2-small", device="cpu"):
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(
        model_name, device=device, dtype=dtype_for(device)
    )


def load_tlens_from_finetuned(hf_model, tok, device="cpu"):
    """Wrap YOUR dpo'd checkpoint in TransformerLens.

    The step people miss: after DPO you have an HF checkpoint and must fold it
    into a HookedTransformer. Passing hf_model= is mandatory - without it you
    silently get stock GPT-2 and every comparison downstream is base-vs-base.
    """
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(
        "gpt2-small", hf_model=hf_model, tokenizer=tok,
        device=device, dtype=dtype_for(device),
    )


def load_sae(hook, device="cpu"):
    from sae_lens import SAE
    res = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=hook, device=device)
    return res[0] if isinstance(res, tuple) else res   # signature varies by version


# --------------------------------------------------------------------------
def demo(device, layer):
    hook = HOOK_TMPL.format(layer)
    b = budget(device)
    print(f"[env] device={device} dtype={dtype_for(device)} hook={hook}")
    print(f"[env] budget={b}")

    model = load_tlens(device=device)
    sae = load_sae(hook, device=device)

    text = ["The doctor said that", "I really think you should"][: b["tlens_batch"]]
    _, cache = model.run_with_cache(text)

    acts = cache[hook].float()             # cast: SAE may be fp32 while model is bf16
    feats = sae.encode(acts)
    recon = sae.decode(feats)

    l0 = (feats > 0).float().sum(-1).mean().item()
    mse = torch.nn.functional.mse_loss(recon, acts).item()

    print(f"acts  {tuple(acts.shape)}")
    print(f"feats {tuple(feats.shape)}  (d_sae={sae.cfg.d_sae})")
    print(f"L0 {l0:.1f}  |  explained var {1 - mse/acts.var().item():.3f}")
    print("top features:", feats[0, -1].topk(5).indices.tolist())
    print(f"neuronpedia: gpt2-small/{layer}-res-jb/<feature_id>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--layer", type=int, default=7)
    args = ap.parse_args()
    demo(resolve_device(args.device), args.layer)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Layer choice
# --------------------------------------------------------------------------
# GPT-2 small has 12 layers. Early = token-level, late = output-shaped,
# mid = abstract features. Layers 5-9 are the target band.
#
# Run the pipeline at ~3 layers, not one: a null result at a single layer can't
# distinguish "no feature change" from "wrong layer", and that's unfalsifiable.
#
# Hook points per layer:
#   blocks.N.hook_resid_pre    <- start here, best SAE coverage
#   blocks.N.hook_resid_post
#   blocks.N.hook_mlp_out
#   blocks.N.attn.hook_z