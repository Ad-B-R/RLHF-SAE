import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 7                                  # mid-stack; see note at bottom
HOOK = f"blocks.{LAYER}.hook_resid_pre"


def load_hf(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token          # GPT-2 ships no pad token; TRL needs one
    tok.padding_side = "left"              # correct side for decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    return model, tok

def load_tlens(model_name="gpt2-small"):
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


def load_tlens_from_finetuned(hf_model, tok):
    """Wrap YOUR dpo'd checkpoint in TransformerLens.

    This is the step people miss: after DPO you have an HF checkpoint, and you
    must fold it into a HookedTransformer to analyse it. Pass the HF model in
    explicitly - `from_pretrained` alone would silently give you stock GPT-2.
    """
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(
        "gpt2-small", hf_model=hf_model, tokenizer=tok, device=DEVICE
    )

def load_sae(hook=HOOK):
    from sae_lens import SAE
    res = SAE.from_pretrained(
        release="gpt2-small-res-jb", sae_id=hook, device=DEVICE
    )
    return res[0] if isinstance(res, tuple) else res   # signature varies by version

def demo():
    model = load_tlens()
    sae = load_sae()

    text = ["The doctor said that", "I really think you should"]
    _, cache = model.run_with_cache(text)

    acts = cache[HOOK]                     # [batch, seq, d_model]
    feats = sae.encode(acts)               # [batch, seq, d_sae]
    recon = sae.decode(feats)

    l0 = (feats > 0).float().sum(-1).mean().item()
    mse = torch.nn.functional.mse_loss(recon, acts).item()
    var = acts.var().item()

    print(f"acts  {tuple(acts.shape)}")
    print(f"feats {tuple(feats.shape)}  (d_sae = {sae.cfg.d_sae})")
    print(f"L0 {l0:.1f}  |  explained var {1 - mse/var:.3f}")

    # top firing features on the last token of the first prompt
    top = feats[0, -1].topk(5)
    print("top features:", top.indices.tolist())
    print("neuronpedia:  gpt2-small/{}-res-jb/<feature_id>".format(LAYER))


if __name__ == "__main__":
    demo()