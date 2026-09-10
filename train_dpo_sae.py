"""
Usage:
    python train_dpo.py --smoke          # 50 examples, 10 steps, sanity check
    python train_dpo.py                  # real run
    python train_dpo.py --anchor ./sft-gpt2 --out ./dpo-from-sft
"""

import argparse
import os
import torch

ANCHOR = "gpt2"
OUT = "./gpt2-dpo-hh"


def split_dialogue(example):
    """Split the shared prefix off a chosen/rejected pair.

    hh-rlhf rows look like:
        "\n\nHuman: how do I ...\n\nAssistant: well ...\n\nHuman: ok\n\nAssistant: sure"

    chosen and rejected share everything up to the FINAL "\n\nAssistant:";
    only the last turn differs. So we split there.
    """
    marker = "\n\nAssistant:"
    chosen, rejected = example["chosen"], example["rejected"]

    idx = chosen.rfind(marker)
    if idx == -1:
        return {"prompt": "", "chosen": "", "rejected": ""}   # dropped below

    prompt = chosen[: idx + len(marker)]

    # Guard: the two must actually share the prefix. A handful of rows in
    # hh-rlhf have differing earlier turns - those are unusable and get dropped.
    if not rejected.startswith(prompt):
        return {"prompt": "", "chosen": "", "rejected": ""}

    return {
        "prompt": prompt,
        "chosen": chosen[idx + len(marker):],
        "rejected": rejected[idx + len(marker):],
    }


def build_dataset(split="train", n=None, max_prompt_chars=1500):
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))

    before = len(ds)
    ds = ds.map(split_dialogue, remove_columns=ds.column_names)

    ds = ds.filter(
        lambda x: len(x["prompt"]) > 0
        and len(x["chosen"].strip()) > 0
        and len(x["rejected"].strip()) > 0
        and x["chosen"].strip() != x["rejected"].strip()
        and len(x["prompt"]) < max_prompt_chars      # GPT-2 ctx is only 1024 tokens
    )
    print(f"[data] {split}: {before} -> {len(ds)} usable ({100*len(ds)/before:.1f}%)")
    return ds


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(args.anchor)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.anchor, torch_dtype=dtype)

    train_ds = build_dataset("train", n=50 if args.smoke else None)
    eval_ds = build_dataset("test", n=20 if args.smoke else 500)

    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,                         # KL strength; 0.1 is standard
        learning_rate=args.lr,                  # DPO wants ~10x lower LR than SFT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1,

        # --- 6GB VRAM settings ---
        per_device_train_batch_size=2,          # DPO forwards chosen AND rejected,
        gradient_accumulation_steps=8,          # so activation batch is 2x this
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,
        bf16=cuda,
        precompute_ref_log_probs=True,          # <-- the big one: frees the ref model

        # GPT-2 context is 1024 total
        max_length=512,
        max_prompt_length=384,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )

    kwargs = dict(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds)
    try:                                        # TRL renamed this around 0.12
        trainer = DPOTrainer(**kwargs, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(**kwargs, tokenizer=tok)

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\n[done] checkpoint -> {os.path.abspath(args.out)}")
    print("Metrics to check: rewards/accuracies should climb above 0.5;")
    print("rewards/margins should grow. If margins explode, lower beta or LR.")


if __name__ == "__main__":
    main()