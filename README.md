# In-Progress
# DPO-SAE

How does Direct Preference Optimization change what a language model represents internally?

This project fine-tunes GPT-2 small in two stages- SFT, and DPO on Anthropic's hh-rlhf
`harmless-base` data, and will compare the two checkpoints through one shared Sparse
Autoencoder.

```
GPT-2 small ──SFT──▶ sft-gpt2-hh-21k/final ──DPO──▶ dpo-gpt2-hh-21k/final
 (stock)              (the anchor)                   (the DPO'd side)
```

**Status:** SFT and DPO are done; both checkpoints are exported. The SAE stage has not started.

---

## Layout

| File | What it is |
|---|---|
| [src/sft_gpt2_hh.ipynb](src/sft_gpt2_hh.ipynb) | Stage 1 — SFT on the `chosen` completions. Writes the anchor. |
| [src/dpo_gpt2_hh.ipynb](src/dpo_gpt2_hh.ipynb) | Stage 2 — DPO from the anchor, on data SFT never saw. |
| [src/load_model.py](src/load_model.py) | Exploratory loader for the upcoming SAE stage. Not part of training. |
| `src/sae_gpt2_*.ipynb` | Empty placeholders. |
| [CLAUDE.md](CLAUDE.md) | Design notes and the reasoning behind each choice. |

Both notebooks are **Colab-only**. They mount Drive and write everything under
`/content/drive/MyDrive/DPO-SAE/`.

Stack: `transformers`, `trl==1.13.0`, `datasets`, `accelerate`, `wandb` (project `DPO-SAE`),
`torch`. Add `WANDB_API_KEY` in Colab's secrets panel — never paste it into a cell.

---

## Why two stages

hh-rlhf contains no demonstration data, and stock GPT-2 has never seen its
`Human:`/`Assistant:` format. Running DPO straight from stock GPT-2 would mean the reference
model is off-distribution, and most of the weight change would be the model learning the
format rather than the preference- which would contaminate the feature comparison the whole
project rests on. Hence, incorporating 2 different stages- 

1. SFT
2. DPO 
---

## Data

`Anthropic/hh-rlhf`, `harmless-base`, train split — 42,537 preference pairs. Each row is two
complete dialogues, identical except for the final assistant response; a human labelled one
better.

**The two stages use disjoint halves.**

```python
shuffled = raw.shuffle(seed=1337)          # deterministic
sft = shuffled.select(range(0, 21268))     # ~20.4k pairs after filtering
dpo = shuffled.select(range(21268, 42537)) # ~20.4k pairs after filtering
```

Same seed always gives the same ordering, so the two ranges cannot share a row. SFT writes the
seed and both ranges to its manifest; the DPO notebook reads them from there rather than
retyping them.

This matters because DPO's reference model *is* the SFT checkpoint. If the halves overlapped,
the reference would already have memorised those `chosen` completions, shrinking the logprob
gap the DPO loss works on.

Rows are split at the last `"\n\nAssistant:"` — everything before is the prompt, everything
after is the completion. Rows are dropped when the marker is missing, when the two dialogues
don't share a prompt, when a completion is empty, or when the prompt leaves no room for a real
response within the 512-token budget. Roughly 98% survive. A fixed 512-pair validation slice
comes off each half.

---

## The anchor

`sft-gpt2-hh-21k/final/` is the most load-bearing artifact here. It is used three ways:

- the starting weights for DPO,
- the frozen reference π_ref inside the DPO loss,
- the model the SAE will be trained on.

If it changes, those three stop referring to the same model and every feature comparison
becomes invalid. **Never overwrite or retrain it.** The DPO notebook hashes its weights before
and after the run and refuses to write anywhere inside the SFT directory.

It is a plain HF export — `config.json`, `model.safetensors` (~498 MB, fp32), tokenizer files.
Load it like any other model.

---
<!-- 
## What was run

**Stage 1 — SFT.** 20,421 training pairs, one epoch, ~639 steps, ~25 min on a T4.

| | |
|---|---|
| Objective | Causal LM on prompt + chosen |
| Learning rate | 5e-5, cosine, 3% warmup |
| Effective batch | 32 pairs/step |
| Max length | 512 |

**Stage 2 — DPO.** ~20,400 training pairs, one epoch, 1,276 steps, ~30 min on a T4.

| | | |
|---|---|---|
| β | 0.1 | KL strength against the reference. Rafailov et al.'s hh-rlhf setting. |
| Learning rate | 5e-6 | ~10× below SFT — DPO overshoots easily. |
| Effective batch | 16 pairs/step | Halved: chosen and rejected both go forward, so the real forward batch is 8. |
| Loss | vanilla sigmoid DPO | |
| Max length | 512 | |

Both runs used seed 1337 and fp16 on a T4 (bf16 is selected automatically on Ampere cards).
DPO caches the reference logprobs in one pass up front, so no second model stays in memory —
this is what makes the run fit on a small GPU, and why constructing the trainer takes a few
minutes before training starts.

Exact counts, hyperparameters and final metrics for each run are in its `manifest.json`.

Each run writes intermediate checkpoints alongside `final/`, plus a rolling `_resume/` state
for Colab disconnects (deletable once `final/` exists).

---

## Did DPO work?

The loss curve won't tell you. Check the eval metrics:

- **`rewards/accuracies`** — should climb clearly above 0.5. Flat means no preference learned.
- **`rewards/margins`** — should grow steadily. A sharp early spike means β or the LR is too
  high; lower β first.
- **`rewards/chosen` vs `rewards/rejected`** — should diverge. Both drifting negative with
  `rejected` falling faster is normal.
- **Read the generations.** Rewards can rise while the text degrades.

---

## Running it

1. Open `src/sft_gpt2_hh.ipynb` in Colab on a GPU runtime, add the `WANDB_API_KEY` secret, run
   all cells (~25 min).
2. Open `src/dpo_gpt2_hh.ipynb` and run all cells. It needs the SFT run's `final/` and
   `manifest.json` on Drive (~30 min).
3. After a disconnect during DPO, set `RESUME = True` and re-run from the top.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "/content/drive/MyDrive/DPO-SAE/dpo-gpt2-hh-21k/final"   # or sft-gpt2-hh-21k/final
model = AutoModelForCausalLM.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)
```

---

## Next: the SAE stage

Train **one** SAE on the anchor's activations, then apply it unchanged to both checkpoints.
Sharing the dictionary is what makes the comparison meaningful — feature #N is the same
direction on both sides. Two separately trained SAEs would confound "DPO changed the model"
with "the two SAEs trained differently."

Candidate hook points are the mid-stack residual stream, `blocks.{5..9}.hook_resid_pre`.

Feature differences alone are only correlational. The result the project is aiming at needs the
causal step: ablate the features that changed and show the behaviour moves with them.

---
-->

## References

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model* (2023).
- Bai et al., *Training a Helpful and Harmless Assistant with RLHF* (2022) — the source of
  hh-rlhf. 