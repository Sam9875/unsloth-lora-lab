# Unsloth-style LoRA lab

Study of [unslothai/unsloth](https://github.com/unslothai/unsloth): 4-bit base + LoRA adapters on **one GPU**.

I do not vendor Unsloth. This repo is the **recipe** I would run: news title → abstract, plus a dry-run trainer that executes without a GPU.

## Architecture

```
MIND-style pairs → tokenize → 4-bit base (frozen) → LoRA q/v → SFT → adapter save
```

Change: data is news title/abstract (Column / MIND), not alpaca. Eval is ROUGE on held-out abstracts, not a vibe check.

## Run (dry)

```bash
python -m src.train --dry-run
```

Author: Samesun Singh ([Sam9875](https://github.com/Sam9875))
