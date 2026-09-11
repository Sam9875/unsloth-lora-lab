"""LoRA recipe. --dry-run never downloads weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS = [
    {"title": "EU battery recycling draft", "abstract": "Targets rise for lithium and nickel recovery."},
    {"title": "Cold-start news ranking", "abstract": "Two-tower retrieval plus a popularity prior."},
]


def dry_run() -> dict:
    steps = [
        "load 4-bit base (skipped)",
        "attach LoRA rank=8 on q_proj, v_proj",
        f"sft on {len(PAIRS)} title→abstract pairs",
        "save adapter to artifacts/lora/",
    ]
    report = {
        "mode": "dry-run",
        "rank": 8,
        "targets": ["q_proj", "v_proj"],
        "pairs": len(PAIRS),
        "eval": "rougeL on held-out abstracts",
        "steps": steps,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/dry-run.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(dry_run() if args.dry_run else dry_run(), indent=2))
