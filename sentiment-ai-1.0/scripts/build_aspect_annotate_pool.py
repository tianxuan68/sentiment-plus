"""从 train.csv 抽样 ≥3000 句，供 5 号标注（杨国东未交付时的兜底脚本）。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample_for_annotation(
    train_path: str | Path = "data/raw/train.csv",
    n: int = 3000,
    seed: int = 68,
    output_path: str | Path = "data/samples/aspect_annotate_pool.csv",
) -> dict:
    train_p = ROOT / train_path
    out_p = ROOT / output_path
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with train_p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sentence = (row.get("sentence") or "").strip()
            if len(sentence) < 6:
                continue
            rows.append(
                {
                    "id": str(i),
                    "sentence": sentence,
                    "label": row.get("label", ""),
                }
            )

    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))

    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sentence", "label"])
        writer.writeheader()
        writer.writerows(sampled)

    return {"sampled": len(sampled), "seed": seed, "output_path": str(out_p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=68)
    args = parser.parse_args()
    result = sample_for_annotation(n=args.n, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
