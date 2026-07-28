"""写出 aspect_ner.jsonl 与 aspect_sentiment.csv。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def export_ner_jsonl(
    records: list[dict],
    path: str | Path = "data/processed/aspect_ner.jsonl",
) -> int:
    out = ROOT / path if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            line = {
                "id": rec["id"],
                "tokens": rec["tokens"],
                "labels": rec["labels"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return len(records)


def export_aspect_sentiment(
    records: list[dict],
    path: str | Path = "data/processed/aspect_sentiment.csv",
) -> int:
    out = ROOT / path if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "sentence", "aspect", "polarity", "polarity_text", "aspect_span"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})
    return len(records)
