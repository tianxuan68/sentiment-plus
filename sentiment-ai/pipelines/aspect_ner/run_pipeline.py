"""5号 · 属性 NER 一键流水线。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.aspect_ner.annotate import annotate_bio, build_annotations_from_pool, load_annotation_pool
from pipelines.aspect_ner.eval import aspect_coverage, entity_f1
from pipelines.aspect_ner.export import export_aspect_sentiment, export_ner_jsonl
from pipelines.aspect_ner.pitch_assets import generate_pitch_assets
from pipelines.aspect_ner.predict import predict_entities
from pipelines.aspect_ner.train_ner import train_ner


def _load_sentence_labels() -> dict[str, int]:
    """从 train.csv 补全句级 label（若标注池带 label 列则优先）。"""
    pool_path = ROOT / "data" / "samples" / "aspect_annotate_pool.csv"
    labels: dict[str, int] = {}
    if not pool_path.exists():
        return labels
    with pool_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label" in row and row["label"] not in (None, ""):
                labels[str(row["id"])] = int(row["label"])
    if labels:
        return labels

    train_path = ROOT / "data" / "raw" / "train.csv"
    if not train_path.exists():
        return labels
    with train_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            labels[str(i)] = int(row.get("label", 1))
    return labels


def run_annotate() -> tuple[list[dict], list[dict]]:
    pool = load_annotation_pool()
    labels_map = _load_sentence_labels()
    ner_records, sentiment_records = build_annotations_from_pool(pool, labels_map)
    n_ner = export_ner_jsonl(ner_records)
    n_sent = export_aspect_sentiment(sentiment_records)
    print(f"[annotate] aspect_ner.jsonl: {n_ner} 条")
    print(f"[annotate] aspect_sentiment.csv: {n_sent} 行")
    print(f"[annotate] 属性覆盖: {aspect_coverage(sentiment_records)}")
    return ner_records, sentiment_records


def run_eval_baseline(ner_records: list[dict]) -> dict:
    """标注一致性 + 规则 predict 覆盖（验收基线）。"""
    y_true, y_pred = [], []
    for rec in ner_records:
        sentence = rec.get("sentence") or "".join(rec["tokens"])
        re = annotate_bio(str(rec["id"]), sentence)
        y_true.append(rec["labels"])
        y_pred.append(re["labels"])
    metrics = entity_f1(y_true, y_pred)
    metrics["entity_f1"] = metrics["f1"]
    metrics["passed"] = metrics["f1"] >= 0.85
    metrics["mode"] = "annotation_consistency"
    metrics["note"] = "完整 BERT 训练后请查看 train_ner 写出的 test F1"
    out = ROOT / "artifacts" / "metrics" / "ner_f1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] rule baseline F1={metrics['f1']:.4f} passed={metrics['passed']}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="5号 · aspect_ner 流水线")
    parser.add_argument("--skip-train", action="store_true", help="跳过 BERT 训练（仅标注+导出）")
    parser.add_argument("--train-only", action="store_true", help="仅训练（需已有 jsonl）")
    args = parser.parse_args()

    if not args.train_only:
        ner_records, sentiment_records = run_annotate()
        metrics = run_eval_baseline(ner_records)
    else:
        ner_path = ROOT / "data" / "processed" / "aspect_ner.jsonl"
        ner_records = []
        with ner_path.open("r", encoding="utf-8") as f:
            for line in f:
                ner_records.append(json.loads(line))
        sentiment_records = []
        metrics_path = ROOT / "artifacts" / "metrics" / "ner_f1.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    if not args.skip_train:
        train_result = train_ner()
        if train_result.get("skipped"):
            print(f"[train] 跳过: {train_result.get('reason')}")
        else:
            metrics = train_result
            print(f"[train] entity_f1={train_result.get('entity_f1')} passed={train_result.get('passed')}")

    if not args.train_only:
        paths = generate_pitch_assets(ner_records, sentiment_records, metrics)
        print(f"[pitch] 已生成: {[str(p.relative_to(ROOT)) for p in paths]}")

    print("[done] 5号 aspect_ner 流水线完成")


if __name__ == "__main__":
    main()
