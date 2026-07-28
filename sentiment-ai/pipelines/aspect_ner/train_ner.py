"""BertForTokenClassification 训练（可选，需 torch + transformers）。"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_ner(config: dict | None = None) -> dict:
    """训练 NER 并写出 metrics。
    入参 config: 默认读 configs/aspect_ner.json
    返回: { entity_f1, precision, recall, model_path, skipped? }
    """
    cfg_path = ROOT / "configs" / "aspect_ner.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if config:
        cfg.update(config)

    jsonl_path = ROOT / "data" / "processed" / "aspect_ner.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError("请先运行 run_pipeline.py 生成 aspect_ner.jsonl")

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        return {
            "skipped": True,
            "reason": f"缺少 torch/transformers: {exc}",
            "entity_f1": None,
            "model_path": None,
        }

    from pipelines.aspect_ner.constants import bio_labels_for_aspects
    from pipelines.aspect_ner.eval import entity_f1

    records = _load_jsonl(jsonl_path)
    seed = int(cfg.get("random_seed", 68))
    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * float(cfg.get("train_ratio", 0.8)))
    n_val = int(n * float(cfg.get("val_ratio", 0.1)))
    train_recs = records[:n_train]
    val_recs = records[n_train : n_train + n_val]
    test_recs = records[n_train + n_val :]

    label_list = bio_labels_for_aspects()
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    model_name = cfg.get("model_name", "bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class NerDataset(Dataset):
        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            tokens = item["tokens"]
            labels = item["labels"]
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=int(cfg.get("max_length", 128)),
                padding="max_length",
            )
            word_ids = enc.word_ids()
            label_ids = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    lab = labels[wid] if wid < len(labels) else "O"
                    label_ids.append(label2id.get(lab, label2id["O"]))
                else:
                    label_ids.append(-100)
                prev = wid
            enc["labels"] = label_ids
            return {k: torch.tensor(v) for k, v in enc.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = ROOT / cfg.get("artifacts_dir", "artifacts/aspect_ner") / "best"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir.parent / "checkpoints"),
        num_train_epochs=int(cfg.get("epochs", 3)),
        per_device_train_batch_size=int(cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 16)),
        learning_rate=float(cfg.get("learning_rate", 3e-5)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=NerDataset(train_recs),
        eval_dataset=NerDataset(val_recs) if val_recs else None,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # 测试集实体 F1
    y_true, y_pred = [], []
    model.eval()
    for item in test_recs:
        tokens = item["tokens"]
        true_labels = item["labels"]
        enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        word_ids = enc.word_ids()
        pred_labels = ["O"] * len(tokens)
        seen = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(tokens):
                pred_labels[wid] = id2label.get(pred_ids[i], "O")
        y_true.append(true_labels)
        y_pred.append(pred_labels)

    metrics = entity_f1(y_true, y_pred) if y_true else {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    result = {
        "entity_f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "model_path": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "test_size": len(test_recs),
        "target_f1": float(cfg.get("entity_f1_target", 0.85)),
        "passed": metrics["f1"] >= float(cfg.get("entity_f1_target", 0.85)),
    }

    metrics_path = ROOT / cfg.get("metrics_path", "artifacts/metrics/ner_f1.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
