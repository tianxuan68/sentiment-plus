"""生成路演 PNG：BIO 示例、5 类覆盖、NER F1 卡。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
PITCH_DIR = ROOT / "pitch_assets" / "no05_ner"

# 中文字体 fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name: str) -> Path:
    PITCH_DIR.mkdir(parents=True, exist_ok=True)
    path = PITCH_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_bio_example(ner_record: dict) -> Path:
    tokens = ner_record.get("tokens", [])
    labels = ner_record.get("labels", [])
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    ax.set_title("BIO 属性实体标注示例", fontsize=14, pad=12)

    colors = {
        "质量": "#1890ff",
        "价格": "#722ed1",
        "物流": "#13c2c2",
        "服务": "#52c41a",
        "包装": "#fa8c16",
    }
    x = 0.02
    y = 0.45
    for tok, lab in zip(tokens[:40], labels[:40]):
        aspect = lab[2:] if lab.startswith(("B-", "I-")) else None
        color = colors.get(aspect, "#f0f0f0") if aspect else "#fafafa"
        ax.text(x, y, tok, fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="#ccc"))
        x += min(0.045 + len(tok) * 0.012, 0.12)
        if x > 0.92:
            x = 0.02
            y -= 0.35

    patches = [mpatches.Patch(color=c, label=a) for a, c in colors.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    return _save(fig, "bio_example.png")


def render_aspect_cover(coverage: dict[str, int]) -> Path:
    aspects = list(coverage.keys())
    counts = [coverage[a] for a in aspects]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(aspects, counts, color=["#1890ff", "#722ed1", "#13c2c2", "#52c41a", "#fa8c16"])
    ax.set_title("五类属性实体覆盖（样本量）", fontsize=14)
    ax.set_ylabel("条数")
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=10)
    return _save(fig, "aspect_cover.png")


def render_ner_f1_card(metrics: dict) -> Path:
    f1 = metrics.get("entity_f1", metrics.get("f1", 0))
    prec = metrics.get("precision", 0)
    rec = metrics.get("recall", 0)
    passed = metrics.get("passed", f1 >= 0.85)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.72, "NER 实体 F1", ha="center", fontsize=16, color="#666")
    ax.text(0.5, 0.48, f"{f1 * 100:.1f}%", ha="center", fontsize=48, fontweight="bold", color="#1890ff")
    ax.text(0.5, 0.28, f"Precision {prec:.3f}  ·  Recall {rec:.3f}", ha="center", fontsize=12, color="#888")
    status = "达标 (≥85%)" if passed else "未达标"
    color = "#52c41a" if passed else "#ff4d4f"
    ax.text(0.5, 0.12, status, ha="center", fontsize=14, color=color, fontweight="bold")
    return _save(fig, "ner_f1_card.png")


def generate_pitch_assets(
    ner_records: list[dict],
    sentiment_records: list[dict],
    metrics: dict,
) -> list[Path]:
    from pipelines.aspect_ner.eval import aspect_coverage

    paths = []
    if ner_records:
        paths.append(render_bio_example(ner_records[0]))
    cov = aspect_coverage(sentiment_records)
    paths.append(render_aspect_cover(cov))
    paths.append(render_ner_f1_card(metrics))
    return paths
