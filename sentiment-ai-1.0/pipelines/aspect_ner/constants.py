"""5号 · 属性 NER：实体类型、BIO 标签与触发词表。"""

from __future__ import annotations

# README / 任务细化：≥5 类固定属性
ASPECT_TYPES: tuple[str, ...] = ("质量", "价格", "物流", "服务", "包装")

#  aspect -> 文本中出现即视为该属性的触发词（组内可扩展，类型名不可改）
ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "质量": ("质量", "味道", "口感", "材质", "做工", "新鲜", "分量"),
    "价格": ("价格", "价钱", "贵", "便宜", "性价比", "实惠", "划算"),
    "物流": ("物流", "快递", "发货", "配送", "速度", "送达", "包邮"),
    "服务": ("服务", "客服", "态度", "售后", "耐心", "热情"),
    "包装": ("包装", "外观", "盒子", "精美", "简陋", "完好"),
}

NEG_HINTS = ("差", "烂", "慢", "贵", "简陋", "失望", "退货", "破损", "咸", "油", "脏", "冷")
POS_HINTS = ("好", "快", "赞", "满意", "不错", "喜欢", "精美", "实惠", "耐心", "新鲜", "完好")

BIO_LABEL_O = "O"


def bio_labels_for_aspects() -> list[str]:
    labels = [BIO_LABEL_O]
    for aspect in ASPECT_TYPES:
        labels.append(f"B-{aspect}")
        labels.append(f"I-{aspect}")
    return labels


def aspect_from_bio(label: str) -> str | None:
    if label == BIO_LABEL_O or "-" not in label:
        return None
    return label.split("-", 1)[1]


def polarity_text(polarity: int) -> str:
    return "正向" if polarity == 1 else "负向"
