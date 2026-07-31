"""
从魔塔 ModelScope 下载 bert-base-uncased 到本地。

目标目录:
    sentiment-ai/artifacts/pretrained/bert-base-uncased/

用法（在 sentiment-ai 根目录）:
    python -m pipelines.sentiment.bert.en.download_model

说明:
    1. 优先拉 safetensors；若只有旧版 pytorch_model.bin，会尝试转换键名并另存 model.safetensors
    2. 避免 transformers 5.x 加载 LayerNorm.gamma/beta 失败 / 静默未加载的问题
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = AI_ROOT / "artifacts" / "pretrained" / "bert-base-uncased"
# 魔塔上的英文 BERT 底座
DEFAULT_MODEL_ID = "google-bert/bert-base-uncased"


def convert_bin_to_safetensors(model_dir: Path) -> None:
    """若存在旧 bin，转换成可被 transformers 5 正确加载的 safetensors。"""
    bin_path = model_dir / "pytorch_model.bin"
    safe_path = model_dir / "model.safetensors"
    if safe_path.exists():
        print(f"[download] 已有 {safe_path.name}，跳过转换")
        return
    if not bin_path.exists():
        print("[download] 未找到 pytorch_model.bin，跳过转换")
        return

    import torch
    from transformers import BertConfig, BertForSequenceClassification

    print("[download] 转换 pytorch_model.bin -> model.safetensors ...")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    mapped = {}
    for k, v in sd.items():
        nk = k.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
            "LayerNorm.beta", "LayerNorm.bias"
        )
        if nk.startswith("cls."):
            continue
        mapped[nk] = v

    cfg = BertConfig.from_pretrained(str(model_dir), num_labels=2)
    model = BertForSequenceClassification(cfg)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"[download] missing={missing} unexpected={unexpected}")
    model.save_pretrained(str(model_dir))
    # 保留原 bin 备份，避免 transformers 误优先读坏键名
    bak = model_dir / "pytorch_model.bin.bak"
    if not bak.exists():
        bin_path.rename(bak)
        print(f"[download] 原 bin 已备份为 {bak.name}")
    print("[download] safetensors 转换完成")


def download(model_id: str, out_dir: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] model_id={model_id}")
    print(f"[download] out_dir={out_dir}")

    # 只拉训练/推理需要的文件（只要 safetensors，不拉 bin/onnx/flax）
    allow_patterns = [
        "config.json",
        "configuration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
    ]
    cache_path = snapshot_download(
        model_id,
        cache_dir=str(out_dir.parent / ".modelscope_cache"),
        allow_file_pattern=allow_patterns,
    )
    cache_path = Path(cache_path)
    print(f"[download] cache -> {cache_path}")

    # 同步到目标目录
    for p in cache_path.iterdir():
        if p.is_file():
            # 跳过明显用不到的大文件
            if p.suffix in {".onnx", ".msgpack", ".mlmodel", ".h5"}:
                continue
            if p.name.startswith("tf_model") or p.name.startswith("rust_model"):
                continue
            shutil.copy2(p, out_dir / p.name)
        elif p.is_dir() and p.name not in {".cache", "._____temp"}:
            dest = out_dir / p.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(p, dest)

    convert_bin_to_safetensors(out_dir)
    print("[download] 目录文件:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name} ({p.stat().st_size / 1024 / 1024:.2f} MB)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="魔塔下载 bert-base-uncased")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    download(args.model_id, args.out_dir)
    print("[download] 完成。配置已指向: artifacts/pretrained/bert-base-uncased")


if __name__ == "__main__":
    main()
