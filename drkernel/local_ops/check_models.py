#!/usr/bin/env python3
# check models' completeness and report basic info without loading the full model into memory 

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_model_dir(model_dir: Path) -> dict:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer config file: {tokenizer_config_path}")

    index_data = json.loads(index_path.read_text())
    weight_map = index_data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (model_dir / name).exists()]
    if missing_shards:
        raise FileNotFoundError(f"Missing shard files: {missing_shards}")

    shard_sizes = {name: (model_dir / name).stat().st_size for name in shard_names}
    total_size = sum(shard_sizes.values())

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    return {
        "model_dir": str(model_dir),
        "config_class": config.__class__.__name__,
        "model_type": getattr(config, "model_type", "unknown"),
        "vocab_size": getattr(config, "vocab_size", "unknown"),
        "tokenizer_class": tokenizer.__class__.__name__,
        "shard_count": len(shard_names),
        "total_shard_size_gb": round(total_size / (1024**3), 2),
        "largest_shard_gb": round(max(shard_sizes.values()) / (1024**3), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight local validation for downloaded HF model directories.")
    parser.add_argument("model_dirs", nargs="+", help="One or more local model directories to validate.")
    args = parser.parse_args()

    for raw_dir in args.model_dirs:
        model_dir = Path(raw_dir).expanduser().resolve()
        print(f"=== Checking {model_dir} ===")
        result = check_model_dir(model_dir)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("status: OK")
        print()


if __name__ == "__main__":
    main()
