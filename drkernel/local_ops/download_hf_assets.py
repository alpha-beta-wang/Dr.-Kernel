#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PRESETS = {
    "datasets": [
        ("hkust-nlp/drkernel-coldstart-8k", "dataset"),
        ("hkust-nlp/drkernel-rl-data", "dataset"),
        ("hkust-nlp/drkernel-validation-data", "dataset"),
    ],
    "models": [
        ("Qwen/Qwen3-14B-Base", "model"),
        ("hkust-nlp/drkernel-14b", "model"),
    ],
}
PRESETS["all"] = PRESETS["datasets"] + PRESETS["models"]


def download(repo_id: str, repo_type: str, root: Path, revision: str | None) -> None:
    local_dir = root / repo_id
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo_id={repo_id} repo_type={repo_type} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dr.Kernel datasets/models from Hugging Face.")
    parser.add_argument("--root", required=True, help="Destination root directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--repo-id", action="append", default=[], help="Extra repo ids to download.")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    jobs: list[tuple[str, str]] = []
    if args.preset:
        jobs.extend(PRESETS[args.preset])
    jobs.extend((repo_id, args.repo_type) for repo_id in args.repo_id)

    if not jobs:
        raise SystemExit("No assets requested. Use --preset or --repo-id.")

    for repo_id, repo_type in jobs:
        download(repo_id=repo_id, repo_type=repo_type, root=root, revision=args.revision)


if __name__ == "__main__":
    main()
