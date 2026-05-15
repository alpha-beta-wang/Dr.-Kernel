#!/usr/bin/env python3
"""Dr.Kernel config loader - reads YAML, applies overrides, outputs bash exports.
Usage:
    eval $(python3 config/load_config.py rl)
    eval $(python3 config/load_config.py eval)
    python3 config/load_config.py rl --help
Priority: CLI args > existing env vars > YAML defaults
"""

import os
import sys
from pathlib import Path

DESCRIPTIONS = {
    "DRKERNEL_DATA_ROOT": "data root dir (models+datasets+checkpoints+results)",
    "TRAIN_BATCH_SIZE": "training batch size, must >= GPU count",
    "PPO_MINI_BATCH_SIZE": "PPO mini batch size, must >= GPU count",
    "MAX_PROMPT_LENGTH": "max prompt length in tokens",
    "MAX_RESPONSE_LENGTH": "max response length in tokens",
    "ROLLOUT_GPU_MEMORY_UTIL": "vLLM GPU memory utilization (0.0-1.0)",
    "ROLLOUT_MAX_BATCHED_TOKENS": "vLLM max batched tokens",
    "ROLLOUT_MODE": "rollout mode (sync/async_vllm)",
    "ROLLOUT_N": "number of samples per prompt",
    "ALGORITHM": "RL algorithm (trloo/grpo/ppo etc.)",
    "LEARNING_RATE": "learning rate",
    "ENABLE_MULTI_TURN": "enable multi-turn dialogue",
    "MAX_TURN": "max dialogue turns (training)",
    "VAL_MAX_TURN": "max dialogue turns (validation)",
    "FREE_CACHE_ENGINE": "free vLLM cache before training",
    "ACTOR_PARAMETER_OFFLOAD": "actor parameter offload to CPU",
    "ACTOR_OPTIMIZER_OFFLOAD": "optimizer state offload to CPU",
    "N_VAL": "number of validation samples",
    "VAL_BEFORE_TRAIN": "run validation before training",
    "ENFORCE_EAGER": "disable CUDA graph for compatibility",
    "N_SAMPLES": "samples per evaluation case",
    "BATCH_SIZE": "evaluation batch size",
    "ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE": "tensor parallelism size (= GPU count)",
    "FSDP_SIZE": "FSDP shard count (= GPU count)",
    "MULTI_TURN": "multi-turn evaluation",
    "MAX_USER_TURNS": "max user turns",
    "NUM_CORRECT_TRIALS": "correctness trial count",
    "REWARD_WEIGHTS": "reward weights (speedup_correctness_compile)",
    "MICRO_BATCH_SIZE_PER_GPU": "micro batch size per GPU",
    "MAX_LENGTH": "max sequence length in tokens",
    "SP_SIZE": "Ulysses sequence parallelism size",
    "TOTAL_EPOCHS": "total training epochs",
    "SAVE_FREQ": "checkpoint save frequency (steps)",
    "CPU_OFFLOAD": "CPU offload for memory saving",
    "OFFLOAD_PARAMS": "parameter offload for memory saving",
    "WANDB_MODE": "WandB mode (offline/online/disabled)",
    "GPU_MONITOR_INTERVAL": "GPU monitor interval in seconds",
    "PYTORCH_CUDA_ALLOC_CONF": "CUDA memory allocator config",
    "PROMPT_OVERSAMPLING_FACTOR": "prompt oversampling factor",
    "SAMPLE_OVERSAMPLING_FACTOR": "sample oversampling factor",
    "REWARD_MANAGER": "reward manager type",
    "REWARD_FUNC_NAME": "reward function name",
    "NUM_PERF_TRIALS": "number of performance trials",
    "IS_GET_LAST_TURN": "use only last turn result",
}


def load_yaml(config_path):
    try:
        import yaml as _yaml
        with open(config_path, "r") as f:
            return _yaml.safe_load(f)
    except ImportError:
        return _parse_simple_yaml(config_path)


def _parse_simple_yaml(path):
    data = {}
    current_section = None
    with open(path, "r") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped[0].isspace():
                current_section = stripped.rstrip(":").strip()
                if current_section not in data:
                    data[current_section] = {}
            else:
                if current_section is None:
                    continue
                kv = stripped.strip()
                if ":" not in kv:
                    continue
                key, _, val = kv.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val == "":
                    val = ""
                else:
                    try:
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
                data[current_section][key] = val
    return data


def find_config():
    script_dir = Path(__file__).resolve().parent
    loc = script_dir / "defaults.yaml"
    if loc.exists():
        return str(loc)
    cwd = Path.cwd()
    for loc in [cwd / "config" / "defaults.yaml", cwd / "defaults.yaml"]:
        if loc.exists():
            return str(loc)
    print("[ERROR] Cannot find config/defaults.yaml", file=sys.stderr)
    sys.exit(1)


def show_help(mode, defaults):
    print("\n" + mode.upper() + " configurable parameters:\n")
    print("  {:<42} {:<20} {}".format("ENV_VAR", "Default", "Description"))
    print("  {:<42} {:<20} {}".format("-"*40, "-"*18, "-"*30))
    for key, val in sorted(defaults.items()):
        desc = DESCRIPTIONS.get(key, "")
        val_str = str(val)
        if len(val_str) > 18:
            val_str = val_str[:15] + "..."
        print("  {:<42} {:<20} {}".format(key, val_str, desc))

    print("""
Override methods (priority: CLI > env var > YAML):
  1. export ENV_VAR=value
  2. --key value  (kebab-case, auto-converted to UPPER_CASE)
  3. Set in slurm script: export KEY=value before calling run_*.sh

Examples:
  bash local_ops/run_rl.sh --train-batch-size 32 --max-prompt-length 2048
  export ROLLOUT_GPU_MEMORY_UTIL=0.60 && bash local_ops/run_eval.sh
""")


def key_to_env(key):
    return key.upper().replace("-", "_")


def parse_cli_overrides(args):
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and arg not in ("--help", "-h"):
            raw = arg[2:]
            if "=" in raw:
                key, val = raw.split("=", 1)
            else:
                key = raw
                i += 1
                if i < len(args) and not args[i].startswith("--"):
                    val = args[i]
                else:
                    val = "true"
                    i -= 1
            overrides[key_to_env(key)] = val
        i += 1
    return overrides


def format_value(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    return '"{}"'.format(val)


def main():
    args = list(sys.argv[1:])
    if not args:
        print("Usage: load_config.py <rl|eval|coldstart> [--key value ...] [--help]", file=sys.stderr)
        sys.exit(1)

    mode = args.pop(0).lower()
    if mode not in ("rl", "eval", "coldstart"):
        print("[ERROR] Unknown mode: {}. Options: rl, eval, coldstart".format(mode), file=sys.stderr)
        sys.exit(1)

    if "--help" in args or "-h" in args:
        config = load_yaml(find_config())
        defaults = config.get(mode, {})
        show_help(mode, defaults)
        sys.exit(0)

    cli_overrides = parse_cli_overrides(args)
    config = load_yaml(find_config())
    yaml_defaults = config.get(mode, {})

    for key, val in config.get("paths", {}).items():
        yaml_defaults["DRKERNEL_" + key.upper()] = val

    model_map = {"base_repo": "DRKERNEL_BASE_MODEL_REPO", "eval_repo": "DRKERNEL_EVAL_MODEL_REPO"}
    for yk, ek in model_map.items():
        if yk in config.get("model", {}):
            yaml_defaults[ek] = config["model"][yk]

    exports = []
    for key, val in sorted(yaml_defaults.items()):
        if key in cli_overrides:
            val = cli_overrides[key]
        elif key in os.environ:
            continue
        exports.append("export {}={}".format(key, format_value(val)))

    for line in exports:
        print(line)


if __name__ == "__main__":
    main()
