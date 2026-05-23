#!/usr/bin/env python3
"""
RL Training Metrics Filter

Reads the verbose verl training output from stdin, extracts key RL metrics from
the per-step summary lines, and prints a clean formatted table.

Pass-through: all original output lines are forwarded to stdout unchanged.
The clean summary is printed as a separate formatted line after each step.
"""

import sys
import re


# Key metrics to extract from verl's step output
# Format: "step:N - key1:val1 - key2:val2 - ..."
METRICS_OF_INTEREST = [
    ("step", "step"),
    ("score/mean", "critic/score/mean"),
    ("score/max", "critic/score/max"),
    ("reward/mean", "critic/rewards/mean"),
    ("reward/max", "critic/rewards/max"),
    ("correctness%", "critic/rewards_extra/correctness/mean"),
    ("success%", "critic/rewards_extra/success/mean"),
    ("compile%", "critic/rewards_extra/compilation/mean"),
    ("speedup%", "critic/rewards_extra/is_speedup_positive/mean"),
    ("resp_len", "response_length/mean"),
    ("clip%", "response_length/clip_ratio"),
    ("solve_0%", "batch/solve_none_ratio"),
    ("solve_100%", "batch/solve_all_ratio"),
    ("actor_loss", "actor/loss"),
    ("kl_div", "actor/kl_divergence"),
    ("grad_norm", "actor/grad_norm"),
    ("time/step_s", "timing_s/step"),
    ("throughput", "perf/throughput"),
    ("gpu_mem_gb", "perf/max_memory_allocated_gb"),
]

# Formatting widths
HEADER_FMT = (
    "{step:>6s} │ {score:>8s} │ {reward:>8s} │ {corr:>6s} │ {succ:>6s} │ "
    "{comp:>6s} │ {spdup:>6s} │ {rlen:>6s} │ {clip:>5s} │ "
    "{s0:>5s} │ {s100:>5s} │ {loss:>8s} │ {kl:>8s} │ "
    "{time:>8s} │ {tput:>6s} │ {mem:>7s}"
)

ROW_FMT = (
    "{step:>6s} │ {score:>8s} │ {reward:>8s} │ {corr:>6s} │ {succ:>6s} │ "
    "{comp:>6s} │ {spdup:>6s} │ {rlen:>6s} │ {clip:>5s} │ "
    "{s0:>5s} │ {s100:>5s} │ {loss:>8s} │ {kl:>8s} │ "
    "{time:>8s} │ {tput:>6s} │ {mem:>7s}"
)


def extract_metrics(line: str) -> dict:
    """Parse verl's step metrics line into a dict."""
    if not line.strip().startswith("step:"):
        return {}
    
    # Split by " - " to get individual metric pairs
    parts = line.strip().split(" - ")
    metrics = {}
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        metrics[key.strip()] = val.strip()
    return metrics


def format_val(val_str: str, as_pct: bool = False) -> str:
    """Format a metric value for display."""
    try:
        v = float(val_str)
    except (ValueError, TypeError):
        return val_str[:6]
    
    if as_pct:
        return f"{v * 100:5.1f}%"
    
    if abs(v) < 0.001 and v != 0:
        return f"{v:.2e}"[:6]
    elif abs(v) < 10:
        return f"{v:.4f}"[:8]
    elif abs(v) < 1000:
        return f"{v:.2f}"[:8]
    else:
        return f"{v:.1f}"[:8]


def main():
    header_printed = False
    step_count = 0
    
    for line in sys.stdin:
        # Pass through all output
        sys.stdout.write(line)
        sys.stdout.flush()
        
        metrics = extract_metrics(line)
        if not metrics:
            continue
        
        step_count += 1
        
        # Print header every 20 steps
        if not header_printed or step_count % 20 == 1:
            print(file=sys.stderr)
            print("=" * 130, file=sys.stderr)
            print(HEADER_FMT.format(
                step=" Step",
                score=" Score  ",
                reward=" Reward ",
                corr=" Corr%",
                succ=" Succ%",
                comp=" Comp%",
                spdup=" Spdup%",
                rlen="RspLen",
                clip="Clip%",
                s0="Sol0%",
                s100="S100%",
                loss="Loss",
                kl="KL",
                time="Time/s",
                tput="Tput",
                mem="GPU_Mem",
            ), file=sys.stderr)
            print("=" * 130, file=sys.stderr)
            header_printed = True
        
        # Resolve metrics
        step = metrics.get("step", "?")
        
        def get_metric(verl_key: str, as_pct: bool = False) -> str:
            val = metrics.get(verl_key, "-")
            if val == "-":
                return "-"
            return format_val(val, as_pct)
        
        row = ROW_FMT.format(
            step=f"{step:>6s}",
            score=get_metric("critic/score/mean"),
            reward=get_metric("critic/rewards/mean"),
            corr=get_metric("critic/rewards_extra/correctness/mean", as_pct=True),
            succ=get_metric("critic/rewards_extra/success/mean", as_pct=True),
            comp=get_metric("critic/rewards_extra/compilation/mean", as_pct=True),
            spdup=get_metric("critic/rewards_extra/is_speedup_positive/mean", as_pct=True),
            rlen=get_metric("response_length/mean"),
            clip=get_metric("response_length/clip_ratio", as_pct=True),
            s0=get_metric("batch/solve_none_ratio", as_pct=True),
            s100=get_metric("batch/solve_all_ratio", as_pct=True),
            loss=get_metric("actor/loss"),
            kl=get_metric("actor/kl_divergence"),
            time=get_metric("timing_s/step"),
            tput=get_metric("perf/throughput"),
            mem=get_metric("perf/max_memory_allocated_gb"),
        )
        
        print(row, file=sys.stderr)
        sys.stderr.flush()


if __name__ == "__main__":
    main()
