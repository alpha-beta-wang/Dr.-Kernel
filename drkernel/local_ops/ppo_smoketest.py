#!/usr/bin/env python3
"""
PPO OOM Smoke Test — matches verl's exact device mesh pattern.
FSDP: 1D mesh (8,) dim "fsdp" — same as verl default fsdp_size=-1
SP: 2D mesh (4, 2) dims ("dp", "sp") — Ulysses only
"""

import os, time, functools, warnings
import torch, torch.distributed as dist, torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.device_mesh import init_device_mesh
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

PPO_MICRO_TOKEN = int(os.environ.get("PPO_MICRO_TOKEN", 9344))
SP_SIZE = int(os.environ.get("SP_SIZE", 2))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
ROLLOUT_N = int(os.environ.get("ROLLOUT_N", 8))
PPO_MINI_BATCH_SIZE = int(os.environ.get("PPO_MINI_BATCH_SIZE", 16))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", 1152))
MAX_RESPONSE_LEN = int(os.environ.get("MAX_RESPONSE_LEN", 8192))
USE_GRAD_CKPT = int(os.environ.get("USE_GRAD_CKPT", 1))
USE_REMOVE_PADDING = int(os.environ.get("USE_REMOVE_PADDING", 0))
SEQ_LEN = MAX_PROMPT_LEN + MAX_RESPONSE_LEN

MODEL_PATH = os.environ.get(
    "RL_MODEL_PATH",
    "/share/personal/S/huanglongsheng/drkernel_data/checkpoints/coldstart/global_step_300-sft-761600/global_step_600",
)

def log(msg):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def memory_report(tag=""):
    a = torch.cuda.memory_allocated(LOCAL_RANK) / 1e9
    r = torch.cuda.memory_reserved(LOCAL_RANK) / 1e9
    p = torch.cuda.max_memory_allocated(LOCAL_RANK) / 1e9
    if RANK == 0:
        print(f"  MEM [{tag}: alloc={a:.2f}G reserv={r:.2f}G peak={p:.2f}G", flush=True)
    return a, r, p

# === verl utilities ===
def get_device_id():
    return torch.cuda.current_device()

def get_torch_device():
    return torch.cuda

def init_fn(x: nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x

def get_init_weight_context_manager(use_meta_tensor=True, mesh=None):
    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            ctx = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            ctx = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        ctx = cpu_init_weights
    return ctx

@torch.no_grad()
def offload_fsdp_model_to_cpu(model):
    _lazy_init(model, model)
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        handle.flat_param_to(torch.device("cpu"), non_blocking=False)
        handle.flat_param._local_shard = handle.flat_param.data
    get_torch_device().empty_cache()

@torch.no_grad()
def load_fsdp_model_to_gpu(model):
    _lazy_init(model, model)
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=False)
        handle.flat_param._local_shard = handle.flat_param.data

@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            for k, v in optimizer.state[p].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[p][k] = v.to(device_id, non_blocking=False)

def main():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl")

    # === CRITICAL: Two separate meshes like verl ===
    assert WORLD_SIZE == 8
    # 1. FSDP mesh: 1D (matching verl's fsdp_size=-1 default)
    fsdp_mesh = init_device_mesh("cuda", mesh_shape=(WORLD_SIZE,), mesh_dim_names=["fsdp"])

    # 2. SP mesh: 2D (Ulysses only, NOT used for FSDP)
    assert WORLD_SIZE % SP_SIZE == 0
    dp_size = WORLD_SIZE // SP_SIZE
    sp_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, SP_SIZE), mesh_dim_names=["dp", "sp"])

    if not int(os.environ.get("SKIP_MONKEY_PATCH", 0)):
        from verl.utils.ulysses import set_ulysses_sequence_parallel_group
        set_ulysses_sequence_parallel_group(sp_mesh["sp"].get_group())

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if RANK == 0:
        print(f"=== PPO Smoke Test (verl mesh pattern) ===", flush=True)
        print(f"  GPU={torch.cuda.get_device_name(0)} total={total_mem:.1f}G", flush=True)
        print(f"  FSDP mesh: {fsdp_mesh}", flush=True)
        print(f"  SP mesh: {sp_mesh}", flush=True)
        print(f"  WORLD={WORLD_SIZE} SP={SP_SIZE} dp={dp_size}", flush=True)
        print(f"  PPO_MICRO={PPO_MICRO_TOKEN} SEQ={SEQ_LEN}", flush=True)
        print(f"  TRAIN_BS={TRAIN_BATCH_SIZE} ROLLOUT_N={ROLLOUT_N} MINI_BS={PPO_MINI_BATCH_SIZE}", flush=True)

    log("Creating model...")
    model_config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, attn_implementation="flash_attention_2"
    )

    # Use sp_mesh for init context (matching verl's pattern)
    init_context = get_init_weight_context_manager(
        use_meta_tensor=not model_config.tie_word_embeddings, mesh=sp_mesh
    )
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, config=model_config, dtype=torch.bfloat16, trust_remote_code=True
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"Model: {n_params:.1f}B params")
    memory_report("after model create")

    if not int(os.environ.get("SKIP_MONKEY_PATCH", 0)):
        log("Applying Ulysses SP monkey patch...")
        from verl.models.transformers.monkey_patch import apply_monkey_patch
        apply_monkey_patch(model=model, ulysses_sp_size=SP_SIZE, use_remove_padding=bool(USE_REMOVE_PADDING))
        memory_report("after monkey patch")

    if USE_GRAD_CKPT:
        log("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        memory_report("after grad ckpt")

    # === FSDP wrap: use FSDP mesh (1D), NOT SP mesh ===
    log("FSDP wrap (on fsdp_mesh)...")
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=int(5e7))
    mp_config = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    sharding_strategy = ShardingStrategy.FULL_SHARD

    model = FSDP(
        model,
        cpu_offload=None,
        param_init_fn=init_fn,
        auto_wrap_policy=auto_wrap_policy,
        device_id=get_device_id(),
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_config,
        sync_module_states=True,
        device_mesh=fsdp_mesh,  # <-- FSDP mesh, NOT sp_mesh!
        use_orig_params=False,
        forward_prefetch=False,
    )
    memory_report("after FSDP")
    dist.barrier()

    # --- Offload/Load cycle ---
    log("Offload -> CPU...")
    offload_fsdp_model_to_cpu(model)
    memory_report("after offload model")

    log("Creating AdamW...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
    memory_report("after optimizer")

    log("Load -> GPU...")
    load_fsdp_model_to_gpu(model)
    memory_report("after load model")
    load_fsdp_optimizer(optimizer, get_device_id())
    memory_report("after load optimizer")
    dist.barrier()

    # === PPO update ===
    torch.cuda.reset_peak_memory_stats(LOCAL_RANK)
    total_seq = TRAIN_BATCH_SIZE * ROLLOUT_N
    micro_batch_size = max(1, PPO_MICRO_TOKEN // SEQ_LEN)
    grad_accum = max(1, PPO_MINI_BATCH_SIZE // micro_batch_size)

    dp_rank = sp_mesh["dp"].get_local_rank()
    seq_per_dp = total_seq // sp_mesh["dp"].size()

    rng = torch.Generator()
    rng.manual_seed(dp_rank)
    input_ids = torch.randint(0, 151936, (seq_per_dp, SEQ_LEN), dtype=torch.long, generator=rng)

    log(f"PPO update: micro_bs={micro_batch_size} grad_accum={grad_accum} seq_per_dp={seq_per_dp}")
    model.train()

    fwd_peak, bwd_peak = 0.0, 0.0
    num_micro = seq_per_dp // micro_batch_size

    for mb_start in range(0, seq_per_dp, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, seq_per_dp)
        mb_idx = mb_start // micro_batch_size
        mb_input_ids = input_ids[mb_start:mb_end].cuda()
        attn_mask = torch.ones_like(mb_input_ids, dtype=torch.bool).cuda()

        outputs = model(input_ids=mb_input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = mb_input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
        loss = loss / grad_accum
        loss.backward()

        fwd_peak = max(fwd_peak, torch.cuda.max_memory_allocated(LOCAL_RANK) / 1e9)
        bwd_peak = max(bwd_peak, torch.cuda.max_memory_allocated(LOCAL_RANK) / 1e9)

        if RANK == 0 and (mb_idx < 3 or mb_idx >= num_micro - 1 or mb_idx % 10 == 0):
            cur = torch.cuda.memory_allocated(LOCAL_RANK) / 1e9
            print(f"  micro {mb_idx+1}/{num_micro}: fwd_peak={fwd_peak:.1f}G bwd_peak={bwd_peak:.1f}G cur={cur:.1f}G", flush=True)

        del mb_input_ids, attn_mask, outputs, logits, shift_logits, shift_labels, loss

    log("Optimizer step...")
    torch.cuda.synchronize()
    optimizer.step()
    optimizer.zero_grad()
    memory_report("after optimizer step")

    peak = torch.cuda.max_memory_allocated(LOCAL_RANK) / 1e9
    peak_r = torch.cuda.max_memory_reserved(LOCAL_RANK) / 1e9
    all_p = [torch.zeros(1).cuda() for _ in range(WORLD_SIZE)]
    dist.all_gather(all_p, torch.tensor([peak], device=f"cuda:{LOCAL_RANK}"))
    max_p = max(p.item() for p in all_p)
    min_p = min(p.item() for p in all_p)

    if RANK == 0:
        margin = total_mem - max_p
        mpct = margin / total_mem * 100
        status = "PASS" if mpct > 15 else ("MARGINAL" if mpct > 8 else "OOM RISK")
        print(f"\n{'='*60}")
        print(f"  PPO SMOKE TEST RESULTS", flush=True)
        print(f"{'='*60}")
        print(f"  GPU total:      {total_mem:.2f} GiB", flush=True)
        print(f"  Peak allocated: {max_p:.2f} GiB (min={min_p:.2f})", flush=True)
        print(f"  Peak reserved:  {peak_r:.2f} GiB", flush=True)
        print(f"  Margin:         {margin:.2f} GiB ({mpct:.1f}%)", flush=True)
        print(f"  FWD peak:       {fwd_peak:.2f} GiB", flush=True)
        print(f"  BWD peak:       {bwd_peak:.2f} GiB", flush=True)
        print(f"  Config: MICRO={PPO_MICRO_TOKEN} SP={SP_SIZE} SEQ={SEQ_LEN}", flush=True)
        print(f"          ROLLOUT_N={ROLLOUT_N} MINI_BS={PPO_MINI_BATCH_SIZE}", flush=True)
        print(f"          grad_ckpt={USE_GRAD_CKPT} remove_pad={USE_REMOVE_PADDING}", flush=True)
        print(f"  Status:         {status}", flush=True)
        print(f"{'='*60}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
