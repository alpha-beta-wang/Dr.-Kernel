# Dr.Kernel 使用指南

> 本文档面向首次使用的新人，覆盖三个模块的完整使用流程。配置细节见 `config/defaults.yaml`，命令行帮助见 `--help`。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据目录结构](#2-数据目录结构)
3. [模块一：Coldstart (SFT 冷启动)](#3-模块一coldstart-sft-冷启动)
4. [模块二：RL 训练](#4-模块二rl-训练)
5. [模块三：Evaluation 评估](#5-模块三evaluation-评估)
6. [配置系统](#6-配置系统)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 依赖安装

```bash
cd drkernel
bash setup.sh
```

### 1.2 启动 KernelGYM 服务

RL 和 Eval 模块需要 KernelGYM 服务器处理内核评估请求：

```bash
cd ..  # 到 KernelGYM 根目录
./start_all_with_monitor.sh
```

RL/Eval 脚本通过 `local_ops/start_kernelgym_services.sh` 自动启动服务，无需手动操作。但如果服务地址不同，设置环境变量：

```bash
export KERNELGYM_SERVER_URL="http://<server-ip>:10907"
```

### 1.3 最小运行清单

每次运行前确认：
- `KERNELGYM_SERVER_URL` 指向运行中的 KernelGYM 服务（RL/Eval 需要）
- 模型路径有效
- 数据集路径有效
- GPU 数量匹配（见各模块说明）

---

## 2. 数据目录结构

```
DRKERNEL_DATA_ROOT/                  # 默认 /nfs_global/I/$USER/WangYongsheng/drkernel
├── models/                          # 模型文件
│   ├── Qwen/Qwen3-14B-Base/        # 基座模型 (Coldstart 输入)
│   └── hkust-nlp/drkernel-14b/     # SFT 后模型 (RL/Eval 输入)
├── datasets/                        # 数据集 (HuggingFace 格式)
│   └── hkust-nlp/
│       ├── drkernel-coldstart-8k/   # Coldstart 训练数据
│       ├── drkernel-rl-data/        # RL 训练数据
│       └── drkernel-validation-data/# RL/Eval 验证数据
├── checkpoints/                     # 训练产出
│   ├── coldstart/                   # SFT checkpoint
│   └── rl/                          # RL checkpoint
├── results/                         # 评估产出
│   └── drkernel-14b-eval-4gpu/     # 按 RUN_NAME 分目录
└── logs/                            # 运行日志
```

### 自定义路径

```bash
export DRKERNEL_DATA_ROOT="/your/custom/path"
```

---

## 3. 模块一：Coldstart (SFT 冷启动)

### 用途

在基座模型上做监督微调，让模型学会多轮内核生成对话格式。这是 RL 训练的前提步骤。

### 输入

| 项目 | 路径 |
|------|------|
| 基座模型 | `DRKERNEL_DATA_ROOT/models/Qwen/Qwen3-14B-Base` |
| 训练数据 | `DRKERNEL_DATA_ROOT/datasets/hkust-nlp/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet` |

### 输出

| 项目 | 路径 |
|------|------|
| Checkpoint | `DRKERNEL_DATA_ROOT/checkpoints/coldstart/drkernel-14b-coldstart-4gpu/` |
| 日志 | `../logs/drk-sft-<jobid>.out` |

### 启动方式

**Slurm 集群（推荐）：**
```bash
# 默认 4 GPU
sbatch slurm/run_drkernel_coldstart.slurm

# 自定义 GPU 数和数据路径
sbatch --gres=gpu:8 \
       --export=ALL,DRKERNEL_DATA_ROOT=/your/path \
       slurm/run_drkernel_coldstart.slurm
```

**本地调试：**
```bash
bash local_ops/run_coldstart.sh
```

### 关键参数

```bash
# 查看全部参数
bash local_ops/run_coldstart.sh --help

# 常用覆盖
bash local_ops/run_coldstart.sh \
    --train-batch-size 64 \
    --learning-rate 1e-5 \
    --total-epochs 8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | 32 | 总 batch size |
| `MICRO_BATCH_SIZE_PER_GPU` | 1 | 每 GPU micro batch |
| `MAX_LENGTH` | 12288 | 最大序列长度 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `TOTAL_EPOCHS` | 4 | 训练轮数 |
| `SP_SIZE` | 2 | Ulysses 序列并行度 |
| `CPU_OFFLOAD` | true | CPU offload 省显存 |
| `OFFLOAD_PARAMS` | true | 参数 offload 省显存 |
| `SAVE_FREQ` | 30 | 每 N 步保存 checkpoint |
| `RESUME_PATH` | null | 续训时指向实验目录下的 `global_step_N` 子目录 |
| `RESUME_MODE` | auto | verl resume 模式: `auto`\|`disable`\|`resume_path` |
| `MAX_ACTOR_CKPT_TO_KEEP` | null | 最多保留 N 个 actor checkpoint，旧自动删除 (null=保留全部) |
| `MAX_MODEL_LEN` | auto | vLLM max total sequence length，自动计算 prompt+response+512，可通过 --max_model_len 覆盖 |

### GPU 需求

- **最低**: 4× L40 (48GB)
- **推荐**: 4× A100 (80GB)
- Slurm 脚本默认申请 4 GPU (`--gres=gpu:4`)

---

## 4. 模块二：RL 训练

### 用途

在 SFT 模型基础上做强化学习，通过 TRLOO + MRS + PR/PRS 算法优化内核生成质量。

### 前置条件

- Coldstart 完成，或已有 `hkust-nlp/drkernel-14b` 模型
- KernelGYM 服务运行中

### 输入

| 项目 | 路径 |
|------|------|
| SFT 模型 | `DRKERNEL_DATA_ROOT/models/hkust-nlp/drkernel-14b` |
| 训练数据 | `DRKERNEL_DATA_ROOT/datasets/hkust-nlp/drkernel-rl-data/` |
| 验证数据 | `DRKERNEL_DATA_ROOT/datasets/hkust-nlp/drkernel-validation-data/` |

### 输出

| 项目 | 路径 |
|------|------|
| Checkpoint | `DRKERNEL_DATA_ROOT/checkpoints/rl/` |
| WandB 日志 | 本地离线模式 (`wandb/` 目录) |
| 运行日志 | `../logs/drk-rl-<jobid>.out` |

### 启动方式

**Slurm 集群（推荐）：**
```bash
# 默认 16 GPU（A100/L40/L40S）
sbatch slurm/run_drkernel_rl.slurm

# 8 GPU 模式（资源受限时）
sbatch --gres=gpu:8 \
       --qos=gpu-short \
       -t 6:00:00 \
       --export=ALL,N_GPUS_PER_NODE=8,TRAIN_BATCH_SIZE=8,PPO_MINI_BATCH_SIZE=8 \
       slurm/run_drkernel_rl.slurm
```

**本地调试：**
```bash
export KERNELGYM_SERVER_URL="http://localhost:10907"
bash local_ops/run_rl.sh
```

### 关键参数

```bash
# 查看全部参数
bash local_ops/run_rl.sh --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | 16 | 训练 batch size，**必须 >= GPU 数** |
| `PPO_MINI_BATCH_SIZE` | 16 | PPO mini batch，**必须 >= GPU 数** |
| `MAX_PROMPT_LENGTH` | 1152 | 最大 prompt 长度 |
| `MAX_RESPONSE_LENGTH` | 16384 | 最大回复长度，Qwen3 thinking模型需要空间做推理再生成代码 |
| `ROLLOUT_GPU_MEMORY_UTIL` | 0.35 | vLLM 显存利用率（L40S 48GB保守值，A100可提高） |
| `ROLLOUT_MAX_BATCHED_TOKENS` | 1664 | vLLM 单 batch 最大 token 数 |
| `ROLLOUT_MODE` | sync | 生成模式 (sync/async_vllm) |
| `ROLLOUT_N` | 16 | 每条 prompt 生成样本数 |
| `ALGORITHM` | trloo | RL 算法 |
| `LEARNING_RATE` | 1e-6 | 学习率 |
| `ENABLE_MULTI_TURN` | false | 多轮对话 |
| `MAX_TURN` | 1 | 最大对话轮次 |
| `VAL_MAX_TURN` | 1 | 验证时最大轮次 |
| `ACTOR_PARAMETER_OFFLOAD` | true | Actor 参数 CPU offload |
| `ACTOR_OPTIMIZER_OFFLOAD` | true | 优化器状态 CPU offload |
| `FREE_CACHE_ENGINE` | false | 训练前释放 vLLM KV cache |
| `ENFORCE_EAGER` | true | 禁用 CUDA graph（兼容性） |
| `N_VAL` | 1 | 验证样本数 |
| `VAL_BEFORE_TRAIN` | false | 训练前先验证 |
| `PROMPT_OVERSAMPLING_FACTOR` | 1.0 | prompt 过采样倍数 |
| `SAMPLE_OVERSAMPLING_FACTOR` | 1.0 | 样本过采样倍数 |
| `REWARD_MANAGER` | kernel_async | 奖励管理器类型 |
| `REWARD_FUNC_NAME` | calculate_reward_speedup | 奖励函数名 |
| `NUM_PERF_TRIALS` | 100 | 性能测试次数 |
| `IS_GET_LAST_TURN` | true | 只取最后一轮结果 |
| `RESUME_PATH` | null | 续训 checkpoint 路径 |
| `RESUME_MODE` | auto | resume 模式 |
| `MAX_ACTOR_CKPT_TO_KEEP` | null | 最多保留 checkpoint 数 |
| `PYTORCH_CUDA_ALLOC_CONF` | "" | CUDA 内存分配策略 |
---

## 5. 模块三：Evaluation 评估

### 用途

评估 RL/SFT 模型在 KernelGYM 验证集上的表现，计算 pass@k、compilation rate、correctness rate、speedup 等指标。

### 前置条件

- 已有模型（SFT 或 RL checkpoint）
- KernelGYM 服务运行中

### 输入

| 项目 | 路径 |
|------|------|
| 模型 | `DRKERNEL_DATA_ROOT/models/hkust-nlp/drkernel-14b`（可覆盖） |
| 验证数据 | `DRKERNEL_DATA_ROOT/datasets/hkust-nlp/drkernel-validation-data/` |

### 输出

| 项目 | 路径 |
|------|------|
| 评测结果 | `DRKERNEL_DATA_ROOT/results/<RUN_NAME>/` |
| 日志 | `../logs/drk-eval-<jobid>.out` |

### 启动方式

**Slurm 集群（推荐）：**
```bash
# 单轮 eval（默认，对齐论文 kernel_grading.yaml）
sbatch slurm/run_drkernel_eval.slurm

# 指定数据/模型路径
sbatch --export=ALL,DRKERNEL_DATA_ROOT=/your/path,MODEL_PATH=/your/model \
       slurm/run_drkernel_eval.slurm

# 多轮 eval（通过 CLI 覆盖 YAML 默认值）
sbatch --export=ALL,MULTI_TURN=True,MAX_USER_TURNS=3,ROLLOUT_MODE=async_vllm \
       slurm/run_drkernel_eval.slurm
```

**参数优先级：** CLI > env var > YAML。Slurm 脚本不硬编码参数，所有 eval 参数通过以下方式设定：
- `config/defaults.yaml` — 默认值（已对齐论文）
- `sbatch --export=ALL,KEY=VALUE` — 临时覆盖
- 透传 CLI 参数给 `run_eval.sh`

### 关键参数

```bash
# 查看全部参数
bash local_ops/run_eval.sh --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_SAMPLES` | 4 | 每题生成样本数 |
| `BATCH_SIZE` | 32 | 推理 batch size |
| `ROLLOUT_GPU_MEMORY_UTIL` | 0.40 | vLLM 显存利用率（L40S保守值） |
| `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE` | 8 | 张量并行度（L40S需8卡TP，A100可设1） |
| `MAX_PROMPT_LENGTH` | 20480 | 最大prompt长度。多轮对话历史逐轮累积，设太小(<8192)上下文被截断→越改越烂 |
| `MAX_RESPONSE_LENGTH` | 8192 | 最大回复长度。Qwen3 thinking模型输出分析→代码→建议，设太小(4096)导致98.5%截断 |
| `FSDP_SIZE` | 8 | FSDP 分片数 |
| `MULTI_TURN` | false | 多轮对话模式。论文模型设计为多轮迭代优化，单轮结果显著差于多轮 |
| `MAX_USER_TURNS` | 1 | 最大用户轮次（多轮时设 3+） |
| `NUM_PERF_TRIALS` | 100 | 性能测试重复次数，论文用100次保证统计稳定性 |
| `NUM_CORRECT_TRIALS` | 5 | 正确性测试重复次数 |
| `REWARD_MANAGER` | kernel_async | 奖励管理器类型 |
| `REWARD_FUNC_NAME` | calculate_reward_speedup | 奖励函数名 |
| `REWARD_WEIGHTS` | "0.3_0.4_0.3" | 奖励权重（正确性_加速比_编译） |

### 评测指标

| 指标 | 说明 |
|------|------|
| pass@k | 每题生成k条，至少1条通过的比例（编译+正确+加速>=0.99） |
| Total Score | 加权总分（0.4×正确 + 0.3×加速） |
| Compilation Rate | 生成代码能通过 nvcc 编译的比例 |
| Correctness Rate | 编译通过后输出结果正确的比例 |
| Speedup Positive | 加速比 > 1.0 的比例 |
| Avg Speedup | 所有样本加速比的均值 |
| Coverage | 覆盖的 kernel 种类/耗时占比 |

### GPU 需求

- **最低**: 4× L40S (48GB)
- **推荐**: 8× L40S（需要 TP=8 放整个模型）或 1× A100 (80GB, TP=1)

---

## 6. 配置系统

### 参数优先级

```
CLI 参数 (--key value)  >  环境变量 (export KEY=VALUE)  >  YAML 默认值 (config/defaults.yaml)
```

### 修改默认值

编辑 `config/defaults.yaml` 中对应的节（`rl`/`eval`/`coldstart`），然后提交。

### 临时覆盖

```bash
# 方式1: 环境变量覆盖
export TRAIN_BATCH_SIZE=8
bash local_ops/run_rl.sh

# 方式2: CLI 覆盖
bash local_ops/run_rl.sh --train-batch-size 8

# 方式3: sbatch 时覆盖
sbatch --export=ALL,TRAIN_BATCH_SIZE=8 slurm/run_drkernel_rl.slurm
```

### YAML 结构

```yaml
paths:
  data_root: "/nfs_global/I/${USER}/WangYongsheng/drkernel"
model:
  base_repo: "Qwen/Qwen3-14B-Base"
  eval_repo: "hkust-nlp/drkernel-14b"
rl: {...}        # RL 训练参数
eval: {...}      # 评估参数
coldstart: {...} # SFT 参数
cluster: {...}   # 集群预设
```

---

## 7. 常见问题

### Slurm 脚本不应硬编码参数

参数来源必须单一。Slurm 脚本只负责：SBATCH headers、路径设置、基础设施（TMPDIR、HF_HOME）。模型/训练/评估参数全部通过 YAML 或 CLI 设置。

### 显存不足 (CUDA OOM)

1. 降低 `ROLLOUT_GPU_MEMORY_UTIL`（0.35 → 0.30 → 0.25）
2. 增大 `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE`（4 → 8）
3. 降低 `MAX_RESPONSE_LENGTH`（但 thinking 模型至少 4096）
4. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
5. 开启 `FREE_CACHE_ENGINE=true`

### DRKERNEL_DATA_ROOT 权限错误

默认路径 `/nfs_global/I/` 可能需要特定权限。用 `--export=ALL,DRKERNEL_DATA_ROOT=/your/path` 覆盖。

### multi-turn UID broadcast 错误

`ValueError: UID broadcast shape (768,) vs (256,)` — 设置 `ENABLE_MULTI_TURN=False`。多轮 eval 不受此 bug 影响（eval 路径不同）。

### Qwen3 thinking 模型 response 截断

症状：`finish_reason=length` 比例极高(>50%)。根因：`MAX_RESPONSE_LENGTH` 太小。Qwen3 输出结构为「分析推理→代码生成→优化建议」，至少需要 8192 token。
