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
| `SAVE_FREQ` | 100 | 每 N 步保存 checkpoint |

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
| `TRAIN_BATCH_SIZE` | 2 | 训练 batch size，**必须 >= GPU 数** |
| `PPO_MINI_BATCH_SIZE` | 2 | PPO mini batch，**必须 >= GPU 数** |
| `MAX_PROMPT_LENGTH` | 1152 | 最大 prompt 长度 |
| `MAX_RESPONSE_LENGTH` | 512 | 最大回复长度 |
| `ROLLOUT_GPU_MEMORY_UTIL` | 0.74 | vLLM 显存利用率（48GB 卡慎重） |
| `ROLLOUT_MODE` | sync | 生成模式 (sync/async_vllm) |
| `ROLLOUT_N` | 1 | 每条 prompt 生成样本数 |
| `ALGORITHM` | trloo | RL 算法 |
| `LEARNING_RATE` | 1e-6 | 学习率 |
| `ENABLE_MULTI_TURN` | false | 多轮对话 |
| `ACTOR_PARAMETER_OFFLOAD` | true | Actor 参数 CPU offload |
| `ACTOR_OPTIMIZER_OFFLOAD` | true | 优化器状态 CPU offload |
| `FREE_CACHE_ENGINE` | false | 训练前释放 vLLM 缓存 |

### GPU 需求

- **最低**: 8× L40 (48GB)，需调低 `ROLLOUT_GPU_MEMORY_UTIL`
- **推荐**: 8-16× A100 (80GB)
- **重要**: `TRAIN_BATCH_SIZE` 和 `PPO_MINI_BATCH_SIZE` 必须 >= GPU 数量，否则 FSDP device_mesh 分配会归零报错

### 资源受限场景

Slurm 脚本内置了调优注释，常见调整：

```bash
# 场景：只有 8 GPU 配额 + 6h 时间限制
sbatch --gres=gpu:8 \
       --qos=gpu-short \
       -t 6:00:00 \
       --cpus-per-task=16 \
       --export=ALL,N_GPUS_PER_NODE=8,TRAIN_BATCH_SIZE=8,PPO_MINI_BATCH_SIZE=8,ROLLOUT_GPU_MEMORY_UTIL=0.50 \
       slurm/run_drkernel_rl.slurm
```

---

## 5. 模块三：Evaluation 评估

### 用途

评估模型的内核生成质量，输出 Pass@1、Score、编译率、加速比等指标。

### 前置条件

- KernelGYM 服务运行中
- 待评估模型存在

### 输入

| 项目 | 路径 |
|------|------|
| 待评估模型 | `DRKERNEL_DATA_ROOT/models/hkust-nlp/drkernel-14b`（或其他） |
| 验证数据 | `DRKERNEL_DATA_ROOT/datasets/hkust-nlp/drkernel-validation-data/` |

### 输出

| 项目 | 路径 | 格式 |
|------|------|------|
| 评分结果 | `results/<RUN_NAME>/graded_results.parquet` | Parquet |
| 原始响应 | `results/<RUN_NAME>/raw_responses.jsonl` | JSONL |
| 评估指标 | `results/<RUN_NAME>/metrics.json` | JSON |
| 详细输出 | `results/<RUN_NAME>/eval_outputs/` | 目录 |

**metrics.json 包含：**
- `pass@1` — 通过率
- `score` — 综合评分
- `compile_rate` — 编译成功率
- `correctness` — 正确率
- `mean_speedup` — 平均加速比
- `positive_speedup_rate` — 正向加速比比例

### 启动方式

**Slurm 集群（推荐）：**
```bash
# 默认 8 GPU
sbatch slurm/run_drkernel_eval.slurm

# 自定义模型和输出
sbatch --export=ALL,MODEL_PATH=/path/to/checkpoint,RUN_NAME=my-eval \
       slurm/run_drkernel_eval.slurm
```

**本地调试：**
```bash
export KERNELGYM_SERVER_URL="http://localhost:10907"
bash local_ops/run_eval.sh
```

### 关键参数

```bash
# 查看全部参数
bash local_ops/run_eval.sh --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_SAMPLES` | 4 | 每样例生成次数 |
| `BATCH_SIZE` | 32 | 评估 batch size |
| `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE` | 8 | TP 大小 (= GPU 数) |
| `FSDP_SIZE` | 8 | FSDP 分片数 (= GPU 数) |
| `ROLLOUT_GPU_MEMORY_UTIL` | 0.40 | vLLM 显存利用率（eval 比训练保守） |
| `MAX_PROMPT_LENGTH` | 2048 | 最大 prompt 长度（需 >= thinking format ~1400 tokens） |
| `MULTI_TURN` | true | 多轮评估 |
| `MAX_USER_TURNS` | 3 | 最大用户轮次 |
| `NUM_PERF_TRIALS` | 10 | 性能测试次数 |
| `NUM_CORRECT_TRIALS` | 5 | 正确性测试次数 |
| `REWARD_WEIGHTS` | "0.3_0.4_0.3" | 速度/正确性/编译 权重 |

### GPU 需求

- **最低**: 8× L40 (48GB)
- **推荐**: 8× A100 (80GB)
- FSDP_SIZE 和 TP_SIZE 需要等于 GPU 数量
- 评估比训练省显存（无优化器状态），GPU_MEMORY_UTIL=0.40 已够用

---

## 6. 配置系统

### 三层优先级

```
CLI --key value  >  环境变量 export  >  YAML 默认值 (config/defaults.yaml)
```

### 方式一：命令行覆盖（最高优先级）

```bash
bash local_ops/run_rl.sh --train-batch-size 32 --learning-rate 5e-7
bash local_ops/run_eval.sh --n-samples 8 --batch-size 64
bash local_ops/run_coldstart.sh --total-epochs 8
```

### 方式二：环境变量

```bash
export TRAIN_BATCH_SIZE=32
bash local_ops/run_rl.sh  # 使用 env var 值，跳过 YAML
```

### 方式三：修改 YAML 默认值

编辑 `config/defaults.yaml` 中对应模式的参数。

### 方式四：Slurm --export

```bash
sbatch --export=ALL,TRAIN_BATCH_SIZE=8,N_GPUS_PER_NODE=8 slurm/run_drkernel_rl.slurm
```

### 查看所有可配参数

```bash
python3 config/load_config.py rl --help        # RL 参数
python3 config/load_config.py eval --help      # Eval 参数
python3 config/load_config.py coldstart --help # Coldstart 参数
```

---

## 7. 常见问题

### Q1: 显存不足 (OOM)

**症状**: `torch.OutOfMemoryError`

**解决**:
- 降低 `ROLLOUT_GPU_MEMORY_UTIL`（RL: 0.74→0.50, Eval: 0.40→0.30）
- 增加 GPU 数量（TP size 也随之调整）
- 排除 A30 节点 (`--constraint="A100|L40|L40S"`)
- 确保 `ACTOR_PARAMETER_OFFLOAD=true` 和 `ACTOR_OPTIMIZER_OFFLOAD=true`
- Coldstart: 确认 `CPU_OFFLOAD=true`, `OFFLOAD_PARAMS=true`

### Q2: ppo_mini_batch_size = 0

**症状**: batch_size 被 FSDP 分配到 0

**原因**: `TRAIN_BATCH_SIZE // N_GPUS_PER_NODE < 1` → 向下取整为 0

**解决**: 确保 `TRAIN_BATCH_SIZE >= N_GPUS_PER_NODE`，且 `PPO_MINI_BATCH_SIZE >= N_GPUS_PER_NODE`

### Q3: Eval 100 样本全部被过滤

**症状**: DataProto.concat 空列表，无样本进入评估

**原因**: `MAX_PROMPT_LENGTH` 太小（如 1024），thinking format prompt 实际有 1048-1368 tokens

**解决**: 设置 `MAX_PROMPT_LENGTH=2048`

### Q4: BASH_SOURCE[0] 路径错误

**症状**: `mkdir: cannot create directory '/var/spool/slurmd/.../.cache'`

**原因**: Slurm 拷贝脚本到临时目录执行，`BASH_SOURCE[0]` 不再指向原路径

**解决**: 已修复 — slurm 脚本使用 `${SLURM_SUBMIT_DIR}` 定位项目目录

### Q5: SLURM_GPUS_ON_NODE 为空

**原因**: 集群的 `slurm-tools/v1.0` 模块会清除该变量

**解决**: 脚本已使用 fallback 链 `${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-N}}`，可通过 `--export=N_GPUS_PER_NODE=8` 手动指定

### Q6: r8l40s-a05 ECC 错误

**症状**: `CUDA error: uncorrectable ECC error encountered`

**原因**: 该节点 GPU 硬件故障

**解决**: `--exclude=r8l40s-a05`（Coldstart 已默认排除）

### Q7: 查看某次运行的指标

```bash
cat results/<RUN_NAME>/metrics.json | python3 -m json.tool
```

### Q8: 使用 RL checkpoint 做评估

```bash
export MODEL_PATH=/path/to/checkpoints/rl/drkernel-14b-rl-4gpu/global_step_100
bash local_ops/run_eval.sh
```
