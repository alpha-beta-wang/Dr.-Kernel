# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel 奖励管理器，专门用于 kernel code RL 训练
复用 laser 的架构，集成 KernelServer 进行性能评估
"""

from collections import defaultdict
import torch
import logging

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


# @register("kernel")
class AsyncKernelRewardManager:
    """Kernel 奖励管理器，集成 KernelServer 进行内核性能评估"""

    def __init__(
        self,
        tokenizer,
        num_examine=5,
        compute_score=None,
        reward_fn_key="data_source",
        reward_config=None,
        **kwargs
    ) -> None:
        """
        初始化 KernelRewardManager
        
        Args:
            tokenizer: 分词器
            num_examine: 打印到控制台的样本数量
            compute_score: 自定义评分函数
            reward_fn_key: 用于识别数据源的键
            reward_config: Hydra/OmegaConf 下的 reward_model 配置（唯一客户端配置载体）
            **kwargs: 其他参数
        """

        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model

        self.reward_config = reward_config
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_valid = kwargs.get("is_valid", False)
        self.server_url = self.reward_config.server_url
        self.reward_policy = self.reward_config.reward_policy
        self.task_timeout = self.reward_config.task_timeout
        self.print_status = getattr(self.reward_config, "print_status", False)
        
        # 验证 server_url 不为空
        if not self.server_url:
            raise ValueError("server_url is required for KernelRewardManager")

        self.reward_weights = self.reward_config.reward_weights
        
        self.logger = logging.getLogger(__name__)

        # 打印配置信息（全部来源于 reward_config）
        self.logger.info(f"KernelRewardManager initialized with server: {self.server_url}")
        self.logger.info(f"Reward weights: {self.reward_weights}")
        try:
            enhanced = self.reward_config.enhanced
            use_sandbox_rate_limit = self.reward_config.use_sandbox_rate_limit
            rate_limit = self.reward_config.rate_limit
            timeout = self.reward_config.timeout
            max_concurrent = self.reward_config.max_concurrent
            print(f"[RewardManager] cfg enhanced={enhanced} use_sandbox_rate_limit={use_sandbox_rate_limit} rate_limit={rate_limit} timeout={timeout} max_concurrent={max_concurrent}")
        except Exception:
            pass

    def execute_env(self, response_str: str, ground_truth: str, entry_point: str, uuid: str, response_ids: list[int]):
        """
        Execute the environment and return the result
        We split it since we hope to re-evaluate when the speedup value is anomaly large.
        """
        
        try:
            # 准备批量计算的参数
            solution_strs = [response_str]
            ground_truths = [ground_truth]
            entry_points = [entry_point]
            uuids = [uuid]
            
            # 调用评分函数
            if hasattr(self.compute_score, '__call__'):
                # 检查是否支持批量处理（更稳健地识别 partial 包裹的真实函数）
                is_batch = False
                func_name = ''
                # 直接标记优先
                if getattr(self.compute_score, "_is_batch", False):
                    is_batch = True
                # 尝试从 partial 的 raw_fn 中获取标记或名称
                underlying_func = None
                if hasattr(self.compute_score, 'func'):
                    # functools.partial(func, *args, **kwargs) 中的 func
                    underlying_func = self.compute_score.func
                    if getattr(underlying_func, "_is_batch", False):
                        is_batch = True
                # 对于 _call_with_kwargs 这类包装，raw_fn 通常在 partial.args[0]
                if hasattr(self.compute_score, 'args') and self.compute_score.args:
                    possible_raw_fn = self.compute_score.args[0]
                    if callable(possible_raw_fn):
                        underlying_func = possible_raw_fn
                        if getattr(underlying_func, "_is_batch", False):
                            is_batch = True
                # 名称兜底判断
                if hasattr(self.compute_score, '__name__'):
                    func_name = self.compute_score.__name__
                elif underlying_func is not None and hasattr(underlying_func, '__name__'):
                    func_name = underlying_func.__name__
                if 'batch' in func_name.lower():
                    is_batch = True

                # 仅传递必要控制参数：reward_config 与 is_valid
                safe_kwargs = {"reward_config": self.reward_config, "is_valid": self.is_valid}

                if is_batch:
                    results = self.compute_score(
                        solution_strs, ground_truths, entry_points,
                        uuids=uuids,
                        **safe_kwargs
                    )
                else:
                    # 单个处理
                    results = []
                    for i, (solution_str, ground_truth, entry_point) in enumerate(zip(solution_strs, ground_truths, entry_points)):
                        uuid_val = uuids[i] if i < len(uuids) else None
                        single_kwargs = {**safe_kwargs, "entry_point": entry_point, "uuid": uuid_val}
                        result = self.compute_score(
                            solution_str=solution_str,
                            ground_truth=ground_truth,
                            **single_kwargs
                        )
                        results.append(result)
            else:
                # 使用默认评分函数
                results = []
                for i, (solution_str, ground_truth, entry_point) in enumerate(zip(solution_strs, ground_truths, entry_points)):
                    uuid_val = uuids[i] if i < len(uuids) else None
                    result = default_compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        entry_point=entry_point,
                        uuid=uuid_val,
                        is_valid=self.is_valid,
                    )
                    results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error in reward computation: {e}")
            results = [
                {
                    "score": self.reward_config.reward_policy.penalties.penalty_score,
                    "reward": self.reward_config.reward_policy.penalties.penalty_score,
                    "correctness": False,
                    "success": False,
                    "compiled": False,
                    "error": str(e),
                    "num_custom_kernel": 0,
                    "num_total_kernels": 0,
                    "custom_kernel_cuda_time_in_profiling_us": 0,
                    "total_kernel_run_time_in_profiling_us": 0,
                }
                for _ in range(len(response_ids))
            ]
        
        if len(results) != 1:
            raise ValueError(f"The length of results should be 1, but got {len(results)}")
        
        return results





    def __call__(self, *args, **kwargs):
        import json as _json

        # === Detect calling convention ===
        # DataProto batch (sync eval): __call__(data: DataProto, return_dict=True)
        # Per-sample async engine: __call__(response_ids, content, ground_truth, entry_point, uuid, ...)
        if len(args) >= 1 and hasattr(args[0], "batch") and hasattr(args[0], "non_tensor_batch"):
            data = args[0]
            return_dict = kwargs.pop("return_dict", True) if "return_dict" in kwargs else (args[1] if len(args) >= 2 else True)
        elif len(args) >= 5:
            # Per-sample call from async engine
            response_ids = args[0]
            response_str = args[1]
            ground_truth = args[2]
            entry_point = args[3]
            uuid = args[4]
            return_full_state = kwargs.pop("return_full_state", False) or (args[5] if len(args) >= 6 else False)
            return_dict = kwargs.pop("return_dict", True)
            max_response_length = kwargs.get("response_length")
            vr = len(response_ids) if isinstance(response_ids, (list, tuple)) else int(response_ids.shape[0]) if hasattr(response_ids, "shape") else len(response_ids)
            if max_response_length is not None:
                vr = min(vr, int(max_response_length))
            vr = max(vr, 1)
            rt = torch.zeros(vr, dtype=torch.float32)
            rei = {}

            _gt = ground_truth if isinstance(ground_truth, str) else str(ground_truth) if ground_truth else ""
            _ep = entry_point if isinstance(entry_point, str) else str(entry_point) if entry_point else ""
            _uid = uuid if isinstance(uuid, str) else str(uuid) if uuid else ""
            _rids = list(response_ids) if hasattr(response_ids, "tolist") else response_ids

            results = self.execute_env(str(response_str), _gt, _ep, _uid, _rids)
            sp = results[0].get("speedup", 0.0) or 0.0
            if sp > self.reward_config.speedup_reward_upper_bound:
                results = self.execute_env(str(response_str), _gt, _ep, _uid, _rids)
            result = results[0]
            sp = result.get("speedup", 0.0) or 0.0
            score = result.get("score", result.get("reward", 0.0))
            rt[vr - 1] = score
            rei["correctness"] = result.get("correctness", False)
            rei["performance"] = sp
            rei["is_speedup_positive"] = (sp >= 1.0 + self.reward_config.speedup_eps)
            rei["is_decoy_kernel"] = result.get("decoy_kernel", False)
            rei["compilation"] = result.get("compiled", False)
            rei["success"] = result.get("success", False)
            rei["status"] = result.get("status", "unknown")
            rei["error"] = result.get("error")
            nc = result.get("num_custom_kernel", 0); nt = result.get("num_total_kernels", 0)
            rei["num_custom_kernel"] = nc; rei["num_total_kernels"] = nt
            rei["num_coverage"] = float(f"{(nc / nt if nt > 0 else 0):.2f}")
            ck = result.get("custom_kernel_cuda_time_in_profiling_us", 0); tk = result.get("total_kernel_run_time_in_profiling_us", 0)
            rei["custom_kernel_cuda_time_in_profiling_us"] = ck; rei["total_kernel_run_time_in_profiling_us"] = tk
            rei["time_coverage"] = float(f"{(ck / tk if tk > 0 else 0):.2f}")
            rei["correctness_tensor"] = torch.tensor([float(result.get("correctness", False))])
            rei["performance_tensor"] = torch.tensor([float(sp)])
            rei["compilation_tensor"] = torch.tensor([float(result.get("compiled", False))])
            if self.print_status:
                self.logger.info(
                    "[KernelEvalStatus] idx=0"
                    " status=" + str(rei.get("status", "unknown")) +
                    " compiled=" + str(rei.get("compilation", False)) +
                    " correct=" + str(rei.get("correctness", False)) +
                    " speedup=" + str(sp) +
                    " uuid=" + str(uuid) +
                    " entry=" + str(entry_point) +
                    " error=" + str(rei.get("error"))
                )
            if return_dict:
                ret = {"reward_tensor": rt, "reward_extra_info": rei}
                if return_full_state:
                    ret["env_state"] = result
                return ret
            return (rt, rei, result) if return_full_state else (rt, rei)
        else:
            # Fallback: keyword-arg per-sample call (existing behavior)
            data = args[0] if len(args) >= 1 else None
            return_dict = args[1] if len(args) >= 2 else True

        if hasattr(data, "batch") and hasattr(data, "non_tensor_batch"):
            if "rm_scores" in data.batch.keys():
                return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]
            response_ids = data.batch["responses"]
            sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            ground_truths = [((_json.loads(d.non_tensor_batch.get("reward_model", {})) if isinstance(d.non_tensor_batch.get("reward_model", {}), str) else d.non_tensor_batch.get("reward_model", {})).get("ground_truth", "")) for d in data]
            extra_infos = [((_json.loads(d.non_tensor_batch.get("extra_info", {})) if isinstance(d.non_tensor_batch.get("extra_info", {}), str) else d.non_tensor_batch.get("extra_info", {})) or {}) for d in data]
            entry_points = [ei.get("entry_point", "Model") for ei in extra_infos]
            uuids = [d.non_tensor_batch.get("uid", str(i)) for i, d in enumerate(data)]
            prompt_length = data.batch["prompts"].shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = {"correctness": [], "performance": [], "is_speedup_positive": [], "is_decoy_kernel": [], "compilation": [], "success": [], "status": [], "error": [], "num_custom_kernel": [], "num_total_kernels": [], "num_coverage": [], "custom_kernel_cuda_time_in_profiling_us": [], "total_kernel_run_time_in_profiling_us": [], "time_coverage": [], "correctness_tensor": [], "performance_tensor": [], "compilation_tensor": []}
            for i in range(len(data)):
                rids = response_ids[i].tolist() if hasattr(response_ids[i], "tolist") else response_ids[i]
                results = self.execute_env(sequences_strs[i], ground_truths[i], entry_points[i], str(uuids[i]) if uuids[i] else str(i), rids)
                result = results[0]
                score = result.get("score", result.get("reward", 0.0))
                speedup = result.get("speedup", 0.0) or 0.0
                if speedup > self.reward_config.speedup_reward_upper_bound:
                    print(f"[DEBUG] speedup is anomaly large, re-execute the environment")
                    results = self.execute_env(sequences_strs[i], ground_truths[i], entry_points[i], str(uuids[i]) if uuids[i] else str(i), rids)
                    result = results[0]
                    speedup = result.get("speedup", 0.0) or 0.0
                    score = result.get("score", result.get("reward", 0.0))
                ti = int(valid_response_length[i].item()) - 1
                if ti >= 0:
                    reward_tensor[i, ti] = score
                reward_extra_info["correctness"].append(result.get("correctness", False))
                reward_extra_info["performance"].append(speedup)
                reward_extra_info["is_speedup_positive"].append(speedup >= 1.0 + self.reward_config.speedup_eps)
                reward_extra_info["is_decoy_kernel"].append(result.get("decoy_kernel", False))
                reward_extra_info["compilation"].append(result.get("compiled", False))
                reward_extra_info["success"].append(result.get("success", False))
                reward_extra_info["status"].append(result.get("status", "unknown"))
                reward_extra_info["error"].append(result.get("error"))
                nc = result.get("num_custom_kernel", 0); nt = result.get("num_total_kernels", 0)
                reward_extra_info["num_custom_kernel"].append(nc)
                reward_extra_info["num_total_kernels"].append(nt)
                reward_extra_info["num_coverage"].append(float(f"{(nc / nt if nt > 0 else 0):.2f}"))
                ck = result.get("custom_kernel_cuda_time_in_profiling_us", 0); tk = result.get("total_kernel_run_time_in_profiling_us", 0)
                reward_extra_info["custom_kernel_cuda_time_in_profiling_us"].append(ck)
                reward_extra_info["total_kernel_run_time_in_profiling_us"].append(tk)
                reward_extra_info["time_coverage"].append(float(f"{(ck / tk if tk > 0 else 0):.2f}"))
                reward_extra_info["correctness_tensor"].append(float(result.get("correctness", False)))
                reward_extra_info["performance_tensor"].append(float(speedup))
                reward_extra_info["compilation_tensor"].append(float(result.get("compiled", False)))
                if self.print_status:
                    self.logger.info(
                        "[KernelEvalStatus] idx=" + str(i) +
                        " status=" + str(result.get("status", "unknown")) +
                        " compiled=" + str(result.get("compiled", False)) +
                        " correct=" + str(result.get("correctness", False)) +
                        " speedup=" + str(speedup) +
                        " uuid=" + str(uuids[i]) +
                        " entry=" + str(entry_points[i]) +
                        " error=" + str(result.get("error"))
                    )
            return {"reward_tensor": reward_tensor, "extra_info": reward_extra_info} if return_dict else reward_tensor
        return_full_state = kwargs.get("return_full_state", False)
        response_str = kwargs.get("response_str", "")
        ground_truth = kwargs.get("ground_truth", "")
        entry_point = kwargs.get("entry_point", "")
        uuid = kwargs.get("uuid", "")
        response_ids = kwargs.get("response_ids", data if isinstance(data, list) else [data])
        max_response_length = kwargs.get("response_length")
        vr = len(response_ids)
        if max_response_length is not None:
            vr = min(vr, int(max_response_length))
        vr = max(vr, 1)
        rt = torch.zeros(vr, dtype=torch.float32)
        rei = {}
        print(f"[DEBUG] entry point in reward manager: {entry_point}")
        results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)
        sp = results[0].get("speedup", 0.0) or 0.0
        if sp > self.reward_config.speedup_reward_upper_bound:
            print(f"[DEBUG] speedup is anomaly large, re-execute the environment")
            results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)
        result = results[0]
        sp = result.get("speedup", 0.0) or 0.0
        score = result.get("score", result.get("reward", 0.0))
        rt[vr - 1] = score
        rei["correctness"] = result.get("correctness", False)
        rei["performance"] = sp
        rei["is_speedup_positive"] = (sp >= 1.0 + self.reward_config.speedup_eps)
        rei["is_decoy_kernel"] = result.get("decoy_kernel", False)
        rei["compilation"] = result.get("compiled", False)
        rei["success"] = result.get("success", False)
        rei["status"] = result.get("status", "unknown")
        rei["error"] = result.get("error")
        nc = result.get("num_custom_kernel", 0); nt = result.get("num_total_kernels", 0)
        rei["num_custom_kernel"] = nc; rei["num_total_kernels"] = nt
        rei["num_coverage"] = float(f"{(nc / nt if nt > 0 else 0):.2f}")
        ck = result.get("custom_kernel_cuda_time_in_profiling_us", 0); tk = result.get("total_kernel_run_time_in_profiling_us", 0)
        rei["custom_kernel_cuda_time_in_profiling_us"] = ck; rei["total_kernel_run_time_in_profiling_us"] = tk
        rei["time_coverage"] = float(f"{(ck / tk if tk > 0 else 0):.2f}")
        print(f"[DEBUG] num_custom_kernel in reward manager: {nc}")
        print(f"[DEBUG] num_total_kernels in reward manager: {nt}")
        print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us in reward manager: {ck}")
        print(f"[DEBUG] total_kernel_run_time_in_profiling_us in reward manager: {tk}")
        rei["correctness_tensor"] = torch.tensor([float(result.get("correctness", False))])
        rei["performance_tensor"] = torch.tensor([float(sp)])
        rei["compilation_tensor"] = torch.tensor([float(result.get("compiled", False))])
        if self.print_status:
            self.logger.info(
                "[KernelEvalStatus] idx=0"
                " status=" + str(rei.get("status", "unknown")) +
                " compiled=" + str(rei.get("compilation", False)) +
                " correct=" + str(rei.get("correctness", False)) +
                " speedup=" + str(sp) +
                " uuid=" + str(uuid) +
                " entry=" + str(entry_point) +
                " error=" + str(rei.get("error"))
            )
        if return_dict:
            ret = {"reward_tensor": rt, "reward_extra_info": rei}
            if return_full_state:
                ret["env_state"] = result
            return ret
        return (rt, rei, result) if return_full_state else (rt, rei)
