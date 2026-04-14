import httpx
import asyncio
import json

async def evaluate_kernelbench():
    reference_code = '''
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='cuda')]
'''

    kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    input_ptrs = input_ptr + row_start + col_offsets

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='cuda')]
'''

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:10907/evaluate",
            json={
                "task_id": "softmax-kernel-001",
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": "Model",
                "backend": "triton",
                "num_correct_trials": 5,
                "num_perf_trials": 100,
            }
        )
        return response.json()

result = asyncio.run(evaluate_kernelbench())
print("Full Json")
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
print(f"Compiled: {result['compiled']}")
print(f"Correctness: {result['correctness']}")
print(f"Speedup: {result['speedup']:.2f}x")
print(f"Reference Runtime: {result['reference_runtime']:.4f} ms")
print(f"Kernel Runtime: {result['kernel_runtime']:.4f} ms")