import httpx
import asyncio
import json

async def evaluate_kernel_simple():
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:10907/workflow/submit",
            json={
                "workflow": "kernel_simple",
                "task_id": "my-kernel-task-001",
                "payload": {
                    "task_id": "my-kernel-task-001",
                    "kernel_code": '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    return [x, y]

def get_cases():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    expected = x + y
    return [{"inputs": [x, y], "outputs": expected}]
''',
                    "entry_point": "ModelNew",
                    "backend": "triton",
                    "device": "cuda:0",
                    "run_correctness": True,
                    "run_performance": False,
                    "num_perf_trials": 100,
                }
            }
        )
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "json": response.json(),
        }

result = asyncio.run(evaluate_kernel_simple())
payload = result["json"]
task_result = payload.get("result", {})

print(f"HTTP Status: {result['status_code']}")
print("Full JSON:")
print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
print(f"Compiled: {task_result.get('compiled')}")
print(f"Correctness: {task_result.get('correctness')}")

kernel_runtime = task_result.get("kernel_runtime")
if isinstance(kernel_runtime, (int, float)):
    print(f"Runtime: {kernel_runtime:.4f} ms")
else:
    print(f"Runtime: {kernel_runtime}")

print(f"Error Message: {task_result.get('error_message')}")
print("Metadata:")
print(json.dumps(task_result.get("metadata", {}), indent=2, ensure_ascii=False, default=str))
