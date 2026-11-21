"""Profile Qwen2.5-7B decode steps (batch 8) in float32 vs float16, and sample GPU
utilization during an HF batch-32 generate. Writes results/profiler_top_ops.txt."""
import threading
import time

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROFILE_BATCH, WARMUP_STEPS, PROFILE_STEPS = 8, 3, 10
UTIL_BATCH, UTIL_TOKENS, UTIL_INTERVAL_S = 32, 128, 0.05

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)


def make_batch(batch_size):
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], tokenize=False, add_generation_prompt=True)
        for i in range(batch_size)
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to("cuda")


@torch.inference_mode()
def profile_decode(model):
    """Prefill once, then profile PROFILE_STEPS single-token decode steps."""
    inputs = make_batch(PROFILE_BATCH)
    mask = inputs["attention_mask"]
    out = model(**inputs, use_cache=True)

    def step(out, mask):
        next_token = out.logits[:, -1:].argmax(-1)
        mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return model(input_ids=next_token, attention_mask=mask, past_key_values=out.past_key_values, use_cache=True), mask

    for _ in range(WARMUP_STEPS):
        out, mask = step(out, mask)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_STEPS):
            out, mask = step(out, mask)
        torch.cuda.synchronize()
    return prof.key_averages().table(sort_by="self_device_time_total", row_limit=10, max_name_column_width=50)


@torch.inference_mode()
def sample_utilization(model):
    """Run one batch-32 generate while a thread samples NVML utilization."""
    inputs = make_batch(UTIL_BATCH)
    kwargs = dict(max_new_tokens=UTIL_TOKENS, min_new_tokens=UTIL_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    model.generate(**inputs, **kwargs)  # warmup
    samples, done = [], threading.Event()

    def sampler():
        while not done.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(gpu)
            samples.append((u.gpu, u.memory))
            time.sleep(UTIL_INTERVAL_S)

    thread = threading.Thread(target=sampler)
    thread.start()
    model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    done.set()
    thread.join()
    gpu_util = sum(s[0] for s in samples) / len(samples)
    mem_util = sum(s[1] for s in samples) / len(samples)
    return f"{len(samples)} samples: mean GPU util {gpu_util:.0f}%, mean memory-controller util {mem_util:.0f}%"


report = []
for name, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="cuda").eval()
    table = profile_decode(model)
    util = sample_utilization(model)
    report.append(
        f"=== {MODEL} {name}: top 10 ops by self CUDA time, "
        f"{PROFILE_STEPS} decode steps at batch {PROFILE_BATCH} ===\n{table}\n"
        f"HF generate, batch {UTIL_BATCH}, {UTIL_TOKENS} tokens, NVML sampled every {UTIL_INTERVAL_S * 1000:.0f} ms: {util}\n"
    )
    print(report[-1], flush=True)
    del model
    torch.cuda.empty_cache()

with open("results/profiler_top_ops.txt", "w") as f:
    f.write("\n".join(report))
print("Saved results/profiler_top_ops.txt")
