"""Benchmark Hugging Face Transformers generation across dtypes and batch sizes."""
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
BATCH_SIZES = [1, 8, 32]
NEW_TOKENS = 128
WARMUP_RUNS = 1
TIMED_RUNS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")


def make_batch(batch_size):
    # Cycle through the 20 prompts so every batch size gets full batches.
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to("cuda")


def generate(model, inputs):
    # min_new_tokens == max_new_tokens forces exactly NEW_TOKENS per sequence,
    # so every config does the same amount of work (no early stopping on EOS).
    return model.generate(
        **inputs,
        max_new_tokens=NEW_TOKENS,
        min_new_tokens=NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )


rows = []
for dtype_name, dtype in DTYPES.items():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to("cuda").eval()

    for batch_size in BATCH_SIZES:
        inputs = make_batch(batch_size)

        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                generate(model, inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            latencies = []
            for _ in range(TIMED_RUNS):
                start = time.perf_counter()
                out = generate(model, inputs)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        output_tokens = (out.shape[1] - inputs["input_ids"].shape[1]) * batch_size
        latency = sum(latencies) / len(latencies)
        row = {
            "dtype": dtype_name,
            "batch_size": batch_size,
            "latency_s": round(latency, 3),
            "output_tokens_per_s": round(output_tokens / latency, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        rows.append(row)
        print(row, flush=True)

    del model
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/hf_results.csv", index=False)
print()
print(df.to_string(index=False))

baseline = df[(df.dtype == "float32") & (df.batch_size == 1)].iloc[0]
best = df.loc[df.output_tokens_per_s.idxmax()]
speedup = best.output_tokens_per_s / baseline.output_tokens_per_s
print(
    f"\nBest config: {best.dtype}, batch {best.batch_size} -> {best.output_tokens_per_s} tok/s, "
    f"{speedup:.1f}x the throughput of float32 batch 1 ({baseline.output_tokens_per_s} tok/s)"
)
