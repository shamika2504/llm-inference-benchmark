"""Benchmark Hugging Face Transformers generation across dtypes and batch sizes.

Usage: python hf_benchmark.py [--model Qwen/Qwen2.5-7B-Instruct]
"""
import argparse
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
BATCH_SIZES = [1, 8, 32]
NEW_TOKENS = 128
WARMUP_RUNS = 1
TIMED_RUNS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
args = parser.parse_args()
short_name = args.model.split("/")[-1].lower()

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")


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


def benchmark(model, batch_size):
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
    return {
        "latency_s": round(latency, 3),
        "ms_per_token": round(latency / NEW_TOKENS * 1000, 2),
        "output_tokens_per_s": round(output_tokens / latency, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


rows = []
for dtype_name, dtype in DTYPES.items():
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda").eval()
    except torch.OutOfMemoryError:
        print(f"OOM loading {dtype_name}", flush=True)

    for batch_size in BATCH_SIZES:
        row = {"model": args.model, "dtype": dtype_name, "batch_size": batch_size, "status": "ok"}
        if model is None:
            row["status"] = "OOM"
        else:
            try:
                row.update(benchmark(model, batch_size))
            except torch.OutOfMemoryError:
                row["status"] = "OOM"
                torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)

    del model
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
csv_path = f"results/hf_results_{short_name}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}\n")
print(df.drop(columns="model").to_string(index=False))

# Separate speedups, so batching and precision effects aren't mixed into one number.
ok = df[df.status == "ok"].set_index(["dtype", "batch_size"])


def get(dtype, batch_size, col):
    return ok[col].get((dtype, batch_size))


print(f"\n== {args.model} ==")
print("Precision speedup (tok/s vs float32 at the same batch size):")
for bs in BATCH_SIZES:
    base = get("float32", bs, "output_tokens_per_s")
    for dt in ["float16", "bfloat16"]:
        val = get(dt, bs, "output_tokens_per_s")
        text = f"{val / base:.2f}x" if base and val else "n/a (OOM)"
        print(f"  batch {bs:>2}  {dt:<8}  {text}")

print("Batching speedup (tok/s, batch 32 vs batch 1 at the same dtype):")
for dt in DTYPES:
    b1, b32 = get(dt, 1, "output_tokens_per_s"), get(dt, 32, "output_tokens_per_s")
    text = f"{b32 / b1:.1f}x" if b1 and b32 else "n/a (OOM)"
    print(f"  {dt:<8}  {text}")

print("Memory saving (float16 peak memory vs float32 at the same batch size):")
for bs in BATCH_SIZES:
    m32, m16 = get("float32", bs, "peak_mem_gb"), get("float16", bs, "peak_mem_gb")
    text = f"{m32:.2f} GB -> {m16:.2f} GB ({1 - m16 / m32:.0%} less)" if m32 and m16 else "n/a (OOM)"
    print(f"  batch {bs:>2}  {text}")
