"""Async streaming load test against a running vLLM server (/v1/completions).

Usage:
    vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16   # in another shell
    python vllm_benchmark.py --config-name default
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from prompts import PROMPTS

CONCURRENCY_LEVELS = [1, 8, 32, 64]
MAX_TOKENS = 128

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", required=True)
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--url", default="http://localhost:8000/v1/completions")
args = parser.parse_args()

# Apply the chat template ourselves (the completions endpoint doesn't),
# so prompts match what hf_benchmark.py feeds the model.
tokenizer = AutoTokenizer.from_pretrained(args.model)
TEXTS = [
    tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    for p in PROMPTS
]


async def send_request(session, prompt):
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "ignore_eos": True,  # always generate exactly MAX_TOKENS
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    tokens = 0
    async with session.post(args.url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if ttft is None and chunk["choices"] and chunk["choices"][0]["text"]:
                ttft = time.perf_counter() - start
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
    return ttft, time.perf_counter() - start, tokens


async def run_level(session, concurrency, num_requests):
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(i):
        async with semaphore:
            return await send_request(session, TEXTS[i % len(TEXTS)])

    start = time.perf_counter()
    results = await asyncio.gather(*(limited(i) for i in range(num_requests)))
    wall = time.perf_counter() - start

    ttfts = np.array([r[0] for r in results]) * 1000
    latencies = np.array([r[1] for r in results])
    total_tokens = sum(r[2] for r in results)
    return {
        "config": args.config_name,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "output_tokens_per_s": round(total_tokens / wall, 1),
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1),
        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 1),
        "latency_p50_s": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
    }


async def main():
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await run_level(session, 8, 8)  # warmup, not recorded
        rows = []
        for concurrency in CONCURRENCY_LEVELS:
            row = await run_level(session, concurrency, max(64, 4 * concurrency))
            rows.append(row)
            print(row, flush=True)

    path = "results/vllm_results.csv"
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"Appended {len(rows)} rows to {path}")


asyncio.run(main())
