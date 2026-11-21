"""Answer quality on 100 PubMedQA questions (yes / no / maybe), per backend.

Usage:
    python quality_check.py --backend hf-fp32
    python quality_check.py --backend hf-fp16
    python quality_check.py --backend vllm      # needs a running vLLM server (default config)

Each run saves results/quality_preds_<backend>.csv, then rebuilds
results/quality.csv (accuracy + agreement with hf-fp32) from all prediction files present.
"""
import argparse
import json
import os
import re
import urllib.request

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
NUM_QUESTIONS = 100
MAX_NEW_TOKENS = 8
HF_BATCH_SIZE = 16
BACKENDS = ["hf-fp32", "hf-fp16", "vllm"]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=BACKENDS, required=True)
parser.add_argument("--url", default="http://localhost:8000/v1/completions")
args = parser.parse_args()

data = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").select(range(NUM_QUESTIONS))
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")


def build_prompt(row):
    context = "\n".join(row["context"]["contexts"])
    message = (
        f"Context:\n{context}\n\nQuestion: {row['question']}\n\n"
        "Answer with exactly one word: yes, no, or maybe."
    )
    return tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True)


def parse_answer(text):
    # First standalone yes/no/maybe, ignoring case and punctuation ("Yes." -> yes).
    match = re.search(r"\b(yes|no|maybe)\b", text.lower())
    return match.group(1) if match else "invalid"


def generate_hf(prompts, dtype):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="cuda").eval()
    outputs = []
    for i in range(0, len(prompts), HF_BATCH_SIZE):
        inputs = tokenizer(prompts[i:i + HF_BATCH_SIZE], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs


def generate_vllm(prompts):
    # The completions endpoint accepts a list of prompts in one request.
    payload = {"model": MODEL, "prompt": prompts, "max_tokens": MAX_NEW_TOKENS, "temperature": 0}
    request = urllib.request.Request(args.url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as resp:
        choices = json.load(resp)["choices"]
    return [c["text"] for c in sorted(choices, key=lambda c: c["index"])]


prompts = [build_prompt(row) for row in data]
if args.backend == "vllm":
    raw = generate_vllm(prompts)
else:
    import torch
    raw = generate_hf(prompts, torch.float32 if args.backend == "hf-fp32" else torch.float16)

preds = pd.DataFrame({
    "pubid": data["pubid"],
    "label": data["final_decision"],
    "raw_output": raw,
    "answer": [parse_answer(t) for t in raw],
})
os.makedirs("results", exist_ok=True)
preds.to_csv(f"results/quality_preds_{args.backend}.csv", index=False)
print(f"{args.backend}: accuracy {(preds.answer == preds.label).mean():.0%}, "
      f"invalid {(preds.answer == 'invalid').sum()}, answers {preds.answer.value_counts().to_dict()}")

# Rebuild the summary from every backend that has predictions so far.
reference_path = "results/quality_preds_hf-fp32.csv"
reference = pd.read_csv(reference_path) if os.path.exists(reference_path) else None
rows = []
for backend in BACKENDS:
    path = f"results/quality_preds_{backend}.csv"
    if not os.path.exists(path):
        continue
    p = pd.read_csv(path)
    rows.append({
        "backend": backend,
        "num_questions": len(p),
        "accuracy": round((p.answer == p.label).mean(), 3),
        "agreement_with_fp32": round((p.answer == reference.answer).mean(), 3) if reference is not None else None,
        "invalid_answers": int((p.answer == "invalid").sum()),
    })
summary = pd.DataFrame(rows)
summary.to_csv("results/quality.csv", index=False)
print(summary.to_string(index=False))
