"""Make PNG plots from the CSVs in results/ into results/plots/."""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/plots"
L40S_BANDWIDTH_GB_S = 864  # spec-sheet memory bandwidth, used for the weight-read floor
MODELS = {"Qwen2.5-1.5B": "results/hf_results_qwen2.5-1.5b-instruct.csv",
          "Qwen2.5-7B": "results/hf_results_qwen2.5-7b-instruct.csv"}

# Fixed colors per entity (same color = same dtype / config in every chart).
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
DTYPE_COLORS = {"float32": BLUE, "float16": ORANGE, "bfloat16": AQUA}
CONFIG_COLORS = {"default": BLUE, "eager": ORANGE, "small_batch": AQUA}
CONFIG_LABELS = {"default": "default", "eager": "eager (no CUDA graphs)", "small_batch": "max-num-seqs 8"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.edgecolor": "#c9c8c3", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})

os.makedirs(OUT, exist_ok=True)
hf = {name: pd.read_csv(path) for name, path in MODELS.items()}
vllm = pd.read_csv("results/vllm_results.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}/{name}")


# 1. HF tokens/sec vs batch size, one subplot per model.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, df) in zip(axes, hf.items()):
    for dtype, color in DTYPE_COLORS.items():
        d = df[(df.dtype == dtype) & (df.status == "ok")]
        ax.plot(d.batch_size, d.output_tokens_per_s, marker="o", color=color, label=dtype)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32], ["1", "8", "32"])
    ax.set_title(name)
    ax.set_xlabel("Batch size")
axes[0].set_ylabel("Output tokens / sec")
axes[0].legend(title="dtype")
fig.suptitle("HF Transformers throughput vs batch size (128 new tokens, L40S)", fontsize=12)
save(fig, "hf_throughput_vs_batch.png")

# 2. HF ms/token at batch 1, per dtype, with the weight-read floor (weights / bandwidth).
fig, ax = plt.subplots(figsize=(7, 4.2))
width = 0.26
for i, (dtype, color) in enumerate(DTYPE_COLORS.items()):
    for j, (name, df) in enumerate(hf.items()):
        row = df[(df.dtype == dtype) & (df.batch_size == 1)].iloc[0]
        x = j + (i - 1) * width
        ax.bar(x, row.ms_per_token, width * 0.92, color=color, label=dtype if j == 0 else None)
        ax.text(x, row.ms_per_token + 0.8, f"{row.ms_per_token:.0f}", ha="center", fontsize=9, color="#0b0b0b")
        floor = row.peak_mem_gb / L40S_BANDWIDTH_GB_S * 1000
        ax.hlines(floor, x - width * 0.46, x + width * 0.46, color="#0b0b0b", linewidth=2,
                  label="weight-read floor\n(weights ÷ 864 GB/s)" if i == 0 and j == 0 else None)
ax.set_xticks([0, 1], list(hf))
ax.set_ylabel("ms per output token")
ax.set_title("Batch-1 decode time per token\n1.5B is overhead-bound, 7B is bandwidth-bound")
ax.legend(loc="upper left")
save(fig, "hf_ms_per_token_batch1.png")

# 3. vLLM tokens/sec vs concurrency, with the HF 7B fp16 batch-32 reference.
hf_ref = hf["Qwen2.5-7B"].query("dtype == 'float16' and batch_size == 32").output_tokens_per_s.iloc[0]
fig, ax = plt.subplots(figsize=(7, 4.2))
for config, color in CONFIG_COLORS.items():
    d = vllm[vllm.config == config]
    ax.plot(d.concurrency, d.output_tokens_per_s, marker="o", color=color, label=CONFIG_LABELS[config])
ax.axhline(hf_ref, color=GRAY, linestyle="--", linewidth=1.5, label=f"HF fp16, batch 32 ({hf_ref:.0f} tok/s)")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 8, 32, 64], ["1", "8", "32", "64"])
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Aggregate output tokens / sec")
ax.set_title("vLLM throughput vs concurrency (Qwen2.5-7B, fp16)")
ax.legend(loc="upper left")
save(fig, "vllm_throughput_vs_concurrency.png")

# 4. vLLM p95 TTFT vs concurrency (log scale).
fig, ax = plt.subplots(figsize=(7, 4.2))
for config, color in CONFIG_COLORS.items():
    d = vllm[vllm.config == config]
    ax.plot(d.concurrency, d.ttft_p95_ms, marker="o", color=color, label=CONFIG_LABELS[config])
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xticks([1, 8, 32, 64], ["1", "8", "32", "64"])
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("p95 time to first token (ms, log scale)")
ax.set_title("vLLM p95 TTFT vs concurrency (Qwen2.5-7B, fp16)")
ax.legend(loc="upper left")
save(fig, "vllm_p95_ttft_vs_concurrency.png")
