"""Charts for QuantLab, drawn only from the measured result JSONs."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "charts")

ORDER = ["fp32", "bf16", "fp8_w8a8", "int8_rtn", "int4_rtn_pc",
         "int4_rtn_g128", "int4_awq_g128", "int4_gptq_g128"]
LABELS = {"fp32": "fp32", "bf16": "bf16", "fp8_w8a8": "fp8\nW8A8",
          "int8_rtn": "int8\nRTN", "int4_rtn_pc": "int4 RTN\nper-channel",
          "int4_rtn_g128": "int4 RTN\ngroup-128", "int4_awq_g128": "int4\nAWQ",
          "int4_gptq_g128": "int4\nGPTQ"}
COLORS = {"fp32": "#444444", "bf16": "#7f7f7f", "fp8_w8a8": "#9467bd",
          "int8_rtn": "#1f77b4", "int4_rtn_pc": "#ff7f0e",
          "int4_rtn_g128": "#d62728", "int4_awq_g128": "#2ca02c",
          "int4_gptq_g128": "#17becf"}


def load():
    data = {}
    for fn in os.listdir(RES):
        if fn.endswith(".json") and fn != "summary.json":
            d = json.load(open(os.path.join(RES, fn)))
            data[d["variant"]] = d
    return data


def scatter(data, ykey, ylabel, title, fname, ylog=False):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for v in ORDER:
        if v not in data:
            continue
        d = data[v]
        y = d["perplexity"]["ppl"] if ykey == "ppl" else d["decode"]["mean_tok_per_s"]
        ax.scatter([d["stored_mb"]], [y], s=110, color=COLORS[v], zorder=3)
        ax.annotate(LABELS[v], (d["stored_mb"], y),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("stored model size (MB)")
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def bars(data):
    vs = [v for v in ORDER if v in data]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    metrics = [("stored_mb", "stored size (MB)"),
               ("ppl", "perplexity (lower is better)"),
               ("tok", "decode tok/s (higher is better)")]
    for ax, (key, label) in zip(axes, metrics):
        vals = []
        for v in vs:
            d = data[v]
            vals.append(d["stored_mb"] if key == "stored_mb" else
                        d["perplexity"]["ppl"] if key == "ppl" else
                        d["decode"]["mean_tok_per_s"])
        ax.bar([LABELS[v] for v in vs], vals,
               color=[COLORS[v] for v in vs])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=7)
        if key == "ppl":
            lo = min(vals) * 0.92
            ax.set_ylim(lo, max(vals) * 1.05)
    fig.suptitle("QuantLab tradeoffs, Qwen2.5-0.5B-Instruct on CPU")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff_bars.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    scatter(data, "ppl", "perplexity (Alice, 3072 tokens)",
            "Quality vs size: perplexity against stored model size",
            "ppl_vs_size.png", ylog=True)
    scatter(data, "tok", "decode tokens per second (greedy, 64 tokens)",
            "Speed vs size: decode throughput against stored model size",
            "speed_vs_size.png")
    bars(data)
    print("charts done")
