"""QuantLab evaluator: one variant, one JSON, identical measurements.

Every variant is measured the same way on the same hardware:

* Streaming perplexity over the first 3072 tokens of Alice's Adventures
  in Wonderland from "CHAPTER I." (the CacheLab evaluation slice and
  scoring: 256-token chunks through a growing KV cache, every token
  scored from the logits that precede it).
* Decode speed: greedy generation of exactly 64 tokens (min_new_tokens
  forces full length so every variant produces the same amount of work)
  on the four CacheLab Part B prompts, wall-clocked per prompt.
* Spot check: six factual completions, 16 greedy tokens each, scored by
  case-insensitive substring match against acceptable answers.
* Stored size: exact bytes of the model in its stored form
  (quant_linear.model_stored_bytes).

Usage: python evaluate.py --variant int4_gptq_g128 [--tag NAME]
       [--model-path PATH] [--base-dtype fp32|bf16]
"""

import argparse
import json
import math
import os
import platform
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "quant"))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from calibration import eval_ids  # noqa: E402
from quant_linear import (QuantLinear, model_stored_bytes, replace_linear,  # noqa: E402
                          target_linears)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_05B = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/servecore/models/Qwen2.5-0.5B-Instruct"

DECODE_PROMPTS = [
    "The history of the Roman Empire began when",
    "To make a good omelette, you first need to",
    "The mitochondria is known as the powerhouse of the cell because",
    "In a shocking turn of events, scientists announced yesterday that",
]

SPOTCHECK = [
    ("The capital of France is", ["paris"]),
    ("The chemical symbol for water is", ["h2o"]),
    ("The first President of the United States was", ["washington"]),
    ("The largest planet in our solar system is", ["jupiter"]),
    ("Two plus two equals", ["4", "four"]),
    ("The author of Romeo and Juliet is", ["shakespeare"]),
]

DESCRIPTIONS = {
    "fp32": "FP32 reference: every weight in fp32, no quantization.",
    "bf16": "BF16 baseline: every weight in bf16, no quantization.",
    "int8_rtn": "Weight-only INT8: per-output-channel symmetric "
                "round-to-nearest (from scratch), dequantized to fp32 "
                "on the fly.",
    "int4_rtn_pc": "Weight-only INT4: per-output-channel symmetric "
                   "round-to-nearest (from scratch), packed 2 codes/byte.",
    "int4_rtn_g128": "Weight-only INT4: group-128 (input dim) symmetric "
                     "round-to-nearest (from scratch), packed 2 codes/byte.",
    "int4_gptq_g128": "GPTQ (from-scratch implementation of Frantar et "
                      "al.): INT4 group-128, Hessian from 2048 "
                      "calibration tokens of a disjoint Alice slice, "
                      "per-column error compensation.",
    "int4_awq_g128": "AWQ core (from-scratch implementation of Lin et "
                     "al.): activation-aware per-input-channel scaling, "
                     "alpha grid search minimizing layer output error "
                     "on calibration inputs, INT4 group-128.",
    "fp8_w8a8": "FP8 W8A8, real compute path: weights per-output-channel "
                "e4m3, activations dynamically per-token e4m3, matmul "
                "via torch._scaled_mm on CPU.",
}


def build_variant(variant, model_path, base_dtype):
    dtype = torch.bfloat16 if base_dtype == "bf16" else torch.float32
    if variant == "bf16":
        return AutoModelForCausalLM.from_pretrained(model_path,
                                                    dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if variant == "fp32":
        return model
    linears = target_linears(model)
    if variant == "int8_rtn":
        for name, lin in linears:
            replace_linear(model, name, QuantLinear.from_rtn(lin, 8, lin.in_features))
    elif variant == "int4_rtn_pc":
        for name, lin in linears:
            replace_linear(model, name, QuantLinear.from_rtn(lin, 4, lin.in_features))
    elif variant == "int4_rtn_g128":
        for name, lin in linears:
            replace_linear(model, name, QuantLinear.from_rtn(lin, 4, 128))
    elif variant == "fp8_w8a8":
        for name, lin in linears:
            replace_linear(model, name, QuantLinear.from_fp8(lin))
    elif variant in ("int4_gptq_g128", "int4_awq_g128"):
        method = "gptq" if "gptq" in variant else "awq"
        art = torch.load(os.path.join(BASE, "artifacts", f"{method}.pt"),
                         weights_only=False)
        by_name = dict(linears)
        for name, spec in art.items():
            lin = by_name[name]
            q = QuantLinear(lin.in_features, lin.out_features, lin.bias)
            q.mode, q.bits, q.group_size = "int", 4, spec["group_size"]
            q.register_buffer("codes", spec["codes"])
            q.register_buffer("scales", spec["scales"])
            if spec["in_scales"] is not None:
                q.register_buffer("in_scales", spec["in_scales"])
            else:
                q.in_scales = None
            replace_linear(model, name, q)
    else:
        raise ValueError(f"unknown variant {variant}")
    return model


@torch.no_grad()
def streaming_ppl(model, ids, chunk=256):
    total_nll, count = 0.0, 0
    past = None
    prev_logits = None
    t0 = time.perf_counter()
    for i in range(0, len(ids), chunk):
        part = torch.tensor(ids[i:i + chunk], dtype=torch.long).unsqueeze(0)
        out = model(input_ids=part, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0].float()
        if prev_logits is None:
            preds, targets = logits[:-1], part[0][1:]
        else:
            preds = torch.cat([prev_logits, logits[:-1]], dim=0)
            targets = part[0]
        lp = F.log_softmax(preds, dim=-1)
        total_nll += -float(lp.gather(-1, targets.unsqueeze(-1)).sum())
        count += len(targets)
        prev_logits = logits[-1:]
    dt = time.perf_counter() - t0
    return {"ppl": round(math.exp(total_nll / count), 3),
            "tokens_scored": count, "time_s": round(dt, 1),
            "s_per_token": round(dt / len(ids), 4)}


@torch.no_grad()
def decode_bench(model, tok, n_tokens=64):
    # warmup, untimed
    ids = tok(DECODE_PROMPTS[0], return_tensors="pt").input_ids
    model.generate(ids, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                   use_cache=True, pad_token_id=tok.eos_token_id)
    runs = []
    for p in DECODE_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=n_tokens,
                             min_new_tokens=n_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        dt = time.perf_counter() - t0
        new = out.shape[1] - ids.shape[1]
        runs.append({"prompt": p, "new_tokens": new, "time_s": round(dt, 2),
                     "tok_per_s": round(new / dt, 2)})
    mean = sum(r["tok_per_s"] for r in runs) / len(runs)
    return {"prompts": runs, "mean_tok_per_s": round(mean, 2),
            "tokens_per_prompt": n_tokens}


@torch.no_grad()
def spotcheck(model, tok):
    results = []
    for prompt, acceptable in SPOTCHECK:
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=16, min_new_tokens=16,
                             do_sample=False, use_cache=True,
                             pad_token_id=tok.eos_token_id)
        cont = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ok = any(a in cont.lower() for a in acceptable)
        results.append({"prompt": prompt, "continuation": cont.strip(),
                        "ok": ok})
    return {"score": sum(r["ok"] for r in results), "total": len(results),
            "runs": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model-path", default=MODEL_05B)
    ap.add_argument("--base-dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()
    tag = args.tag or args.variant
    torch.set_num_threads(os.cpu_count() or 2)

    out_path = os.path.join(BASE, "results", f"{tag}.json")
    if os.path.exists(out_path):
        print(f"{tag}: already exists, skipping", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    t0 = time.perf_counter()
    model = build_variant(args.variant, args.model_path, args.base_dtype)
    model.eval()
    build_s = time.perf_counter() - t0
    print(f"{tag}: built in {build_s:.0f}s", flush=True)

    stored = model_stored_bytes(model)
    ids = eval_ids(tok)
    ppl = streaming_ppl(model, ids)
    print(f"{tag}: ppl={ppl['ppl']} ({ppl['time_s']}s)", flush=True)
    dec = decode_bench(model, tok)
    print(f"{tag}: decode mean {dec['mean_tok_per_s']} tok/s", flush=True)
    spot = spotcheck(model, tok)
    print(f"{tag}: spotcheck {spot['score']}/{spot['total']}", flush=True)

    result = {
        "variant": tag,
        "method": DESCRIPTIONS[args.variant],
        "model": os.path.basename(args.model_path.rstrip("/")),
        "base_dtype": args.base_dtype,
        "stored_bytes": stored,
        "stored_mb": round(stored / 1e6, 2),
        "perplexity": ppl,
        "decode": dec,
        "spotcheck": spot,
        "hardware": {
            "platform": platform.machine(),
            "cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "build_s": round(build_s, 1),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{tag}: wrote {out_path}", flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
