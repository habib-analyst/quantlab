"""QuantLab driver: runs every variant as its own subprocess (memory
isolation), quantizes GPTQ/AWQ first if their artifacts are missing,
then assembles results/summary.json from the per-variant JSONs.

Safe to re-run: finished variants are skipped by the evaluator, and
existing artifacts are reused.
"""

import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(BASE, "logs")
MODEL_15B = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/cachelab/models/Qwen2.5-1.5B-Instruct"


def run(cmd, log_name):
    log_path = os.path.join(LOGS, log_name)
    print(f"== {' '.join(cmd)}  (log: {log_name})", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "a") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           cwd=BASE)
    dt = time.perf_counter() - t0
    print(f"   exit {r.returncode} in {dt:.0f}s", flush=True)
    return r.returncode


def evaluate(variant, tag=None, model_path=None, base_dtype=None):
    cmd = [sys.executable, os.path.join(BASE, "eval", "evaluate.py"),
           "--variant", variant]
    if tag:
        cmd += ["--tag", tag]
    if model_path:
        cmd += ["--model-path", model_path]
    if base_dtype:
        cmd += ["--base-dtype", base_dtype]
    return run(cmd, f"eval_{tag or variant}.log")


def main():
    os.makedirs(LOGS, exist_ok=True)
    py = sys.executable

    evaluate("fp32")            # harness validation + reference numbers
    evaluate("int8_rtn")
    evaluate("int4_rtn_pc")
    evaluate("int4_rtn_g128")
    evaluate("bf16")

    for method in ("gptq", "awq"):
        art = os.path.join(BASE, "artifacts", f"{method}.pt")
        if not os.path.exists(art):
            run([py, os.path.join(BASE, "quant", "quantize.py"),
                 "--method", method], f"quantize_{method}.log")
        evaluate(f"int4_{method}_g128")

    evaluate("fp8_w8a8")        # slow on this CPU build; runs last of the 0.5B set

    # 1.5B spot check: RTN group-128 against its own bf16 baseline
    evaluate("bf16", tag="15b_bf16", model_path=MODEL_15B, base_dtype="bf16")
    evaluate("int4_rtn_g128", tag="15b_int4_rtn_g128", model_path=MODEL_15B,
             base_dtype="bf16")

    build_summary()
    print("RUN_ALL DONE", flush=True)


def build_summary():
    res_dir = os.path.join(BASE, "results")
    entries = {}
    for fn in sorted(os.listdir(res_dir)):
        if fn.endswith(".json") and fn != "summary.json":
            with open(os.path.join(res_dir, fn)) as f:
                d = json.load(f)
            entries[d["variant"]] = d

    def slim(d, ref_bytes):
        return {
            "variant": d["variant"],
            "method": d["method"],
            "model": d["model"],
            "stored_mb": d["stored_mb"],
            "size_ratio_vs_ref": round(d["stored_bytes"] / ref_bytes, 4)
            if ref_bytes else None,
            "ppl": d["perplexity"]["ppl"],
            "decode_tok_per_s": d["decode"]["mean_tok_per_s"],
            "spotcheck": f"{d['spotcheck']['score']}/{d['spotcheck']['total']}",
        }

    main = {}
    spot = {}
    ref05 = entries.get("fp32", {}).get("stored_bytes")
    ref15 = entries.get("15b_bf16", {}).get("stored_bytes")
    for tag, d in entries.items():
        if tag.startswith("15b_"):
            spot[tag] = slim(d, ref15)
        else:
            main[tag] = slim(d, ref05)
    hw = next(iter(entries.values()))["hardware"] if entries else {}
    summary = {"hardware": hw, "model_main": "Qwen2.5-0.5B-Instruct",
               "variants": main, "spotcheck_1.5b": spot}
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
