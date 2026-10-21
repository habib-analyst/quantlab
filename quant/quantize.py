"""Calibration-driven quantization driver for GPTQ and AWQ.

Usage: python quantize.py --method gptq|awq

Loads Qwen2.5-0.5B-Instruct in fp32, walks the decoder blocks with the
BlockCalibrator, quantizes each block's seven linears with the chosen
method, advances the calibration states through the quantized block,
and finally saves a packed artifact (codes + scales per linear) under
artifacts/ plus a per-layer JSON log. The evaluator loads the artifact
into a fresh model; nothing about the eval text is used here.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from awq import awq_quantize  # noqa: E402
from calibration import BlockCalibrator, calib_chunks  # noqa: E402
from gptq import gptq_quantize  # noqa: E402
from quant_linear import QuantLinear, pack_int4  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/hatch/workspace/goals/ai-inference-engineering-buildout/projects/servecore/models/Qwen2.5-0.5B-Instruct"
BLOCK_LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj"]
GROUP = 128


def get_submodule(root, rel_name):
    mod = root
    for p in rel_name.split("."):
        mod = getattr(mod, p)
    return mod


def set_submodule(root, rel_name, new):
    parts = rel_name.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gptq", "awq"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    chunks = calib_chunks(tok)
    print(f"calibration chunks: {len(chunks)} x {chunks[0].shape[0]} tokens",
          flush=True)
    calib = BlockCalibrator(model, chunks)
    n_blocks = len(model.model.layers)

    artifact = {}
    layer_log = []
    t0 = time.perf_counter()
    for b in range(n_blocks):
        block = model.model.layers[b]
        caps = calib.capture(b, BLOCK_LINEARS)
        for rel in BLOCK_LINEARS:
            linear = get_submodule(block, rel)
            full_name = f"model.layers.{b}.{rel}"
            cap = caps[rel]
            if args.method == "gptq":
                codes, scales, info, _ = gptq_quantize(
                    linear.weight.detach(), cap["H"], group_size=GROUP)
                in_scales = None
            else:
                codes, scales, in_scales, info, _ = awq_quantize(
                    linear.weight.detach(), cap["Xsub"], cap["mean_abs"],
                    group_size=GROUP)
            q = QuantLinear.from_codes(linear, codes, scales, GROUP,
                                       in_scales=in_scales)
            set_submodule(block, rel, q)
            artifact[full_name] = {
                "codes": pack_int4(codes), "scales": scales,
                "in_scales": in_scales, "group_size": GROUP,
                "in_features": linear.in_features,
                "out_features": linear.out_features}
            layer_log.append({"layer": full_name, **info})
            print(f"  block {b} {rel}: {info}", flush=True)
        calib.advance(b)
        print(f"block {b} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    out_pt = os.path.join(BASE, "artifacts", f"{args.method}.pt")
    torch.save(artifact, out_pt)
    with open(os.path.join(BASE, "artifacts", f"{args.method}_layers.json"),
              "w") as f:
        json.dump(layer_log, f, indent=2)
    print(f"saved {out_pt} ({os.path.getsize(out_pt) / 1e6:.1f} MB) "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    print("QUANTIZE DONE", flush=True)


if __name__ == "__main__":
    main()
