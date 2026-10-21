# QuantLab: quantization tradeoffs, measured on one model, one machine

I am Habib Ur Rehman, an ML engineer. This is Project 3 of my AI
inference engineering programme. Project 1 (ServeCore) was about
serving, Project 2 (CacheLab) about the KV cache and speculative
decoding. This one is about quantization: shrinking a model's weights
from 32-bit floats to 8 or 4 bits, and what that costs in quality and
speed when you actually measure it instead of quoting paper tables.

## The 30-second version

I quantized Qwen2.5-0.5B-Instruct eight ways and measured every variant
identically: same text, same prompts, same hardware. Three findings.
First, how you round matters more than how many bits you keep: naive
4-bit rounding wrecks the model (perplexity 33.1 against 11.7 for the
fp32 reference), while GPTQ's error compensation at the same 4 bits and
the same file size lands at 13.7. Second, 8-bit is nearly free in
quality (int8 round-to-nearest: 11.72; FP8 W8A8: 11.94) at less than
half the fp32 size. Third, on a CPU without fused low-bit kernels,
every quantized variant decodes far slower than the fp32 and bf16
baselines in my from-scratch implementation, because the weights are
dequantized on the fly for every token. Smaller files, slower tokens:
that is the honest tradeoff on this stack, and the prefill numbers show
where it flips.

## Hardware and honesty rules

Every number below was measured on the machine that produced it:

- CPU-only VM, 2 x86-64 vCPUs, 7.75 GiB RAM, no GPU
- PyTorch 2.13.0+cpu, Transformers 5.19.0, Python 3.12.3
- Main model: Qwen2.5-0.5B-Instruct; spot check: Qwen2.5-1.5B-Instruct
- The VM is shared. Load average ranged from about 3 to about 8 during
  the runs, so absolute decode speeds are pessimistic and vary between
  runs. Perplexity and file sizes do not depend on load and are exact.

Nothing is estimated or extrapolated. Stored sizes are counted byte by
byte from the tensors each variant actually keeps
(`quant/quant_linear.py: model_stored_bytes`). Where something did not
work or was not finished, the limits section at the end says so.

## The methods

All quantization code is mine, written for this project
(`quant/` directory). No quantization library was used.

- **fp32 / bf16**: the reference and the baseline. No quantization.
- **INT8 RTN** (`int8_rtn`): weight-only, per-output-channel symmetric
  round-to-nearest. Weights stored as int8 plus one fp32 scale per row,
  dequantized to fp32 inside every forward call.
- **INT4 RTN per-channel** (`int4_rtn_pc`): the same, at 4 bits, one
  scale per output channel, codes packed two per byte.
- **INT4 RTN group-128** (`int4_rtn_g128`): the same, with one scale
  per row per 128 input columns.
- **INT4 GPTQ** (`int4_gptq_g128`): GPTQ (Frantar et al.,
  arXiv:2210.17323) implemented from scratch. A Hessian H = sum of
  x x^T is accumulated from 2048 calibration tokens per linear layer,
  factorized once (float64 Cholesky with damping), and columns are
  quantized left to right while their rounding errors are pushed into
  the remaining columns through the inverse-Hessian Cholesky factor.
  Groups of 128 columns, scales recomputed per group from the
  compensated weights. Calibration walks block by block, so later
  blocks are calibrated on inputs that already carry earlier blocks'
  quantization error, as in the original.
- **INT4 AWQ** (`int4_awq_g128`): the core of AWQ (Lin et al.,
  arXiv:2306.00978) implemented from scratch. Per linear, the mean
  absolute activation per input channel is measured on calibration
  data, a grid search over alpha in {0, 0.25, 0.5, 0.75, 1.0} picks the
  per-channel scaling that minimizes the layer's actual output error
  on held calibration rows, and the scaled weights are quantized with
  group-128 round-to-nearest. At inference the activations are divided
  by the same scales inside the quantized linear, the paper's
  equivalent transform.
- **FP8 W8A8** (`fp8_w8a8`): real FP8 compute, not a simulation.
  Weights are stored as float8_e4m3fn with one fp32 scale per output
  channel; activations are quantized per token to e4m3 at runtime; the
  matmul runs through `torch._scaled_mm` with row-wise scaling. I
  verified the operator on this CPU build before building on it: it
  agrees with an fp32 reference matmul to about 4% max relative error
  on random matrices, which is e4m3 grid error, not a bug. What it is
  not is fast on CPU: it is a correctness fallback, and the decode
  numbers below show exactly what that costs.

Calibration and evaluation never share text. Evaluation scores tokens
[0, 3072) of Alice's Adventures in Wonderland from "CHAPTER I.";
calibration uses the disjoint slice [8192, 10240).

Embeddings are never quantized in any variant (the 0.5B model ties its
embedding and output head, 136M of its 494M parameters). This is why
the int4 totals below are 0.37x of fp32 rather than 0.125x: the
embeddings alone are 545 MB in fp32 inside every variant's total, and
only the 358M block parameters shrink.

## The measurements

Identical for every variant:

- **Perplexity**: streaming over the 3072-token Alice slice in
  256-token chunks through a growing KV cache, every token scored from
  the logits that precede it. Same harness and scoring as CacheLab;
  my fp32 run reproduces CacheLab's 11.70 exactly (11.697).
- **Decode speed**: greedy generation of exactly 64 tokens
  (min_new_tokens forces full length, so every variant does the same
  work) on the four CacheLab prompts, wall-clocked.
- **Spot check**: six factual completions ("The capital of France
  is", and five like it), 16 greedy tokens each, scored by substring
  match. Coarse on purpose; read it with the perplexity, not instead
  of it.
- **Stored size**: exact bytes of the model in stored form.

### Results: Qwen2.5-0.5B-Instruct

| Variant | Stored (MB) | vs fp32 | Perplexity | Decode (tok/s) | Spot check |
|---|---|---|---|---|---|
| fp32 reference | 1976.13 | 1.00x | 11.697 | 4.74 | 4/6 |
| bf16 baseline | 988.07 | 0.50x | 11.697 | 4.50 | 4/6 |
| INT8 RTN per-channel | 903.87 | 0.46x | 11.724 | 0.45 | 4/6 |
| FP8 W8A8 (real scaled_mm) | 903.87 | 0.46x | 11.941 | 0.24 | 6/6 |
| INT4 GPTQ group-128 | 734.92 | 0.37x | 13.707 | 0.32 | 4/6 |
| INT4 AWQ group-128 | 735.90 | 0.37x | 14.184 | 0.23 | 4/6 |
| INT4 RTN group-128 | 734.92 | 0.37x | 16.824 | 0.32 | 5/6 |
| INT4 RTN per-channel | 724.95 | 0.37x | 33.134 | 0.32 | 0/6 |

Prefill tells a different speed story than decode. The perplexity pass
(3072 tokens, processed in 256-token chunks) took: bf16 49.0 s, int8
95.4 s, fp32 102.4 s, int4 group-128 109.4 s, int4 per-channel 116.1 s,
AWQ 124.5 s, GPTQ 125.5 s, FP8 159.5 s. At chunk scale the dequantize
cost is amortized over 256 rows at once and the quantized variants sit
within about 1.2x of fp32 (FP8 at 1.6x); at decode scale, one row at a
time, the same overhead is paid per token and they fall to roughly a
tenth of the baseline speed.

Quantization itself cost, on this box: GPTQ 601 s and AWQ 249 s of
calibration and quantization for the whole model. RTN variants and FP8
quantize in seconds.

### Reading the table

1. **Rounding strategy dominates bit width.** The three group-128
   int4 variants share a file size to within a megabyte, but their
   perplexities are 13.7 (GPTQ), 14.2 (AWQ) and 16.8 (naive RTN).
   GPTQ's compensation and AWQ's scaling each recover a large part of
   what naive rounding loses. Going from group-128 to per-channel
   scales at the same 4 bits is the single worst move in the table:
   16.8 to 33.1, and the spot check collapses with it (0/6, with
   continuations that are visibly garbage). Per-channel int4 RTN is
   not a usable configuration for this model.
2. **8 bits is the quality bargain.** INT8 RTN costs 0.03 perplexity
   over fp32 and FP8 W8A8 costs 0.24 (it also quantizes activations,
   which INT8 here does not), both at 904 MB. FP8's 6/6 spot check,
   against 4/6 for fp32, is not evidence that FP8 is better: six
   prompts is a tiny sample, greedy paths wobble, and the perplexity
   over 3071 scored tokens is the trustworthy quality number. It says
   FP8 is slightly worse, and both are close to free.
3. **AWQ's search behaves the way the paper says.** Over 168 layers,
   the winning alpha was 0.25 for 127 layers, 0.5 for 30, 0.75 for 2,
   and 0 (no scaling) for only 9. The chosen scaling cut the average
   layer output error by 13.4% relative to unscaled rounding before
   the weights ever reached the model level, and the model-level
   result (14.18 vs 16.82 for RTN) confirms the direction. GPTQ still
   edges it here.
4. **On CPU, small files do not mean fast tokens.** Every dequantizing
   variant decodes between 0.23 and 0.45 tok/s against 4.5 to 4.7 for
   the baselines, because my forward rebuilds the full fp32 weight
   matrix for every single token. Production int4/int8 kernels fuse
   dequantization into the matmul precisely to avoid this; llama.cpp
   and Marlin-class kernels would rewrite the speed column. What does
   transfer: at prefill-like batch sizes the penalty mostly vanishes,
   and FP8's slow decode is a property of this CPU fallback operator,
   not of FP8, whose speed story lives on GPU tensor cores.

### Spot check: Qwen2.5-1.5B-Instruct

One variant repeated on the bigger model, against its own bf16
baseline (PPL 4.162, 3087.43 MB, 2.39 tok/s decode, spot check 6/6):

- INT4 RTN group-128 on the 1.5B: stored 1163.08 MB (0.38x of bf16),
  perplexity 7.563. Naive rounding hurts the 1.5B proportionally more
  than the 0.5B (1.82x its baseline perplexity, against 1.44x), one
  more reason the calibration-based methods matter. Its decode bench
  was started and stopped unfinished after about 1 hour 45 minutes of
  wall time: at the rate the 0.5B dequant measurements imply for a
  model with 3.7x the per-step dequantization work, the fixed
  64-token bench needed hours more on the contended VM. The perplexity
  and size above are measured and complete; the decode number is
  simply absent, and I would rather ship it absent than estimate it.

## How to choose, from these numbers only

- **You need the smallest file and have calibration data and ten
  minutes**: GPTQ int4. 735 MB at perplexity 13.7 is the best
  quality-per-byte in the table.
- **No calibration pipeline, quality first**: INT8 RTN. 904 MB,
  perplexity within noise of fp32, quantizes in seconds. On this CPU
  stack you pay for it in decode speed; on a stack with a fused int8
  kernel you would not.
- **You are headed to GPU serving**: FP8 W8A8. Quality here is
  near-free (11.94) and it is the only format in the table whose
  production kernels turn the smaller representation into speed. Just
  do not judge it by this CPU fallback's decode number.
- **Staying on CPU with this kind of naive runtime**: bf16. Same
  perplexity as fp32 at half the size and the fastest prefill
  measured. The quantized formats win on disk and memory, not on
  tokens per second, until a fused kernel enters the picture.
- **Never**: per-channel int4 round-to-nearest, at least not for a
  model of this size. The file is barely smaller than group-128 and
  the model is effectively broken.

## Run it yourself

```bash
pip install -r requirements.txt
# RTN and FP8 variants need no calibration:
/usr/bin/python3 eval/evaluate.py --variant int4_rtn_g128
/usr/bin/python3 eval/evaluate.py --variant fp8_w8a8
# GPTQ / AWQ quantize first (writes artifacts/), then evaluate:
/usr/bin/python3 quant/quantize.py --method gptq
/usr/bin/python3 eval/evaluate.py --variant int4_gptq_g128
# Everything, resumable, plus results/summary.json:
/usr/bin/python3 eval/run_all.py
/usr/bin/python3 charts/make_charts.py   # regenerates the PNG charts
```

Model weights are not in this repo. Qwen2.5-0.5B-Instruct and
Qwen2.5-1.5B-Instruct are public on Hugging Face under the Qwen org;
the paths in the scripts point at the copies used here (the 0.5B under
the ServeCore project folder, the 1.5B under CacheLab's). One
operational note: this suite ran across two service restarts that
killed the driver mid-run. The driver is resumable (finished variants
are skipped, quantization artifacts are reused), and every number in
the table comes from a completed evaluation run recorded in
`results/`.

## Honest limits

- One model family at two small sizes, one contended 2-core VM.
  Absolute speeds will differ everywhere else; the perplexity ordering
  (per-channel int4 collapses, grouping rescues it, GPTQ and AWQ
  recover much of the rest, 8-bit is nearly free) is the claim, and it
  matches the cited papers.
- Decode speeds measure my implementation choice (full dequantization
  per forward call, `torch._scaled_mm` fallback for FP8), not a
  property of the formats. A fused-kernel runtime would produce a
  different speed table, most likely an inverted one.
- GPTQ here has no activation-order permutation, and AWQ here applies
  its scales as an explicit input division inside the quantized linear
  rather than folded into the preceding norm, with the scale clamp
  noted in `quant/awq.py`. Both are the papers' core algorithms, not
  the full official stacks.
- The spot check is six prompts with substring scoring. Treat swings
  of one or two prompts as noise; only the per-channel int4 collapse
  (0/6, backed by perplexity 33.1) is a strong signal.
- Calibration is 2048 tokens from one book. More diverse calibration
  data is the first thing I would change before trusting these int4
  weights on other text.
- The 1.5B spot check is perplexity and size only, for the reason
  given above.

## Papers

- Frantar et al., GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers, arXiv:2210.17323
- Lin et al., AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration, arXiv:2306.00978
- Micikevicius et al., FP8 Formats for Deep Learning,
  arXiv:2209.05433
