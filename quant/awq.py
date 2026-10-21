"""AWQ, the core idea implemented from scratch following Lin et al.
(arXiv:2306.00978).

AWQ's observation: a small fraction of input channels matter far more
than the rest, because their activations are consistently large. Scaling
those channels up before quantization (and dividing the incoming
activations by the same factors, an exactly equivalent transform)
spreads the important channels over more of the quantization grid and
protects them.

What is implemented here, per linear layer:

1. Per-input-channel activation magnitude s_x = mean |x| over the
   calibration tokens that actually flowed into this layer.
2. A grid search over alpha in {0, 0.25, 0.5, 0.75, 1.0}: candidate
   scales s = s_x^alpha, normalized to mean 1 (a constant factor is
   absorbed by the equivalent transform; the search is over the shape
   of s, and s is clamped to [0.01, 100] for numerical safety).
3. For each alpha, the scaled weights W * diag(s) are quantized with
   group-128 round-to-nearest int4, unscaled back, and scored by the
   relative output error ||(W - W_hat) X||_F / ||W X||_F on a held
   subsample of the layer's real calibration inputs.
4. The best alpha wins. The winning scales ship inside the QuantLinear
   as in_scales, so inference divides activations by s before the
   dequantized matmul, exactly the AWQ equivalent transform. Production
   AWQ folds the division into the preceding norm or activation; the
   math is identical, the placement is mine.
"""

import torch

from quant_linear import dequant_codes, rtn_codes

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def awq_quantize(W, Xsub, mean_abs, group_size=128):
    """Quantize weight W (out, in) with AWQ scaling search.

    Xsub: (n, in) subsample of real calibration inputs to this layer.
    mean_abs: (in,) mean |activation| per input channel.
    Returns (codes, scales, in_scales, info).
    """
    W = W.detach().float()
    X = Xsub.float()
    ref = X @ W.T
    ref_norm = torch.linalg.norm(ref).clamp_min(1e-12)

    best = None
    errors = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            s = torch.ones_like(mean_abs)
        else:
            s = mean_abs.clamp_min(1e-8) ** alpha
            s = (s / s.mean()).clamp(0.01, 100.0)
        Ws = W * s.unsqueeze(0)
        codes, scales = rtn_codes(Ws, 4, group_size)
        What = dequant_codes(codes, scales, group_size) / s.unsqueeze(0)
        err = (torch.linalg.norm(X @ (W - What).T) / ref_norm).item()
        errors[alpha] = round(err, 5)
        if best is None or err < best[0]:
            best = (err, codes, scales, s, What, alpha)
    err, codes, scales, s, What, best_alpha = best
    info = {"alpha_errors": errors, "best_rel_err": round(err, 5),
            "best_alpha": best_alpha}
    return codes, scales, s, info, What
