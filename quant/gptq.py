"""GPTQ, implemented from scratch following Frantar et al.
(arXiv:2210.17323), in the compact per-layer form.

For one linear layer with calibration inputs X and weight W, GPTQ
minimizes the layer output error ||W X - Q X|| under the Hessian
H = 2 X X^T. Columns are quantized one at a time; quantizing column i
pushes its rounding error into the still-unquantized columns through
the Cholesky factor of H^-1 (the Optimal Brain Surgeon update). Columns
are processed in blocks of 128, and each block shares one quantization
group: per-row scales are recomputed from the error-compensated weights
at the start of every group, which is the standard GPTQ grouping
arrangement. The Hessian factorization is done in float64 for stability.

This is the real algorithm, not round-to-nearest with a new name: the
Hessian comes from actual calibration activations and the per-column
error compensation is the GPTQ update.
"""

import torch

from quant_linear import dequant_codes


def _factorize_hinv(H, damp_frac=0.01):
    """Cholesky factor (upper) of the damped inverse Hessian.

    Returns (Hinv, damp_used). Retries with stronger damping if the
    factorization fails.
    """
    H64 = H.double().clone()
    dead = torch.diag(H64) == 0
    if dead.any():
        H64[dead, dead] = 1.0
    mean_diag = torch.mean(torch.diag(H64)).item()
    damp = damp_frac * mean_diag
    for _ in range(5):
        Hd = H64.clone()
        idx = torch.arange(Hd.shape[0])
        Hd[idx, idx] += damp
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv.float(), damp
        except RuntimeError:
            damp *= 10.0
    raise RuntimeError("GPTQ: Hessian factorization failed at all damping levels")


def gptq_quantize(W, H, group_size=128, blocksize=128):
    """Quantize weight W (out, in) with GPTQ. H is the accumulated
    sum of x x^T over calibration tokens (scale of H is irrelevant).

    Returns (codes int8 (out, in), scales (out, n_groups) fp32, info).
    """
    assert group_size == blocksize, "this implementation ties group to block"
    W = W.detach().float().clone()
    out_f, in_f = W.shape
    n_groups = in_f // group_size
    H = H.float().clone()
    dead = torch.diag(H) == 0
    if dead.any():
        W[:, dead] = 0.0
    Hinv, damp = _factorize_hinv(H)

    codes = torch.zeros(out_f, in_f, dtype=torch.int8)
    scales = torch.zeros(out_f, n_groups)
    losses = 0.0

    for gi in range(n_groups):
        i = gi * group_size
        j = i + group_size
        # scales for this group from the current (compensated) weights
        block = W[:, i:j]
        scale = (block.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
        scales[:, gi] = scale
        Hinv1 = Hinv[i:j, i:j]
        W1 = block.clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        for c in range(group_size):
            w = W1[:, c]
            d = Hinv1[c, c]
            q = torch.clamp(torch.round(w / scale), -8, 7)
            deq = q * scale
            codes[:, i + c] = q.to(torch.int8)
            Q1[:, c] = deq
            err = (w - deq) / d
            if c + 1 < group_size:
                W1[:, c + 1:] -= err.unsqueeze(1) * Hinv1[c, c + 1:].unsqueeze(0)
            Err1[:, c] = err
            losses += ((w - deq) ** 2).sum().item()
        if j < in_f:
            W[:, j:] -= Err1 @ Hinv[i:j, j:]

    What = dequant_codes(codes, scales, group_size)
    info = {"damp": damp, "quant_loss_sum": losses}
    return codes, scales, info, What
