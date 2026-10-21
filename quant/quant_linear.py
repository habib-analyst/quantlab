"""From-scratch quantized linear layers for QuantLab.

Two storage formats live here:

* ``int`` mode: weight-only integer quantization. Codes are stored either
  as int8 tensors (8-bit) or packed two-per-byte uint8 tensors (4-bit).
  Scales are per output channel, optionally per group of ``group_size``
  input columns. An optional per-input-channel scale vector implements
  the AWQ equivalent transform (the input is divided by the scales, the
  weights were multiplied by them before quantization). The forward pass
  dequantizes the weight to the input's dtype and calls F.linear. That is
  the honest CPU story for weight-only quantization without fused
  kernels: memory shrinks, compute does not get faster.

* ``fp8`` mode: real W8A8 FP8 compute. Weights are stored as
  float8_e4m3fn with one fp32 scale per output channel. Activations are
  quantized per token (per row) to e4m3 at runtime and the matmul runs
  through ``torch._scaled_mm`` with row-wise scaling. On this CPU build
  of PyTorch that operator is numerically correct but slow (a fallback,
  not a tensor-core kernel); the benchmark numbers in this project
  measure exactly that, they are not hidden.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0


# ---------------------------------------------------------------- RTN core

def rtn_codes(W, bits, group_size):
    """Round-to-nearest symmetric quantization of a 2D weight.

    Returns (codes, scales): codes is an int8 tensor shaped like W with
    values in [-(2^(bits-1)), 2^(bits-1)-1]; scales is (out, n_groups)
    fp32. group_size == W.shape[1] gives per-channel quantization.
    """
    out_f, in_f = W.shape
    assert in_f % group_size == 0, f"in={in_f} not divisible by group {group_size}"
    n_groups = in_f // group_size
    qmax = 2 ** (bits - 1) - 1          # 7 for int4, 127 for int8
    qmin = -(2 ** (bits - 1))           # -8 for int4, -128 for int8
    Wg = W.float().reshape(out_f, n_groups, group_size)
    scale = (Wg.abs().amax(dim=2) / qmax).clamp_min(1e-12)
    codes = torch.clamp(torch.round(Wg / scale.unsqueeze(2)), qmin, qmax)
    return codes.reshape(out_f, in_f).to(torch.int8), scale


def dequant_codes(codes, scales, group_size):
    """Inverse of rtn_codes: returns the fp32 dequantized weight."""
    out_f, in_f = codes.shape
    n_groups = in_f // group_size
    Wg = codes.float().reshape(out_f, n_groups, group_size)
    return (Wg * scales.unsqueeze(2)).reshape(out_f, in_f)


def pack_int4(codes):
    """Pack int8 codes in [-8, 7] two per byte along the input dim."""
    vals = (codes.to(torch.int16) + 8).to(torch.uint8)   # 0..15
    lo = vals[:, 0::2]
    hi = vals[:, 1::2]
    return lo | (hi << 4)


def unpack_int4(packed, in_f):
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    vals = torch.stack([lo, hi], dim=2).reshape(packed.shape[0], in_f)
    return (vals - 8).to(torch.int8)


def fp8_quantize_weight(W):
    """Per-output-channel e4m3 weight quantization.

    Returns (w_fp8 float8_e4m3fn tensor, scale (out, 1) fp32).
    """
    scale = (W.float().abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
    return (W.float() / scale).to(torch.float8_e4m3fn), scale


# ---------------------------------------------------------------- the layer

class QuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear with a quantized weight."""

    def __init__(self, in_features, out_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = None            # "int" or "fp8"
        self.bits = None
        self.group_size = None
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    # -- construction -------------------------------------------------

    @classmethod
    def from_rtn(cls, linear, bits, group_size, in_scales=None):
        W = linear.weight.detach().float()
        codes, scales = rtn_codes(W, bits, group_size)
        q = cls(linear.in_features, linear.out_features, linear.bias)
        q.mode, q.bits, q.group_size = "int", bits, group_size
        if bits == 4:
            q.register_buffer("codes", pack_int4(codes))
        else:
            q.register_buffer("codes", codes)
        q.register_buffer("scales", scales)
        if in_scales is not None:
            q.register_buffer("in_scales", in_scales.float())
        else:
            q.in_scales = None
        return q

    @classmethod
    def from_codes(cls, linear, codes, scales, group_size, in_scales=None):
        """Build from precomputed int4 codes (GPTQ / AWQ outputs)."""
        q = cls(linear.in_features, linear.out_features, linear.bias)
        q.mode, q.bits, q.group_size = "int", 4, group_size
        q.register_buffer("codes", pack_int4(codes))
        q.register_buffer("scales", scales.float())
        if in_scales is not None:
            q.register_buffer("in_scales", in_scales.float())
        else:
            q.in_scales = None
        return q

    @classmethod
    def from_fp8(cls, linear):
        w_fp8, scale = fp8_quantize_weight(linear.weight.detach())
        q = cls(linear.in_features, linear.out_features, linear.bias)
        q.mode = "fp8"
        q.register_buffer("w_fp8", w_fp8)
        q.register_buffer("w_scale", scale)
        return q

    # -- accounting ---------------------------------------------------

    def stored_bytes(self):
        n = 0
        if self.mode == "int":
            n += self.codes.untyped_storage().nbytes()
            n += self.scales.untyped_storage().nbytes()
            if self.in_scales is not None:
                n += self.in_scales.untyped_storage().nbytes()
        elif self.mode == "fp8":
            n += self.w_fp8.untyped_storage().nbytes()
            n += self.w_scale.untyped_storage().nbytes()
        if self.bias is not None:
            n += self.bias.untyped_storage().nbytes()
        return n

    def dequantized_weight(self):
        if self.bits == 4:
            codes = unpack_int4(self.codes, self.in_features)
        else:
            codes = self.codes
        return dequant_codes(codes, self.scales, self.group_size)

    # -- forward ------------------------------------------------------

    def forward(self, x):
        if self.mode == "int":
            if self.in_scales is not None:
                x = x / self.in_scales.to(x.dtype)
            W = self.dequantized_weight().to(x.dtype)
            return F.linear(x, W, self.bias)
        if self.mode == "fp8":
            return self._fp8_forward(x)
        raise RuntimeError("QuantLinear mode not set")

    def _fp8_forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features).float()
        sa = (x2.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
        xq = (x2 / sa).to(torch.float8_e4m3fn)
        sb = self.w_scale.reshape(1, -1).contiguous()
        try:
            out = torch._scaled_mm(xq, self.w_fp8.t(), scale_a=sa,
                                   scale_b=sb, out_dtype=torch.float32)
            out = out[0] if isinstance(out, tuple) else out
        except (RuntimeError, ValueError):
            # Some row counts are rejected by the CPU fallback; fall back
            # to one row at a time. Same math, same operator, slower.
            rows = []
            for i in range(x2.shape[0]):
                o = torch._scaled_mm(xq[i:i + 1], self.w_fp8.t(),
                                     scale_a=sa[i:i + 1], scale_b=sb,
                                     out_dtype=torch.float32)
                rows.append(o[0] if isinstance(o, tuple) else o)
            out = torch.cat(rows, dim=0)
        out = out.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


# ---------------------------------------------------------------- helpers

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def target_linears(model):
    """All (name, module) nn.Linear inside decoder blocks, in order."""
    found = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGET_SUFFIXES \
                and ".layers." in name:
            found.append((name, mod))
    return found


def get_parent(model, name):
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]


def replace_linear(model, name, new_module):
    parent, attr = get_parent(model, name)
    if attr.isdigit():
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


def model_stored_bytes(model):
    """Exact parameter bytes of the model in its stored form.

    Counts every unique parameter once (tied weights deduplicated),
    plus the quantization buffers (codes, scales) of QuantLinear
    modules. Rotary embedding buffers are excluded; they are a few
    hundred KB of fp32 shared by every variant and are noted in the
    README instead of counted.
    """
    total = 0
    seen = set()
    for p in model.parameters():
        if p.untyped_storage().data_ptr() in seen:
            continue
        seen.add(p.untyped_storage().data_ptr())
        total += p.untyped_storage().nbytes()
    for mod in model.modules():
        if isinstance(mod, QuantLinear):
            total += mod.stored_bytes()
            if mod.bias is not None:      # bias already counted as a parameter
                total -= mod.bias.untyped_storage().nbytes()
    return total
