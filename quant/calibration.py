"""Calibration data and block-by-block activation capture.

Calibration text is Alice's Adventures in Wonderland (Project Gutenberg
#11), the same book as the perplexity evaluation, but a DISJOINT slice:
evaluation scores tokens [0, 3072) from "CHAPTER I.", calibration uses
tokens [8192, 10240). GPTQ and AWQ never see the evaluated tokens during
calibration.

Capture follows the original GPTQ arrangement. The calibration chunks
are embedded once, then walked block by block. For each block, a forward
pass with hooks records what every linear in the block actually receives
(the Hessian sum X^T X, the mean absolute activation per input channel,
and a row subsample). The block is then quantized, and the calibration
hidden states are advanced through the quantized block before the next
block is captured, so later blocks are calibrated on inputs that already
carry earlier blocks' quantization error.
"""

import os

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_TOKENS = 3072
CALIB_START = 8192
CALIB_TOKENS = 2048
CHUNK = 256


def load_doc_ids(tok):
    text = open(os.path.join(BASE, "data_alice.txt"), encoding="utf-8").read()
    start = text.find("CHAPTER I.")
    text = text[start:]
    return tok(text, add_special_tokens=False).input_ids


def eval_ids(tok):
    return load_doc_ids(tok)[:EVAL_TOKENS]


def calib_chunks(tok):
    ids = load_doc_ids(tok)[CALIB_START:CALIB_START + CALIB_TOKENS]
    return [torch.tensor(ids[i:i + CHUNK], dtype=torch.long)
            for i in range(0, len(ids), CHUNK)]


class BlockCalibrator:
    """Walks calibration hidden states through the model block by block."""

    def __init__(self, model, chunks, subsample_rows=1024):
        self.model = model
        self.core = model.model
        self.chunks = chunks
        self.subsample_rows = subsample_rows
        L = chunks[0].shape[0]
        mask = torch.full((1, 1, L, L), torch.finfo(torch.float32).min)
        self.mask = mask.triu(1)
        self.pos = torch.arange(L).unsqueeze(0)
        with torch.no_grad():
            self.stage = [self.core.embed_tokens(c.unsqueeze(0))
                          for c in chunks]

    def _run_block(self, b, capture_names=None):
        layer = self.core.layers[b]
        caps = {}
        handles = []
        if capture_names is not None:
            mods = dict(layer.named_modules())
            for lname in capture_names:
                mod = mods[lname]
                in_f = mod.in_features
                cap = {"H": torch.zeros(in_f, in_f), "abs_sum": torch.zeros(in_f),
                       "rows": 0, "Xsub": []}
                caps[lname] = cap

                def make_hook(cap):
                    def hook(module, args, out):
                        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                        cap["H"] += x.T @ x
                        cap["abs_sum"] += x.abs().sum(dim=0)
                        cap["rows"] += x.shape[0]
                        have = sum(t.shape[0] for t in cap["Xsub"])
                        if have < self.subsample_rows:
                            cap["Xsub"].append(x[:self.subsample_rows - have].clone())
                    return hook
                handles.append(mod.register_forward_hook(make_hook(cap)))
        outputs = []
        with torch.no_grad():
            for h in self.stage:
                pos_emb = self.core.rotary_emb(h, self.pos)
                out = layer(h, attention_mask=self.mask, position_ids=self.pos,
                            position_embeddings=pos_emb, use_cache=False)
                outputs.append(out[0] if isinstance(out, tuple) else out)
        for h in handles:
            h.remove()
        if capture_names is not None:
            for cap in caps.values():
                cap["mean_abs"] = cap["abs_sum"] / max(cap["rows"], 1)
                cap["Xsub"] = torch.cat(cap["Xsub"], dim=0) if cap["Xsub"] \
                    else torch.zeros(0, cap["H"].shape[0])
                del cap["abs_sum"]
            return caps
        return outputs

    def capture(self, b, linear_names):
        """Record per-linear input statistics for block b (no advance)."""
        return self._run_block(b, capture_names=linear_names)

    def advance(self, b):
        """Push the calibration states through block b as it now stands."""
        self.stage = self._run_block(b)
