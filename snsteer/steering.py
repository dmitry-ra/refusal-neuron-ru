"""Steering of single MLP neurons through a forward pre-hook on mlp.down_proj.

The down_proj input h is silu(gate(x)) * up(x), one value per neuron. Two modes:
  add       - h[..., n] + s at every position (prompt and generated tokens)
  overwrite - h[..., n] = s at every position
with s = dose * median(||resid leaving layer L||) / ||W_down[:, n]||.

Each batch row carries its own condition, so all conditions of one prompt run as one batch.
"""

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Push:
    layer: int
    neuron: int
    value: float
    mode: str = "add"


def find_down_projs(model):
    out = {}
    for name, mod in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.mlp\.down_proj$", name)
        if m:
            out[int(m.group(1))] = mod
    return out


def find_layers(model):
    out = {}
    for name, mod in model.named_modules():
        m = re.search(r"\.layers\.(\d+)$", name)
        if m and hasattr(mod, "mlp"):
            out[int(m.group(1))] = mod
    return out


def column_norm(down_proj, neuron):
    return float(down_proj.weight[:, neuron].detach().double().norm())


def dose_value(dose, median, colnorm):
    return dose * median / colnorm


class Steerer:
    """Owns one pre-hook per steerable layer; rows are set per batch with set_rows()."""

    def __init__(self, model):
        self.down = find_down_projs(model)
        self.plan = {}
        self.handles = [mod.register_forward_pre_hook(self._hook(L)) for L, mod in self.down.items()]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.plan = {}

    def set_rows(self, rows):
        """rows: list (one per batch row) of lists of Push."""
        plan = {}
        nrows = len(rows)
        per_layer = {}
        for r, pushes in enumerate(rows):
            for p in pushes:
                if p.mode not in ("add", "overwrite"):
                    raise ValueError(p.mode)
                per_layer.setdefault(p.layer, []).append((r, p))
        for L, items in per_layer.items():
            neurons = sorted({p.neuron for _, p in items})
            col = {n: i for i, n in enumerate(neurons)}
            add = torch.zeros(nrows, len(neurons), dtype=torch.float64)
            ow_val = torch.zeros(nrows, len(neurons), dtype=torch.float64)
            ow_mask = torch.zeros(nrows, len(neurons), dtype=torch.bool)
            for r, p in items:
                if p.mode == "add":
                    add[r, col[p.neuron]] += p.value
                else:
                    ow_val[r, col[p.neuron]] = p.value
                    ow_mask[r, col[p.neuron]] = True
            plan[L] = (torch.tensor(neurons), add, ow_val, ow_mask)
        self.plan = plan

    def _hook(self, layer):
        def hook(module, args):
            entry = self.plan.get(layer)
            if entry is None:
                return None
            idx, add, ow_val, ow_mask = entry
            h = args[0]
            if h.shape[0] != add.shape[0]:
                raise RuntimeError(f"batch {h.shape[0]} != steering rows {add.shape[0]}")
            cur = h[..., idx]
            new = cur + add.to(h.dtype)[:, None, :]
            new = torch.where(ow_mask[:, None, :], ow_val.to(h.dtype)[:, None, :], new)
            h = h.clone()
            h[..., idx] = new
            return (h,) + tuple(args[1:])
        return hook


class Capture:
    """Records decoder-layer outputs (residual stream) and down_proj inputs for chosen layers."""

    def __init__(self, model, resid_layers=(), mlp_in=()):
        self.resid, self.mlp_in, self.handles = {}, {}, []
        layers = find_layers(model)
        down = find_down_projs(model)
        for L in resid_layers:
            self.handles.append(layers[L].register_forward_hook(self._resid(L)))
        for L, neurons in mlp_in:
            self.handles.append(down[L].register_forward_pre_hook(self._mlp(L, list(neurons))))

    def _resid(self, L):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self.resid[L] = h.detach().float()
        return hook

    def _mlp(self, L, neurons):
        def hook(module, args):
            self.mlp_in[L] = args[0][..., neurons].detach().float()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
