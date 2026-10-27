"""Scale check: residual-norm medians per layer and the natural range of the target neurons.

For each neutral prompt: greedy baseline continuation of --continue-tokens, then one clean forward
over prompt + continuation, recording ||resid|| leaving every layer and the down_proj input of the
target neurons. Writes calibration.json and medians.json (pooled fixed medians, all layers).
"""

import argparse
import json
import os

import torch

from . import config
from .generate import read_jsonl
from .model import encode, load
from .steering import Capture, column_norm, dose_value, find_down_projs


def quantiles(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    v = torch.quantile(x, torch.tensor(qs, dtype=torch.float64))
    out = {f"q{int(q * 100):02d}": round(float(a), 5) for q, a in zip(qs, v)}
    out["absmax"] = round(float(x.abs().max()), 5)
    out["mean"] = round(float(x.mean()), 5)
    out["n"] = int(x.numel())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--continue-tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."))
    args = ap.parse_args()

    tok, model, _ = load(args.threads)
    down = find_down_projs(model)
    targets = [(L, n) for L, n, _ in config.RECIPE]
    layers = list(range(config.N_LAYERS))
    norms = {L: [] for L in layers}
    per_prompt = {L: [] for L, _ in targets}
    acts = {(L, n): {"prompt": [], "response": []} for L, n in targets}
    for row in read_jsonl(args.prompts):
        ids = encode(tok, row["prompt"])
        with torch.inference_mode():
            full = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                  max_new_tokens=args.continue_tokens, temperature=None, top_p=None,
                                  top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        cap = Capture(model, resid_layers=layers, mlp_in=[(L, [n]) for L, n in targets])
        try:
            with torch.inference_mode():
                model(input_ids=full, use_cache=False, logits_to_keep=1)
        finally:
            cap.close()
        p = ids.shape[1]
        for L in layers:
            norms[L] += cap.resid[L][0, 1:].norm(dim=-1).tolist()
        for L, n in targets:
            per_prompt[L].append(float(cap.resid[L][0, 1:p].norm(dim=-1).median()))
            a = cap.mlp_in[L][0, :, 0]
            acts[(L, n)]["prompt"] += a[:p].tolist()
            acts[(L, n)]["response"] += a[p:].tolist()
        print(f"{row['id']} prompt={p} total={full.shape[1]}", flush=True)

    medians = {L: float(torch.tensor(norms[L]).median()) for L in layers}
    result = {"prompts": args.prompts, "continue_tokens": args.continue_tokens,
              "pooled_median_resid": medians,
              "reference_medians": config.REFERENCE_MEDIANS,
              "per_prompt_median_resid": {L: quantiles(v) for L, v in per_prompt.items()},
              "neurons": {}}
    for (L, n), d in acts.items():
        dose = next(dd for LL, nn, dd in config.RECIPE if (LL, nn) == (L, n))
        cn = column_norm(down[L], n)
        allv = d["prompt"] + d["response"]
        st = quantiles(allv)
        s_ref = dose_value(dose, config.REFERENCE_MEDIANS[L], cn)
        s_own = dose_value(dose, medians[L], cn)
        result["neurons"][f"L{L}N{n}"] = {
            "dose": dose, "colnorm": cn, "s_reference": s_ref, "s_own_calibration": s_own,
            "activation_all": st, "activation_prompt": quantiles(d["prompt"]),
            "activation_response": quantiles(d["response"]) if d["response"] else None,
            "s_over_absmax": abs(s_ref) / max(st["absmax"], 1e-12),
        }
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(args.out, "medians.json"), "w") as f:
        json.dump(medians, f, indent=1)
    for L, _ in targets:
        print(f"L{L}: pooled median {medians[L]:.4f} vs reference {config.REFERENCE_MEDIANS[L]}")
    for k, v in result["neurons"].items():
        print(k, json.dumps({x: v[x] for x in ("colnorm", "s_reference", "s_own_calibration",
                                                "s_over_absmax")}), json.dumps(v["activation_all"]))


if __name__ == "__main__":
    main()
