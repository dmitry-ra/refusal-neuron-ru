"""Control vectors for llama.cpp from the pinned weights.

--check compares dahara1's published vectors with dose * p50_L * W_down[:, n] / ||W_down[:, n]||
computed here. --placebo K,... writes gguf/cvec/placeboK.gguf for snsteer.conditions placeboK.
Needs llama.cpp's gguf-py (--gguf-py). Only the needed down_proj tensors are read.
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snsteer import conditions, config  # noqa: E402

COMBINED = "evilqwen_L14p_L13m.gguf"


def snapshot(repo, rev):
    return os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots", rev)


def down_weight(layer):
    root = snapshot(config.MODEL_ID, config.MODEL_REVISION)
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    name = next(k for k in index if k.endswith(f".layers.{layer}.mlp.down_proj.weight") and "visual" not in k)
    with safe_open(os.path.join(root, index[name]), framework="pt") as f:
        return f.get_tensor(name).double()


def direction(targets, cache):
    out = {}
    for L, n, dose in targets:
        if L not in cache:
            cache[L] = down_weight(L)
        col = cache[L][:, n]
        out[L] = out.get(L, 0) + dose * config.REFERENCE_MEDIANS[L] * col / col.norm()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--placebo", default="")
    ap.add_argument("--outdir", default="gguf/cvec")
    args = ap.parse_args()
    sys.path.insert(0, args.gguf_py)
    import gguf
    import numpy as np

    cache = {}
    if args.check:
        want = direction(config.RECIPE, cache)
        r = gguf.GGUFReader(os.path.join(snapshot(config.CVEC_ID, config.CVEC_REVISION), COMBINED))
        for t in r.tensors:
            L = int(t.name.split(".")[1])
            got = torch.tensor(np.array(t.data, dtype=np.float64))
            cos = float(torch.nn.functional.cosine_similarity(got, want[L], dim=0))
            print(json.dumps({"tensor": t.name, "norm": float(got.norm()), "expected_norm": float(want[L].norm()),
                              "cosine": cos, "max_abs_diff": float((got - want[L]).abs().max())}))
    if args.placebo:
        os.makedirs(args.outdir, exist_ok=True)
        for k in [int(x) for x in args.placebo.split(",")]:
            targets = conditions.placebo_targets(k)
            d = direction(targets, cache)
            path = os.path.join(args.outdir, f"placebo{k}.gguf")
            w = gguf.GGUFWriter(path, "controlvector")
            w.add_string("controlvector.model_hint", "qwen35")
            w.add_uint32("controlvector.layer_count", len(d))
            w.add_description("placebo: " + ", ".join(f"L{L} N{n} dose {dose:+g}" for L, n, dose in targets))
            for L in sorted(d):
                w.add_tensor(f"direction.{L}", d[L].float().numpy())
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            print(json.dumps({"file": path, "targets": targets}))


if __name__ == "__main__":
    main()
