"""Wall time per module class during greedy decoding (CPU ops are synchronous, so hooks time them).

Answers how much of a decode step goes to the Gated DeltaNet layers, which run the reference
PyTorch kernels on CPU, versus attention, MLP and the LM head.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch

from .generate import read_jsonl
from .model import encode, load

CLASSES = ("Qwen3_5GatedDeltaNet", "Qwen3_5Attention", "Qwen3_5MLP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batches", default="1,6")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."))
    args = ap.parse_args()
    tok, model, _ = load(args.threads)
    acc, starts = defaultdict(float), {}
    mods = [(n, m) for n, m in model.named_modules() if type(m).__name__ in CLASSES]
    mods.append(("lm_head", model.lm_head))

    def pre(name):
        def f(_m, _a):
            starts[name] = time.perf_counter()
        return f

    def post(name, cls):
        def f(_m, _a, _o):
            acc[cls] += time.perf_counter() - starts[name]
        return f

    for name, m in mods:
        cls = type(m).__name__ if name != "lm_head" else "lm_head"
        m.register_forward_pre_hook(pre(name))
        m.register_forward_hook(post(name, cls))
    ids = encode(tok, read_jsonl(args.prompts)[0]["prompt"])
    res = []
    for b in [int(x) for x in args.batches.split(",")]:
        batch = ids.repeat(b, 1)
        acc.clear()
        t = time.time()
        with torch.inference_mode():
            model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), max_new_tokens=args.tokens,
                           min_new_tokens=args.tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        total = time.time() - t
        rec = {"batch": b, "tokens": args.tokens, "threads": args.threads, "total_s": round(total, 3),
               "per_class_s": {k: round(v, 3) for k, v in acc.items()},
               "per_class_pct": {k: round(100 * v / total, 1) for k, v in acc.items()}}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(args.out, "profile.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
