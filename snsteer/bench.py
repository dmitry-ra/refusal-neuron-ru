"""Throughput of batched greedy generation over thread counts and batch sizes.

Every configuration generates exactly --tokens new tokens (EOS suppressed by min_new_tokens),
so tok/s is comparable. Writes bench.jsonl into --out (default $RUN_DIR).
"""

import argparse
import json
import os
import time

import torch

from . import conditions, config
from .generate import read_jsonl
from .model import encode, load
from .steering import Steerer, column_norm

ORDER = ["baseline", "recipe", "l14", "dose0", "recipe_ppm", "overwrite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--threads", default="6,12,14")
    ap.add_argument("--batches", default="1,3,6")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."))
    args = ap.parse_args()
    threads = [int(x) for x in args.threads.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    tok, model, load_s = load(threads[0])
    print(f"[load] {load_s:.1f}s", flush=True)
    steerer = Steerer(model)
    colnorm = lambda L, n: column_norm(steerer.down[L], n)
    prompt = read_jsonl(args.prompts)[0]
    ids = encode(tok, prompt["prompt"])
    meds = {"reference": dict(config.REFERENCE_MEDIANS), "prompt": dict(config.REFERENCE_MEDIANS)}

    def run(b, n_tok):
        steerer.set_rows([conditions.pushes(n, colnorm, meds)[0] for n in ORDER[:b]])
        batch = ids.repeat(b, 1)
        t = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                 max_new_tokens=n_tok, min_new_tokens=n_tok, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        steerer.clear()
        return time.time() - t, out

    with open(os.path.join(args.out, "bench.jsonl"), "a") as log:
        for th in threads:
            torch.set_num_threads(th)
            run(1, 4)
            for b in batches:
                for rep in range(args.repeats):
                    with open("/proc/loadavg") as f:
                        load_before = f.read().split()[0]
                    dt, out = run(b, args.tokens)
                    steps = out.shape[1] - ids.shape[1]
                    rec = {"threads": th, "batch": b, "conditions": ORDER[:b], "steps": steps,
                           "prompt_tokens": int(ids.shape[1]), "seconds": round(dt, 3),
                           "tok_per_s": round(b * steps / dt, 3), "steps_per_s": round(steps / dt, 3),
                           "load_before": load_before, "repeat": rep}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
