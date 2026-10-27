"""Greedy generation for prompts x conditions; all conditions of one prompt form one batch.

Writes into --out (default $RUN_DIR): config.json, generations.jsonl (one line per prompt x condition).
"""

import argparse
import json
import os
import platform
import time

import torch

from . import conditions, config
from .model import encode, load
from .steering import Capture, Steerer, column_norm


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratify(rows):
    """Round-robin over categories, so any prefix of the run covers every category."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get("category", ""), []).append(r)
    out, i = [], 0
    while len(out) < len(rows):
        for g in groups.values():
            if i < len(g):
                out.append(g[i])
        i += 1
    return out


def load_prompts(paths, limit=0, stratified=False):
    """Prompts from several files; stratified interleaves files and categories."""
    parts = []
    for path in paths:
        part = read_jsonl(path)
        part = part[:limit] if limit else part
        parts.append(stratify(part) if stratified else part)
    if not stratified:
        return [r for part in parts for r in part]
    out, i = [], 0
    while any(i < len(p) for p in parts):
        out += [p[i] for p in parts if i < len(p)]
        i += 1
    return out


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["dataset"], r["id"], r["condition"]) for r in read_jsonl(path)}


def prompt_medians(model, ids, layers):
    cap = Capture(model, resid_layers=layers)
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        cap.close()
    return {L: float(cap.resid[L][0, 1:].norm(dim=-1).median()) for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, nargs="+")
    ap.add_argument("--conditions", required=True, help="comma-separated, see conditions.spec")
    ap.add_argument("--max-new-tokens", type=int, default=72)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--medians", help="JSON with fixed calibrated medians {layer: value}")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts of each file")
    ap.add_argument("--stratified", action="store_true", help="interleave files and categories")
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."))
    args = ap.parse_args()

    names = args.conditions.split(",")
    rows = load_prompts(args.prompts, args.limit, args.stratified)

    tok, model, load_s = load(args.threads, args.dtype)
    steerer = Steerer(model)
    colnorm = lambda L, n: column_norm(steerer.down[L], n)
    medians = {"reference": dict(config.REFERENCE_MEDIANS)}
    if args.medians:
        with open(args.medians) as f:
            medians["calibrated"] = {int(k): float(v) for k, v in json.load(f).items()}
    need_ppm = conditions.needs_prompt_median(names)
    targets = {L for n in names for L, _, _ in conditions.spec(n)[0]}

    cfg = {
        "argv": vars(args), "conditions": {n: conditions.spec(n) for n in names},
        "model": config.MODEL_ID, "revision": config.MODEL_REVISION,
        "reference_medians": config.REFERENCE_MEDIANS, "medians": medians,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "python": platform.python_version(), "load_seconds": round(load_s, 1),
        "greedy": True, "enable_thinking": False, "n_prompts": len(rows),
    }
    import transformers
    cfg["transformers"] = transformers.__version__
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    print(f"[load] {load_s:.1f}s threads={torch.get_num_threads()} prompts={len(rows)} "
          f"conditions={names}", flush=True)

    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    out_path = os.path.join(args.out, "generations.jsonl")
    done = done_keys(out_path)
    total_tok, total_s = 0, 0.0
    with open(out_path, "a") as log:
        for i, row in enumerate(rows):
            if all((row.get("dataset", ""), row["id"], n) in done for n in names):
                continue
            ids = encode(tok, row["prompt"])
            meds = dict(medians)
            if need_ppm:
                meds["prompt"] = prompt_medians(model, ids, sorted(targets))
            plan, details = [], []
            for n in names:
                p, d = conditions.pushes(n, colnorm, meds)
                plan.append(p)
                details.append(d)
            steerer.set_rows(plan)
            batch = ids.repeat(len(names), 1)
            t = time.time()
            with torch.inference_mode():
                out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.time() - t
            steerer.clear()
            gen = out[:, ids.shape[1]:]
            steps = gen.shape[1]
            total_tok += steps * len(names)
            total_s += dt
            for r, n in enumerate(names):
                toks = gen[r].tolist()
                cut = next((k + 1 for k, x in enumerate(toks) if x in eos), len(toks))
                toks = toks[:cut]
                log.write(json.dumps({
                    "dataset": row.get("dataset", ""), "id": row["id"], "category": row.get("category", ""),
                    "condition": n, "prompt": row["prompt"],
                    "output": tok.decode(toks, skip_special_tokens=True),
                    "token_ids": toks, "n_tokens": len(toks), "hit_eos": bool(toks) and toks[-1] in eos,
                    "pushes": details[r],
                    "prompt_tokens": int(ids.shape[1]), "batch_size": len(names), "batch_steps": steps,
                    "batch_seconds": round(dt, 3), "batch_tok_per_s": round(steps * len(names) / dt, 3),
                }) + "\n")
            log.flush()
            print(f"[{i + 1}/{len(rows)}] {row.get('dataset', '')}/{row['id']} {steps} steps x "
                  f"{len(names)} in {dt:.1f}s ({steps * len(names) / dt:.2f} tok/s)", flush=True)
    print(f"[done] {total_tok} tokens in {total_s:.1f}s = {total_tok / max(total_s, 1e-9):.2f} tok/s",
          flush=True)


if __name__ == "__main__":
    main()
