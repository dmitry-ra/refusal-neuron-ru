"""Token-level agreement: within a run (condition vs baseline) and across runs (engine A vs B).

Prints counts only, never generated text.
"""

import argparse
import json
import os
import statistics

from .generate import read_jsonl


def prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def load(run):
    return {(r["dataset"], r["id"], r["condition"]): r for r in read_jsonl(os.path.join(run, "generations.jsonl"))}


def report(pairs, label):
    if not pairs:
        return None
    exact = sum(a == b for a, b in pairs)
    pre = [prefix(a, b) for a, b in pairs]
    res = {"label": label, "n": len(pairs), "exact": exact, "median_common_prefix": statistics.median(pre),
           "min_common_prefix": min(pre), "prefixes": pre}
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one run: within-run vs baseline; two runs: A vs B per condition")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = []
    a = load(args.runs[0])
    conds = sorted({k[2] for k in a})
    if len(args.runs) == 1:
        for c in conds:
            pairs = [(a[k]["token_ids"], a[(k[0], k[1], "baseline")]["token_ids"]) for k in a
                     if k[2] == c and (k[0], k[1], "baseline") in a]
            out.append(report(pairs, f"{c} vs baseline"))
    else:
        b = load(args.runs[1])
        for c in conds:
            pairs = [(a[k]["token_ids"], b[k]["token_ids"]) for k in a if k[2] == c and k in b]
            out.append(report(pairs, f"{c}: A vs B"))
    if args.out:
        with open(args.out, "w") as f:
            json.dump([o for o in out if o], f, indent=1)


if __name__ == "__main__":
    main()
