"""Aggregate judge labels into rates per engine x dataset x condition (no generated text)."""

import argparse
import json
import os
from collections import defaultdict

from .generate import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_runs", nargs="+", help="run dirs containing judge.jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    cells = defaultdict(lambda: defaultdict(int))
    for d in args.judge_runs:
        for r in read_jsonl(os.path.join(d, "judge.jsonl")):
            c = cells[(r.get("engine", "hf"), r["dataset"], r["condition"])]
            c["n"] += 1
            c["refusal"] += r["guard_refusal"] == "Yes"
            c["unsafe"] += r["guard_safety"] == "Unsafe"
            c["controversial"] += r["guard_safety"] == "Controversial"
            c["unparsed"] += r["guard_safety"] is None
    rows = []
    print(f"{'engine':9} {'dataset':18} {'condition':11} {'n':>4} {'refusal%':>9} {'unsafe%':>8} {'contr%':>7}")
    for (eng, ds, cond), c in sorted(cells.items()):
        n = c["n"]
        row = {"engine": eng, "dataset": ds, "condition": cond, **c,
               "refusal_pct": round(100 * c["refusal"] / n, 1), "unsafe_pct": round(100 * c["unsafe"] / n, 1),
               "controversial_pct": round(100 * c["controversial"] / n, 1)}
        rows.append(row)
        print(f"{eng:9} {ds:18} {cond:11} {n:4d} {row['refusal_pct']:9.1f} {row['unsafe_pct']:8.1f} "
              f"{row['controversial_pct']:7.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
