"""Write a side-by-side sample of generations per condition as markdown for human review.

The output holds generated text, so it goes next to the runs, never into the repo.
Selection is by judge labels only, so whoever builds the file does not need to read it.
"""

import argparse
import html
import os
from collections import defaultdict

from .generate import read_jsonl

# Never sampled for review, whatever the labels say.
EXCLUDED_CATEGORIES = {"Child Abuse"}


def pick(prompts, labels, flip_cond, per_category, keep_safe):
    """Per category: first prompts by id where baseline refused and flip_cond is Unsafe,
    plus (keep_safe) the first one where flip_cond stayed Safe."""
    by_cat = defaultdict(list)
    for key in sorted(prompts):
        by_cat[prompts[key]["category"]].append(key)
    chosen = []
    for cat in sorted(by_cat):
        if cat in EXCLUDED_CATEGORIES:
            continue
        keys = by_cat[cat]
        flips = [k for k in keys
                 if labels.get((*k, "baseline"), {}).get("guard_refusal") == "Yes"
                 and labels.get((*k, flip_cond), {}).get("guard_safety") == "Unsafe"]
        picked = flips[:per_category] or keys[:per_category]
        if keep_safe:
            held = [k for k in keys if labels.get((*k, flip_cond), {}).get("guard_safety") == "Safe"]
            picked += [k for k in held[:1] if k not in picked]
        chosen += [(cat, k) for k in picked]
    return chosen


def block(cond, row, label):
    tag = "no label" if not label else f"{label['guard_safety']}, refusal {label['guard_refusal']}"
    cats = ", ".join(label.get("guard_categories") or []) if label else ""
    head = f"**{cond}** - judge: {tag}" + (f" ({cats})" if cats else "")
    eos = "" if row.get("hit_eos") else " [cut at the token limit]"
    return f"{head}{eos}\n\n<pre class=\"gen\">{html.escape(row['output'].strip())}</pre>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="run dir with generations.jsonl")
    ap.add_argument("--judge", required=True, help="run dir with judge.jsonl for the same run")
    ap.add_argument("--conditions", default="baseline,recipe,placebo1")
    ap.add_argument("--flip", default="recipe", help="condition whose Unsafe labels drive the selection")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    conds = args.conditions.split(",")

    rows = {(r["dataset"], r["id"], r["condition"]): r
            for r in read_jsonl(os.path.join(args.gen, "generations.jsonl"))}
    labels = {(r["dataset"], r["id"], r["condition"]): r
              for r in read_jsonl(os.path.join(args.judge, "judge.jsonl"))}
    prompts = {(ds, pid): r for (ds, pid, c), r in rows.items() if c == "baseline"}

    out = [f"Source: `{os.path.basename(os.path.normpath(args.gen))}`, judge "
           f"`{os.path.basename(os.path.normpath(args.judge))}`.\n"]
    shown = 0
    for ds in sorted({ds for ds, _ in prompts}):
        sub = {k: v for k, v in prompts.items() if k[0] == ds}
        # Benign XSTest prompts: no "flip" to look for, show one per category.
        safe_set = ds.startswith("XSTest")
        chosen = pick(sub, labels, args.flip, 1 if safe_set else args.per_category, not safe_set)
        out.append(f"\n# {ds}\n")
        last_cat = None
        for cat, key in chosen:
            if cat != last_cat:
                out.append(f"\n## {cat}\n")
                last_cat = cat
            out.append(f"\n### {key[1]}\n\n**prompt:** {html.escape(prompts[key]['prompt'].strip())}\n")
            for c in conds:
                if (*key, c) in rows:
                    out.append(block(c, rows[(*key, c)], labels.get((*key, c))))
            shown += 1
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"{shown} prompts x {len(conds)} conditions -> {args.out}")


if __name__ == "__main__":
    main()
