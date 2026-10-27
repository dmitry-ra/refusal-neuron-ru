"""Second engine: llama.cpp llama-server with dahara1's control vectors (same recipe as the HF hook).

Prompts are tokenized by the HF tokenizer and sent as token ids, so both engines see identical
input. One server per condition (control vectors are fixed at server start). Writes
generations.jsonl and config.json into --out (default $RUN_DIR) with the same keys as generate.py.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from . import config
from .generate import done_keys, load_prompts
from .model import render

# condition -> (control-vector file, scale); None means no vector.
CONDITIONS = {
    "baseline": None,
    "recipe": ("evilqwen_L14p_L13m.gguf", 1.0),
    "l14": ("L14_N8949.gguf", 1.0),
    "dose0": ("evilqwen_L14p_L13m.gguf", 0.0),
}


def vector(cond):
    """(path, scale) or None; placeboK vectors are written by scripts/make_cvec.py."""
    if cond.startswith("placebo"):
        return os.path.abspath(os.path.join("gguf", "cvec", f"{cond}.gguf")), 1.0
    if CONDITIONS[cond] is None:
        return None
    name, scale = CONDITIONS[cond]
    return cvec_path(name), scale


def cvec_path(name):
    root = os.path.join(os.environ["HF_HOME"], "hub",
                        "models--" + config.CVEC_ID.replace("/", "--"), "snapshots", config.CVEC_REVISION)
    return os.path.join(root, name)


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(port, proc, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--model", required=True, help="GGUF of the pinned revision")
    ap.add_argument("--prompts", required=True, nargs="+")
    ap.add_argument("--conditions", default="baseline,recipe")
    ap.add_argument("--max-new-tokens", type=int, default=72)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=1, help="server slots and concurrent requests")
    ap.add_argument("--ctx-per-slot", type=int, default=1024)
    ap.add_argument("--port", type=int, default=10471)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."))
    args = ap.parse_args()

    all_rows = load_prompts(args.prompts, args.limit, args.stratified)
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    names = args.conditions.split(",")
    os.makedirs(args.out, exist_ok=True)
    cfg = {"argv": vars(args), "engine": "llama.cpp llama-server",
           "conditions": {n: vector(n) for n in names},
           "model": config.MODEL_ID, "revision": config.MODEL_REVISION,
           "cvec": [config.CVEC_ID, config.CVEC_REVISION]}
    ver = subprocess.run([args.server_bin, "--version"], capture_output=True, text=True)
    cfg["server_version"] = (ver.stdout + ver.stderr).strip()
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    out_path = os.path.join(args.out, "generations.jsonl")
    done = done_keys(out_path)
    for n in names:
        rows = [r for r in all_rows if (r.get("dataset", ""), r["id"], n) not in done]
        if not rows:
            continue
        cmd = [args.server_bin, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "-t", str(args.threads), "-tb", str(args.threads), "-np", str(args.parallel),
               "-c", str(args.ctx_per_slot * args.parallel), "--no-webui", "--metrics"]
        vec = vector(n)
        if vec:
            cmd += ["--control-vector-scaled", f"{vec[0]}:{vec[1]}"]
        with open(os.path.join(args.out, f"server-{n}.log"), "w") as slog:
            slog.write(" ".join(cmd) + "\n")
            slog.flush()
            proc = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT)
            try:
                up = wait_healthy(args.port, proc)
                print(f"[{n}] server up in {up:.1f}s", flush=True)

                def one(row):
                    ids = tok(render(tok, row["prompt"]), add_special_tokens=False).input_ids
                    body = {"prompt": ids, "n_predict": args.max_new_tokens, "temperature": 0.0,
                            "top_k": 1, "seed": 0, "cache_prompt": False, "return_tokens": True,
                            "stream": False}
                    t = time.time()
                    res = post(args.port, body, timeout=1800)
                    return row, ids, res, time.time() - t

                t0 = time.time()
                with ThreadPoolExecutor(args.parallel) as ex, open(out_path, "a") as log:
                    for k, (row, ids, res, dt) in enumerate(ex.map(one, rows)):
                        toks = res.get("tokens", [])
                        tim = res.get("timings", {})
                        log.write(json.dumps({
                            "dataset": row.get("dataset", ""), "id": row["id"],
                            "category": row.get("category", ""), "condition": n, "prompt": row["prompt"],
                            "output": res.get("content", ""), "token_ids": toks, "n_tokens": len(toks),
                            "hit_eos": res.get("stop_type") == "eos", "stop_type": res.get("stop_type"),
                            "prompt_tokens": len(ids), "seconds": round(dt, 3),
                            "predicted_per_second": tim.get("predicted_per_second"),
                            "prompt_per_second": tim.get("prompt_per_second"), "engine": "llama.cpp",
                        }) + "\n")
                        log.flush()
                        print(f"[{n} {k + 1}/{len(rows)}] {row['id']} {len(toks)} tok {dt:.1f}s "
                              f"({tim.get('predicted_per_second', 0):.2f} tok/s per slot)", flush=True)
                total = time.time() - t0
                print(f"[{n}] {len(rows)} prompts in {total:.1f}s", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
