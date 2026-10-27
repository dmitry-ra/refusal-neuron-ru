# AGENTS.md

Instructions for a coding agent that is asked to reproduce the experiment in this repository on a
clean CPU-only Linux machine. `README.md` explains the recipe for humans; this file is the
operational checklist.

## Goal

Compare greedy generations of `Qwen/Qwen3.5-4B` without steering (`baseline`) and with the
single-neuron recipe (`recipe`: at every token add `s` to the `down_proj` input of layer 14
neuron 8949 with dose +1 and of layer 13 neuron 7956 with dose -1, layers 0-based), using two
independent engines:

1. HF transformers, fp32, forward pre-hook (`snsteer/generate.py`).
2. llama.cpp `llama-server` with a BF16 GGUF and the published control vector
   `evilqwen_L14p_L13m.gguf` at scale 1.0 (`snsteer/llama_run.py`).

## Guardrails

- Never commit generations, judge outputs, logs, run directories, model files or GGUF files.
  They live in `runs/`, `hf/`, `gguf/`, `.uv/`, `.venv/`, which are ignored. Check
  `git status` before every commit.
- Generated text for the safety prompts can be harmful. Keep it on the machine that produced it;
  do not paste it into commit messages, issues, pull requests or reports. Report rates and
  counts, or short neutral paraphrases.
- `data/sst.jsonl` (SimpleSafetyTests) contains deliberately harmful requests. It is used only to
  measure refusals and judged safety of responses. Do not use it for anything else.
- Do not push to `main`. Work on a feature branch.
- Code, comments and docs are English and ASCII only (`grep -nP '[^\x00-\x7F]'` must be empty), with two
  exceptions: `README.md`, the main README, and `docs/examples.md`, which are in Russian
  (`README.en.md` is the English README). Non-ASCII test data goes in as `\u` escapes.
- Do not change the pinned versions, revisions or constants in `snsteer/config.py` unless the
  task says so; results are only comparable at the pinned state.
- Run every heavy command through `scripts/run_job.sh` so the run is logged, under a timeout and
  `nice -n 10`.

## Prerequisites

- x86_64 Linux, no GPU needed. 32 GB RAM minimum (HF fp32 model takes about 21 GB RSS; the
  llama.cpp BF16 server about 9 GB; running both at once needs about 32 GB plus the OS).
- About 25 GB free disk: HF snapshots 10 GB, BF16 GGUF 8.7 GB, uv Python, cache and venv 2 GB.
- `git`, `curl`, `rsync`, and for llama.cpp `cmake` and a C++17 compiler.
- Network access to `huggingface.co`, `download.pytorch.org`, `pypi.org`, `astral.sh`,
  `github.com`.

## Steps

All commands run from the repository root. `scripts/env.sh` keeps `HF_HOME`, the uv cache, the
uv-managed Python and the venv inside the repository directory.

1. Install uv 0.10.8 user-locally (only host change besides the repository directory):

   ```bash
   mkdir -p .uv && curl -LsSf -o .uv/uv-installer-0.10.8.sh https://astral.sh/uv/0.10.8/install.sh
   sha256sum .uv/uv-installer-0.10.8.sh  # eae5e1dae89cd0b74d357f549ccd6faa94b2ad6c1d89d78972a625655a4556ae
   UV_INSTALL_DIR=$HOME/.local/bin UV_NO_MODIFY_PATH=1 sh .uv/uv-installer-0.10.8.sh
   ```

2. Environment and models. `fetch_models.py` downloads the pinned revisions and fails on any
   sha256 mismatch against the Hub LFS metadata; compare the printed hashes with the table in
   `README.md`.

   ```bash
   scripts/run_job.sh setup 3600 bash -c \
     'uv sync --frozen --group convert && uv run --frozen python scripts/fetch_models.py --out $RUN_DIR/models-manifest.json'
   ```

3. llama.cpp at the pinned commit (skip if a build of this commit exists; pass its directory as
   `LLAMA` below):

   ```bash
   LLAMA=$HOME/llama.cpp-2280f0f
   git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
   git -C "$LLAMA" checkout 2280f0fd163b03d4331d49e08460f66502e35854
   cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release
   cmake --build "$LLAMA/build" -j 8 --target llama-server
   ```

4. BF16 GGUF from the same HF snapshot with llama.cpp's converter, and the vector checks:

   ```bash
   scripts/run_job.sh convert-bf16 3600 scripts/convert_gguf.sh "$LLAMA" bf16
   scripts/run_job.sh cvec 300 uv run --frozen python scripts/make_cvec.py \
     --gguf-py "$LLAMA/gguf-py" --check --placebo 1,2,3
   ```

   Expected: GGUF sha256 `700cebb6c57c015751e7306f808d0dbe40a9cbcfcbda01a83bcc6e210501cbbd`
   with llama.cpp `2280f0f` (another converter version gives another hash, which is acceptable
   but must be recorded). `--check` must print cosine 1.0 and max_abs_diff below 1e-5 for
   `direction.13` and `direction.14`.

5. Baseline vs recipe, HF engine:

   ```bash
   scripts/run_job.sh e0-hf 1800 uv run --frozen python -m snsteer.generate \
     --prompts data/smoke_prompts.jsonl --conditions baseline,recipe,l14,dose0 --max-new-tokens 48 --threads 6
   uv run --frozen python -m snsteer.compare runs/<UTC>-e0-hf
   ```

6. Baseline vs recipe, llama.cpp engine, then engine agreement:

   ```bash
   scripts/run_job.sh e0-llama 1800 uv run --frozen python -m snsteer.llama_run \
     --server-bin "$LLAMA/build/bin/llama-server" --model gguf/Qwen3.5-4B-851bf6e-BF16.gguf \
     --prompts data/smoke_prompts.jsonl --conditions baseline,recipe,l14,dose0 --max-new-tokens 48 --threads 6
   uv run --frozen python -m snsteer.compare runs/<UTC>-e0-llama
   uv run --frozen python -m snsteer.compare runs/<UTC>-e0-hf runs/<UTC>-e0-llama
   ```

7. Optional: scale check (`snsteer.calibrate`), throughput (`snsteer.bench`), judge
   (`snsteer.judge`) and rates (`snsteer.summary`); see `README.md` -> Use.

## Expected checks

- `pushes` in the HF `generations.jsonl` show `colnorm` 0.464718 (L14 N8949) and 0.418273
  (L13 N7956) and `value` (= s) +22.269 and -23.294.
- `dose0 vs baseline`: exact 5/5 in both engines (token for token).
- `recipe vs baseline` and `l14 vs baseline`: 0 exact; the common prefix is usually 0 to 2 tokens.
- HF vs llama.cpp per condition: most prompts identical over 48 tokens; the rest diverge late at
  near-ties (fp32 vs BF16 arithmetic; observed 3 to 4 of 5 identical per condition).
  `--parallel N > 1` changes llama.cpp numerics slightly, so its outputs can differ from
  `--parallel 1`. HF results do not depend on the batch composition (checked: batch 4 vs 7
  identical). Total disagreement from the first token in every prompt means a wrong layer index
  or vector.

## Calibration constants

`s = dose * p50_L / ||W_down[:, n]||` with fixed `p50_14 = 10.349`, `p50_13 = 9.7432`
(`snsteer/config.py: REFERENCE_MEDIANS`). These are the original author's medians of the
residual-stream norm leaving each layer, recovered as the L2 norms of the direction tensors in
`dahara1/EvilQwen-control-vectors` at revision `10f135cc8a13e1fc0f0d6a5a758d8064c3d37d1b`
(Apache-2.0). `scripts/make_cvec.py --check` confirms those tensors equal
`dose * p50_L * W_down[:, n] / ||W_down[:, n]||` for the pinned weights. Our own pooled median
over `data/neutral_prompts.jsonl` comes from `snsteer.calibrate` and is a cross-check only.

## Resources and timing (Intel i5-14500, 20 threads, DDR5)

- HF fp32: model load about 5 s from page cache. Decode is memory-bound: about 3.5 tok/s at
  batch 1; batch 6 gives about 12 to 14 tok/s total. 6 threads are as fast as 12 to 14.
- llama.cpp BF16: about 7.3 tok/s per request alone; with `--parallel 8` / `16` about 4.6 / 2.7
  tok/s per slot, 37 / 43 tok/s total while all slots are busy. 6 threads are as fast as 12.
- Judge: about 3.3 s per response with 3 threads (older 6-core CPU).
- Do not run the HF and llama.cpp engines at the same time on one machine: both are memory-bound
  and together they were slower than llama.cpp alone.
- Smoke (5 prompts x 4 conditions x 48 tokens): about 100 s HF, about 2.5 min llama.cpp
  (one server start per condition).
- Keep the machine's load average below about 75 % of its threads if it is shared.

## Outputs

`scripts/run_job.sh NAME TIMEOUT CMD` creates `runs/<UTC>-NAME/` with `cmd.txt`, `commit.txt`,
`pip-freeze.txt`, `lock.sha256`, `host.txt` (load before and after), `timing.json`, `log.txt`,
and whatever the command writes there (`config.json`, `generations.jsonl`, `judge.jsonl`,
`calibration.json`, `bench.jsonl`). Set `SNSTEER_RUNS` to put run directories elsewhere. None
of this is committed.
