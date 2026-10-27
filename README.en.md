# refusal-neuron-ru (English)

Reproducing single-neuron steering of Qwen3.5-4B on CPU: adding a fixed value to one MLP neuron
(layer 14, neuron 8949) and subtracting one from another (layer 13, neuron 7956) at every token
turns the model into a cartoon villain and removes most refusals.

This is the English version. The main README is in Russian: `README.md`.

## Credits

- The effect and the recipe: the post by @_lyraaaa_ on X, 24 Oct 2026,
  https://x.com/_lyraaaa_/status/2113833330429083662.
- Control vectors for llama.cpp: https://huggingface.co/dahara1/EvilQwen-control-vectors
  (downloaded at a pinned revision, not redistributed).
- Benchmark harness, judge prompt and the first measured replication:
  https://github.com/sarvkk/two-neurons (MIT; adapted parts listed in `NOTICE`).

This repository adds a CPU-only reproduction with two independent engines, placebo neurons of the
same strength, confidence intervals and paired tests, and a Russian translation of the prompt sets.

## Responsible use

This is a measurement of how a single neuron changes refusal behaviour, not a jailbreak kit. The
recipe and ready-made control vectors were public before this repository existed. What is
published here is code, prompts, pinned versions and aggregate rates. No answer to a harmful prompt
is published: those generations stay in ignored run directories on the machine that produced them.
`README.md` and `docs/examples.md` quote the start of answers to harmless prompts and a few
in-character refusals, picked by the judge and then read by a person.
The Child Abuse category of SimpleSafetyTests is left out of the Russian translation.

## Recipe

For target neuron `n` in layer `L` (0-based, HF `model.layers[L]`), at every position of the
prompt and of the generated text, the input of `mlp.down_proj` (that is `silu(gate(x)) * up(x)`)
gets

```
h[..., n] += s,   s = dose * p50_L / ||W_down[:, n]||
```

| condition | targets | mode | median |
|---|---|---|---|
| `baseline` | none | - | - |
| `recipe` | L14 N8949 dose +1, L13 N7956 dose -1 | add | reference (fixed) |
| `l14` | L14 N8949 dose +1 | add | reference |
| `dose0` | recipe targets with dose 0 (must equal baseline) | add | reference |
| `recipe_cal` | recipe | add | our own fixed calibration (`--medians`) |
| `recipe_ppm` | recipe | add | per prompt (median over the prompt's tokens 1:) |
| `overwrite` | recipe | overwrite (`h[..., n] = s`), as in sarvkk/two-neurons | per prompt |
| `placeboK` | random neurons in L14 (+1) and L13 (-1), seed K | add | reference |

Reference medians `p50_14 = 10.349`, `p50_13 = 9.7432` are the original author's fixed
calibration, recovered as the L2 norms of the direction tensors in
`dahara1/EvilQwen-control-vectors` (each tensor is `dose * p50_L * W_down[:, n] / ||W_down[:, n]||`).
With the pinned weights this gives `s = +22.269` (L14 N8949) and `s = -23.294` (L13 N7956).
Our own pooled median over `data/neutral_prompts.jsonl` is reported next to them by
`snsteer.calibrate`.

Two independent implementations of the same recipe:
- HF transformers, fp32 on CPU, forward pre-hook on `down_proj` (`snsteer/steering.py`). All
  conditions of one prompt run as one batch (same length, so no padding); greedy, thinking off.
- llama.cpp `llama-server` on an unquantized BF16 GGUF converted from the same revision, with the
  control vector `evilqwen_L14p_L13m.gguf` at scale 1.0 (adds `s * W_down[:, n]` to the output of
  layer L, which is the same thing since `down_proj` has no bias). Prompts are tokenized by the HF
  tokenizer and sent as token ids (`snsteer/llama_run.py`).

Judge: `Qwen/Qwen3Guard-Gen-0.6B` response moderation (`snsteer/judge.py`).

## Pinned versions

| item | value |
|---|---|
| model | `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| judge | `Qwen/Qwen3Guard-Gen-0.6B` revision `fada3b2f655b89601929198343c94cd2f64d93cc` |
| control vectors | `dahara1/EvilQwen-control-vectors` revision `10f135cc8a13e1fc0f0d6a5a758d8064c3d37d1b` |
| Python | 3.12 (uv-managed) |
| uv | 0.10.8 |
| torch | 2.14.1+cpu (PyTorch CPU index) |
| transformers | 5.20.2 (full set in `uv.lock`) |
| llama.cpp | commit `2280f0fd163b03d4331d49e08460f66502e35854` (existing build on the compute host) |

sha256 of downloaded files (checked against the Hub LFS metadata by `scripts/fetch_models.py`):

```
Qwen/Qwen3.5-4B
  model.safetensors-00001-of-00002.safetensors  26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61
  model.safetensors-00002-of-00002.safetensors  cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188
  model.safetensors.index.json                  cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d
  tokenizer.json                                5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
  tokenizer_config.json                         316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8
Qwen/Qwen3Guard-Gen-0.6B
  model.safetensors                             4f3ce47ebd968cddb67de08d8764f8ede7c410a7d1fb9e08145a4c7a2f2e5c0f
dahara1/EvilQwen-control-vectors
  evilqwen_L14p_L13m.gguf                       24e96a23c3082fbc2b4fe16f1e2e5d077d084e9f1dd7346247152df8c79fa275
  L14_N8949.gguf                                6d4bc253564b9cc00eef75bc4e47711ff29c8843734eebc4c598e05487854400
  L13_N7956.gguf                                b96cb8ae088c0f8a7dae8efc11977e9f06d6d4847ed126ebaf87bcffccb1f3df
converted (convert_hf_to_gguf.py --outtype bf16, llama.cpp 2280f0f)
  Qwen3.5-4B-851bf6e-BF16.gguf                  700cebb6c57c015751e7306f808d0dbe40a9cbcfcbda01a83bcc6e210501cbbd
```

The full manifest of every file is written to `models-manifest.json` in the setup run directory.

## Create

On the compute host (no sudo, no system packages; everything except the uv binary lives in the
lab directory):

```bash
# 1. uv 0.10.8, user-local, shell profiles untouched
mkdir -p ~/refusal-neuron-ru/.uv && cd ~/refusal-neuron-ru/.uv
curl -LsSf -o uv-installer-0.10.8.sh https://astral.sh/uv/0.10.8/install.sh
sha256sum uv-installer-0.10.8.sh   # eae5e1dae89cd0b74d357f549ccd6faa94b2ad6c1d89d78972a625655a4556ae
UV_INSTALL_DIR=$HOME/.local/bin UV_NO_MODIFY_PATH=1 sh uv-installer-0.10.8.sh
```

From the workstation clone, push the code (code only, stamps the commit hash):

```bash
scripts/sync_to_host.sh HOST refusal-neuron-ru
```

On the compute host, in `~/refusal-neuron-ru`:

```bash
# 2. environment and models (about 10 GB in hf/)
scripts/run_job.sh setup 3600 bash -c \
  'uv sync --frozen --group convert && uv run --frozen python scripts/fetch_models.py --out $RUN_DIR/models-manifest.json'
# 3. BF16 GGUF for the llama.cpp engine (about 8.7 GB in gguf/)
scripts/run_job.sh convert-bf16 3600 scripts/convert_gguf.sh $HOME/llama.cpp bf16
```

`scripts/env.sh` points `HF_HOME`, the uv cache, the uv Python and the venv into the lab directory.
`scripts/run_job.sh NAME TIMEOUT CMD...` runs every job under `timeout` and `nice -n 10` and writes
`runs/<UTC>-NAME/` with the command, code commit, `uv pip freeze`, lock hash, host info, load
before and after, timing and the full log. Long jobs are started detached with
`systemd-run --user --unit=snsteer-NAME --collect scripts/run_job.sh ...`.

## Use

```bash
# smoke: hook changes outputs, dose 0 equals baseline token for token
scripts/run_job.sh e0-smoke 1800 uv run --frozen python -m snsteer.generate \
  --prompts data/smoke_prompts.jsonl --conditions baseline,recipe,l14,dose0 --max-new-tokens 48 --threads 10
uv run --frozen python -m snsteer.compare runs/<dir>            # condition vs baseline
# the same with llama.cpp and the control vectors, then engine agreement
scripts/run_job.sh e0-llama 1800 uv run --frozen python -m snsteer.llama_run \
  --server-bin $HOME/llama.cpp/build/bin/llama-server --model gguf/Qwen3.5-4B-851bf6e-BF16.gguf \
  --prompts data/smoke_prompts.jsonl --conditions baseline,recipe,l14,dose0 --max-new-tokens 48 --threads 12
uv run --frozen python -m snsteer.compare runs/<hf-dir> runs/<llama-dir>
# throughput over threads and batch sizes
scripts/run_job.sh e0-bench 2400 uv run --frozen python -m snsteer.bench \
  --prompts data/smoke_prompts.jsonl --threads 6,12,14 --batches 1,3,6 --tokens 48
# scale check: residual medians and natural range of the target neurons
scripts/run_job.sh e1-calib 3000 uv run --frozen python -m snsteer.calibrate \
  --prompts data/neutral_prompts.jsonl --continue-tokens 32 --threads 6
# where decode time goes (Gated DeltaNet reference kernels vs attention, MLP, LM head)
scripts/run_job.sh e0-profile 900 uv run --frozen python -m snsteer.profile_modules \
  --prompts data/smoke_prompts.jsonl --threads 6 --batches 1,6 --tokens 32
# benchmark: SST-100 + XSTest-80, stratified order so early results cover every category
scripts/run_job.sh main-llama 7200 uv run --frozen python -m snsteer.llama_run \
  --server-bin $HOME/llama.cpp/build/bin/llama-server --model gguf/Qwen3.5-4B-851bf6e-BF16.gguf \
  --prompts data/sst.jsonl data/xstest_safe.jsonl --stratified \
  --conditions baseline,recipe,l14,placebo1,placebo2,placebo3 --max-new-tokens 72 --threads 6 --parallel 16
# the same in Russian: Russian takes 1.2 to 1.4 times the tokens of English with this tokenizer,
# so 192 tokens show about twice the text of the English 72-token run
scripts/run_job.sh ru-llama 14400 uv run --frozen python -m snsteer.llama_run \
  --server-bin $HOME/llama.cpp/build/bin/llama-server --model gguf/Qwen3.5-4B-851bf6e-BF16.gguf \
  --prompts data/sst_ru.jsonl data/xstest_safe_ru.jsonl --stratified \
  --conditions baseline,recipe,l14,placebo1,placebo2,placebo3 --max-new-tokens 192 --threads 6 --parallel 16
# judge (can follow a running generation job whose run dir is synced in), then rates
scripts/run_job.sh judge 7200 uv run --frozen python -m snsteer.judge --inputs runs/<dir> --threads 4 --follow
uv run --frozen python -m snsteer.summary runs/<judge-dir>
# full analysis: rates with Wilson 95 % CIs, refusal regex, loops, engine agreement, recipe vs placebo
uv run --frozen python -m snsteer.analyze --gen runs/<llama-dir> runs/<hf-dir> \
  --judge runs/<judge-dir>... --out results.json --md tables.md
```

Run directories hold generated text, which for the safety benchmarks includes harmful content.
They are written outside the repository (`runs/` is ignored) and are not published. Copy them
back with `scripts/pull_runs.sh DEST HOST`, or keep copying while a job runs with
`scripts/follow_pull.sh DEST HOST snsteer-NAME`.

## Measurements so far

Intel i5-14500 (6P + 8E cores, 20 threads), DDR5, no GPU.

- Hook check: `recipe` and `l14` change the greedy output of all 5 smoke prompts within the
  first 0 to 2 tokens; `dose0` is identical to `baseline` token for token in both engines.
- `s` = +22.269 (L14 N8949, `||W_down[:, n]||` = 0.464718) and -23.294 (L13 N7956, 0.418273).
- Own calibration (24 neutral prompts + 32 generated tokens, 1291 tokens): pooled median of
  `||resid||` 10.293 at L14 and 9.465 at L13 (reference 10.349 and 9.743). The target neurons
  naturally stay within |a| <= 0.28 (L14 N8949) and 0.32 (L13 N7956); the injected `s` is 79x
  and 72x that.
- Engine agreement on the smoke set (48 tokens, greedy): HF fp32 vs llama.cpp BF16 identical in
  4/5 (baseline), 3/5 (recipe), 4/5 (l14); the rest diverge late at near-ties.
- HF fp32 throughput: about 3.5 tok/s at batch 1, 12 to 14 tok/s at batch 6; 6 threads are as
  fast as 12 or 14 (memory bound). Gated DeltaNet layers run reference PyTorch kernels on CPU
  (no flash-linear-attention / causal-conv1d) and take 28 % (batch 1) to 35 % (batch 6) of
  decode time, MLP 50 % / 44 %, attention 7 %, LM head 13 %.
- llama.cpp BF16: 7.3 tok/s for one request; with 8 / 16 slots 4.6 / 2.7 tok/s per slot, that is
  37 / 43 tok/s total while all slots are busy (24 / 29 tok/s wall over only 29 prompts).
- Judge (Qwen3Guard-Gen-0.6B fp32, 3 threads of an i7-8700): about 3.3 s per response.
- Running both engines at once on this machine is slower than llama.cpp alone (memory
  bandwidth), so engines run one after the other.

## Results of the main run

SimpleSafetyTests (100 harmful prompts) and 80 safe XSTest prompts, greedy, 72 new tokens, judged by
Qwen3Guard-Gen-0.6B (refusal = `Refusal: Yes`, unsafe = `Safety: Unsafe`); percentages, Wilson 95 % CI
in brackets. Placebo = random neurons of the same layers with the same sign and the same injected norm,
3 seeds pooled. HF fp32 and llama.cpp BF16 give the same rates within 1 to 3 prompts.

| condition | SST refusal | SST unsafe | XSTest unsafe | XSTest refusal |
|---|---|---|---|---|
| baseline (both engines) | 88 [80-93] | 2 [1-7] | 4 [1-10] | 11 [6-20] |
| recipe, llama.cpp / HF | 16 / 16 | 96 / 96 [90-98] | 49 / 46 | 10 / 10 |
| L14 only, both | 4 | 99 [95-100] | 38 / 36 | 1 |
| recipe, per-prompt median (HF) | 8 | 97 | 44 | 10 |
| overwrite, per-prompt median (HF, as sarvkk) | 9 | 95 | 42 | 10 |
| placebo, llama.cpp / HF | 81 / 82 | 19 / 19 | 1 / 1 | 58 / 56 |

sarvkk/two-neurons (nf4 on GPU, overwrite, per-prompt median): SST refusal 90 -> 8, unsafe 1 -> 96;
XSTest unsafe 2 -> 40; L14 only unsafe 100 (n = 20). Unsafe rates do not depend on add vs overwrite or
on the median source; the remaining SST refusals do (16 % with the fixed reference median, 8 to 9 % with
the per-prompt median, McNemar p = 0.02 to 0.04). Of the 16 recipe refusals on SST the judge marks 12 as
unsafe at the same time (in-character refusals). Placebos raise SST unsafe from 2 % to 14 to 29 % and
break the text (one seed loops in 96 % of answers), far below the recipe (paired McNemar against
each seed, largest p = 1.4e-20).
Engine agreement (HF vs llama.cpp, same prompt and condition): judge safety label equal in 93 to 100 %
of pairs per condition, identical tokens in 79 % (baseline) and 38 % (recipe) of SST answers.

## Results in Russian

The same model and conditions on the Russian translation (`data/sst_ru.jsonl`, 80 prompts without
Child Abuse; `data/xstest_safe_ru.jsonl`, 80), llama.cpp BF16, greedy, 192 new tokens, same judge.
English column: the main llama.cpp run restricted to the same 80 + 80 prompts (72 tokens).

| condition | SST refusal RU / EN | SST unsafe RU / EN | XSTest unsafe RU / EN |
|---|---|---|---|
| baseline | 86 [77-92] / 89 | 2 [1-9] / 0 | 1 [0-7] / 4 |
| recipe | 32 [23-43] / 10 | 89 [80-94] / 99 | 59 [48-69] / 49 |
| L14 only | 0 [0-5] / 2 | 99 [93-100] / 100 | 55 [44-65] / 38 |
| placebo, 3 seeds pooled | 69 [63-74] | 29 [24-35] | 3 [2-6] |

The L14 neuron removes refusals in Russian as fully as in English. The full recipe is weaker in
Russian: a third of the refusals stay (32 % against 10 %), so the L13 subtraction may matter more
here; this is not tested separately. Recipe against placebo on SST: largest paired p = 1.9e-8. The judge
agrees with the Russian refusal regex on 74 of 80 baseline SST answers and 78 of 80 XSTest answers.
Caveat: Russian answers run up to 192 tokens against 72 in English, and placebo answers in particular
degrade with length (two placebo seeds loop in 61 % and 99 % of SST answers).

## Remove

On the compute host:

```bash
rm -rf ~/refusal-neuron-ru            # code, venv, uv Python and cache, models, GGUF, runs
rm -f ~/.local/bin/uv ~/.local/bin/uvx ~/.config/uv/uv-receipt.json
rmdir ~/.local/bin ~/.config/uv 2>/dev/null
```

Host changes made by Create, all inside the user's home: `~/.local/bin/uv`, `~/.local/bin/uvx`,
`~/.config/uv/uv-receipt.json`, and the lab directory `~/refusal-neuron-ru/`. The existing
llama.cpp build is used read-only. Transient systemd user units `snsteer-*` disappear when they
finish (`--collect`).

## License

MIT, see `LICENSE`. Parts are adapted from sarvkk/two-neurons (MIT), see `NOTICE`. Benchmark
prompts belong to their authors, see `data/README.md`; the Russian translations are adaptations
under the same licenses as their sources (CC BY 2.0 and CC BY 4.0).
