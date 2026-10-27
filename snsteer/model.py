"""Model loading and prompt rendering."""

import time

import torch

from . import config


def load(threads, dtype="float32"):
    import transformers
    from transformers import AutoTokenizer

    torch.set_num_threads(threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    # The checkpoint is the vision-language class; text-only generation goes through it unchanged.
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration")
    model = cls.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION,
                                dtype=getattr(torch, dtype)).eval()
    return tok, model, time.time() - t0


def render(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


def encode(tok, text):
    ids = tok(render(tok, text), return_tensors="pt", add_special_tokens=False).input_ids
    return ids
