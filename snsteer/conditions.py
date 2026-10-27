"""Named experimental conditions.

Median sources:
  reference  - config.REFERENCE_MEDIANS (the original author's fixed calibration)
  calibrated - fixed medians from our own neutral-text calibration (--medians file)
  prompt     - median over the prompt's own tokens 1: on a clean forward (sarvkk/two-neurons)
"""

import random

from . import config
from .steering import Push, dose_value

PLACEBO_LAYERS = [(14, +1.0), (13, -1.0)]


def placebo_targets(seed):
    rng = random.Random(seed)
    taken = {(L, n) for L, n, _ in config.RECIPE}
    out = []
    for L, dose in PLACEBO_LAYERS:
        while True:
            n = rng.randrange(config.D_MLP)
            if (L, n) not in taken:
                break
        out.append((L, n, dose))
    return out


def spec(name):
    """Returns (targets, mode, median_source)."""
    fixed = {
        "baseline": ([], "add", "reference"),
        "recipe": (config.RECIPE, "add", "reference"),
        "recipe_cal": (config.RECIPE, "add", "calibrated"),
        "recipe_ppm": (config.RECIPE, "add", "prompt"),
        "l14": (config.L14_ONLY, "add", "reference"),
        "dose0": ([(L, n, 0.0) for L, n, _ in config.RECIPE], "add", "reference"),
        "overwrite": (config.RECIPE, "overwrite", "prompt"),
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("placebo"):
        return placebo_targets(int(name[len("placebo"):])), "add", "reference"
    raise ValueError(f"unknown condition {name}")


def needs_prompt_median(names):
    return any(spec(n)[2] == "prompt" for n in names)


def pushes(name, colnorm, medians):
    """colnorm(L, n) -> float; medians: {source: {layer: float}}."""
    targets, mode, src = spec(name)
    out, detail = [], []
    for L, n, dose in targets:
        med = medians[src][L]
        cn = colnorm(L, n)
        s = dose_value(dose, med, cn)
        out.append(Push(L, n, s, mode))
        detail.append({"layer": L, "neuron": n, "dose": dose, "mode": mode, "median_source": src,
                       "median": med, "colnorm": cn, "value": s})
    return out, detail
