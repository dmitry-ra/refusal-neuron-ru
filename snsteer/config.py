"""Pinned models and the steering recipe."""

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

GUARD_ID = "Qwen/Qwen3Guard-Gen-0.6B"
GUARD_REVISION = "fada3b2f655b89601929198343c94cd2f64d93cc"

CVEC_ID = "dahara1/EvilQwen-control-vectors"
CVEC_REVISION = "10f135cc8a13e1fc0f0d6a5a758d8064c3d37d1b"
CVEC_FILES = ["evilqwen_L14p_L13m.gguf", "L14_N8949.gguf", "L13_N7956.gguf"]

N_LAYERS = 32
D_MLP = 9216

# Median ||resid|| leaving layer L (0-based) from the original author's calibration,
# recovered as the L2 norms of the direction tensors in CVEC_FILES (dose 1).
REFERENCE_MEDIANS = {14: 10.349, 13: 9.7432}

# (layer, neuron, dose); layers are 0-based like HF model.layers[L].
RECIPE = [(14, 8949, +1.0), (13, 7956, -1.0)]
L14_ONLY = [(14, 8949, +1.0)]
