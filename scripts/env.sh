# Source from the lab directory: keeps uv, Python, caches and models inside it.
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAB_DIR
export HF_HOME="$LAB_DIR/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR="$LAB_DIR/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$LAB_DIR/.uv/python"
export UV_PROJECT_ENVIRONMENT="$LAB_DIR/.venv"
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"
