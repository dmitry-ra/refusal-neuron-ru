#!/bin/bash
# Usage (on the workstation, from the clone): scripts/sync_to_host.sh HOST [DIR]
# Pushes the working tree (code only) to the compute host and stamps the commit hash.
# HOST "local" syncs into a local DIR.
set -eu
host="$1"; dir="${2:-refusal-neuron-ru}"
cd "$(dirname "$0")/.."
rev="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain)" ] || rev="$rev-dirty"
echo "$rev" > COMMIT
if [ "$host" = local ]; then dest="$dir/"; else dest="$host:$dir/"; fi
rsync -a --exclude .git --exclude .venv --exclude .uv --exclude hf --exclude gguf \
  --exclude runs --exclude __pycache__ ./ "$dest"
rm -f COMMIT
echo "synced $rev to $host:$dir"
