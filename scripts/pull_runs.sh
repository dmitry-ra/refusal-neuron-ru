#!/bin/bash
# Usage (workstation): scripts/pull_runs.sh DEST HOST [DIR]
# Copies run directories (logs, generations, judge labels) from the compute host to DEST.
set -eu
dest="$1"; host="$2"; dir="${3:-refusal-neuron-ru}"
mkdir -p "$dest"
rsync -a "$host:$dir/runs/" "$dest/"
