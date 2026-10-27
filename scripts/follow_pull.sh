#!/bin/bash
# Usage (workstation): scripts/follow_pull.sh DEST HOST UNIT [INTERVAL_SEC] [DIR]
# Pulls run directories every INTERVAL while the systemd user UNIT on HOST is active, then once
# more, so a judge with --follow on DEST sees generations as they arrive.
set -u
dest="$1"; host="$2"; unit="$3"; every="${4:-120}"; dir="${5:-refusal-neuron-ru}"
here="$(cd "$(dirname "$0")" && pwd)"
while timeout 20 ssh "$host" systemctl --user is-active -q "$unit"; do
  "$here/pull_runs.sh" "$dest" "$host" "$dir" || true
  sleep "$every"
done
"$here/pull_runs.sh" "$dest" "$host" "$dir"
echo "unit $unit finished, last pull done"
