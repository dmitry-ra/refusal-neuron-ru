#!/bin/bash
# Usage (on the compute host, from the lab dir): scripts/run_job.sh NAME TIMEOUT_SEC CMD [ARGS...]
# Creates runs/<UTC>-NAME/ with the command, code revision, frozen deps, host info, timing and
# the full stdout/stderr log; CMD sees the run dir as $RUN_DIR. SNSTEER_RUNS overrides runs/.
set -u
name="$1"; tmo="$2"; shift 2
cd "$(dirname "$0")/.."
source scripts/env.sh
ts="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${SNSTEER_RUNS:-$LAB_DIR/runs}/$ts-$name"
export RUN_DIR
mkdir -p "$RUN_DIR"
printf '%q ' "$@" > "$RUN_DIR/cmd.txt"; echo >> "$RUN_DIR/cmd.txt"
echo "timeout=$tmo nice=10 pid=$$ unit=${SYSTEMD_UNIT_NAME:-}" >> "$RUN_DIR/cmd.txt"
if [ -f COMMIT ]; then cp COMMIT "$RUN_DIR/commit.txt"
else git describe --always --dirty > "$RUN_DIR/commit.txt" 2>/dev/null || echo unknown > "$RUN_DIR/commit.txt"; fi
sha256sum uv.lock > "$RUN_DIR/lock.sha256"
uv pip freeze > "$RUN_DIR/pip-freeze.txt" 2>/dev/null
{
  echo "host: $(hostname)"
  echo "kernel: $(uname -r)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'
  free -g | head -2
  echo "load_before: $(cat /proc/loadavg)"
} > "$RUN_DIR/host.txt"
start=$(date +%s.%N)
timeout "$tmo" nice -n 10 "$@" > "$RUN_DIR/log.txt" 2>&1 < /dev/null
rc=$?
end=$(date +%s.%N)
echo "load_after: $(cat /proc/loadavg)" >> "$RUN_DIR/host.txt"
printf '{"start_epoch": %s, "end_epoch": %s, "exit_code": %d}\n' "$start" "$end" "$rc" > "$RUN_DIR/timing.json"
echo "$RUN_DIR rc=$rc"
exit $rc
