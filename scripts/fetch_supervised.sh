#!/usr/bin/env bash
# Run the resumable fetcher until it exits cleanly; restart after crashes
# (e.g. transient out-of-memory on a shared machine). Existing chunks are skipped.
set -u
BIN=${BIN:-target/release/slotfish}
while true; do
  "$BIN" fetch "$@" && break
  echo "[supervisor] fetcher exited with $?; restarting in 5s" >&2
  sleep 5
done
