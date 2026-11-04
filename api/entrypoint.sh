#!/bin/sh
# Fresh multiprocess metrics directory on every start (stale files would double-count).
set -e
rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
exec "$@"
