#!/usr/bin/env bash
# End-to-end run on a scratch stack (project yp-test), then teardown.
#
#   tools/e2e.sh            # build, start, unit + integration tests, failover drills, down -v
#   KEEP=1 tools/e2e.sh     # leave the stack up afterwards
#
# Needs the TensorRT engines and golden files in inference/model_repository
# (inference/export/build_plan.sh + validation/parity.py --write-golden) and the
# test clips (tools/gen_substreams.sh). Uses the repo's .env for secrets.
set -euo pipefail
cd "$(dirname "$0")/.."
P=yp-test
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/yp-e2e.XXXXXX")"
cp -a inference/model_repository "$SCRATCH/model_repository"
export MODEL_REPO_DIR="$SCRATCH/model_repository"
export SERVING_DIR="$SCRATCH/serving"
mkdir -p "$SERVING_DIR"
DC=(docker compose -p "$P" -f docker-compose.yml -f docker-compose.test.yml)
set -a; . ./.env; set +a
NET="${P}_default"

cleanup() {
  if [ "${KEEP:-0}" != 1 ]; then
    "${DC[@]}" --profile testsrc down -v --remove-orphans >/dev/null 2>&1 || true
    rm -rf "$SCRATCH"
  else
    echo "stack $P left running; model repository copy in $SCRATCH"
  fi
}
trap cleanup EXIT

echo "== build and start"
"${DC[@]}" build -q api camera-shard
"${DC[@]}" --profile testsrc up -d --wait --wait-timeout 600 2>&1 | grep -vE "Running|Waiting|Healthy|Started|Created" || true

run_api() { docker run --rm --network "$NET" -e HOME=/tmp -e ADMIN_API_KEY -e TEST_REDIS_URL=redis://redis:6379/15 \
  -u 0 -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD:/src:ro" -v "$MODEL_REPO_DIR:/models:ro" -v "$PWD/videos:/videos:ro" -w /src --entrypoint sh \
  yolo-platform-api:${TAG:-dev} -c "pip install -q --user -r api/requirements-dev.txt 2>/dev/null; $*"; }
run_worker() { docker run --rm --gpus all --network "$NET" -e HOME=/tmp -e TEST_REDIS_URL=redis://redis:6379/15 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility -v "$PWD:/src:ro" -w /src --entrypoint sh \
  yolo-platform-worker:${TAG:-dev} -c "pip install -q --user -r workers/requirements-dev.txt 2>/dev/null; $*"; }

echo "== unit tests"
run_worker "python -m pytest -q -p no:cacheprovider tests/unit --ignore=tests/unit/test_api_units.py"
run_api "python -m pytest -q -p no:cacheprovider tests/unit/test_api_units.py tests/unit/test_shared.py"

echo "== integration tests (through nginx)"
run_api "E2E_BASE=https://nginx E2E_MODELS=/models E2E_VIDEOS=/videos python -m pytest -q -p no:cacheprovider tests/integration ${PYTEST_K:+-k \"$PYTEST_K\"}"

echo "== drill: kill a camera shard, its cameras move to the other one"
"${DC[@]}" up -d --no-deps --scale camera-shard=2 camera-shard >/dev/null 2>&1
run_api "python tools/drills.py shard-failover --project $P"
echo "== drill: kill the video worker mid-job, the job is reclaimed and completes"
run_api "python tools/drills.py job-reclaim --project $P"
echo "== drill: restart Triton, models reload and cameras resume"
run_api "python tools/drills.py triton-restart --project $P"
echo "e2e OK"
