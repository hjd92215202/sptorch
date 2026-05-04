#!/usr/bin/env bash
set -euo pipefail

ORDER=(
  core-tensor
  data
  versioning
  optim
  serialize
  hal
  core-autograd
  runtime-cuda
  core-ops
  nn
  hal-ffi
  distributed
  live-evolution
  sptorch
)

echo "== SPTorch publish rehearsal =="
for c in "${ORDER[@]}"; do
  echo "\n=== $c ==="
  if cargo package -p "$c" --allow-dirty --no-verify; then
    echo "STATUS: OK"
  else
    echo "STATUS: FAIL"
  fi
done
