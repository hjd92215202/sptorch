#!/usr/bin/env bash
set -euo pipefail

ORDER=(
  sptorch-core-tensor
  sptorch-data
  sptorch-versioning
  sptorch-optim
  sptorch-serialize
  sptorch-hal
  sptorch-core-autograd
  sptorch-runtime-cuda
  sptorch-core-ops
  sptorch-nn
  sptorch-hal-ffi
  sptorch-distributed
  sptorch-live-evolution
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
