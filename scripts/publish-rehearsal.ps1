$ErrorActionPreference = "Stop"

$order = @(
  "sptorch-core-tensor",
  "sptorch-data",
  "sptorch-versioning",
  "sptorch-optim",
  "sptorch-serialize",
  "sptorch-hal",
  "sptorch-core-autograd",
  "sptorch-runtime-cuda",
  "sptorch-core-ops",
  "sptorch-nn",
  "sptorch-hal-ffi",
  "sptorch-distributed",
  "sptorch-live-evolution",
  "sptorch"
)

Write-Output "== SPTorch publish rehearsal =="
foreach ($c in $order) {
  Write-Output "`n=== $c ==="
  try {
    cargo package -p $c --allow-dirty --no-verify | Out-Host
    Write-Output "STATUS: OK"
  }
  catch {
    Write-Output "STATUS: FAIL"
  }
}
