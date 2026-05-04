$ErrorActionPreference = "Stop"

$order = @(
  "core-tensor",
  "data",
  "versioning",
  "optim",
  "serialize",
  "hal",
  "core-autograd",
  "runtime-cuda",
  "core-ops",
  "nn",
  "hal-ffi",
  "distributed",
  "live-evolution",
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
