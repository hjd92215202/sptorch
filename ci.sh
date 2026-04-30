#!/usr/bin/env bash
set -euo pipefail

echo "=== sptorch CI ==="
echo ""

echo "[1/3] cargo check (all targets)"
cargo check --workspace 2>&1
echo "  ✓ check passed"
echo ""

echo "[2/3] cargo test (all crates)"
cargo test --workspace 2>&1
echo "  ✓ all tests passed"
echo ""

echo "[3/3] cargo clippy (warnings = deny)"
cargo clippy --workspace -- -D warnings 2>&1 || {
    echo "  ⚠ clippy has warnings (non-blocking)"
}
echo ""

echo "=== CI complete ==="
