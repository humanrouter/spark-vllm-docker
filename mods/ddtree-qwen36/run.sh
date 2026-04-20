#!/bin/bash
set -euo pipefail

echo "[ddtree-qwen36] Applying experimental DFlash+DDTree prototype patches"
python3 "$(dirname "$0")/patch_ddtree_qwen36.py"
