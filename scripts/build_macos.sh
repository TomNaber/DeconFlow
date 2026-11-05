#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/build_deliverable.py --platform macos "$@"
