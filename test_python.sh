#!/usr/bin/env bash
set -e
source .venv/bin/activate
rm -f python/openai_harmony/openai_harmony.cpython-*.so
maturin develop --release
pytest "$@"
