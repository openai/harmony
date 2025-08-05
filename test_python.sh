#!/usr/bin/env bash
set -e
source .venv/bin/activate
maturin develop -F python-binding --release 
pytest "$@"
