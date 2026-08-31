#!/usr/bin/env bash
set -euo pipefail
DEVICE="${1:-auto}"

python -m audio_affect_probe.train --config configs/baseline_mlp.yaml --device "$DEVICE" --zip-results
python -m audio_affect_probe.train --config configs/slow.yaml         --device "$DEVICE" --zip-results
python -m audio_affect_probe.train --config configs/dual.yaml         --device "$DEVICE" --zip-results
python -m audio_affect_probe.train --config configs/salience.yaml     --device "$DEVICE" --zip-results
