#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

source install/setup.bash

python src/camera/calibration/automatic_calibration.py
