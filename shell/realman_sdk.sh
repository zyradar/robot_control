#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
cd ../ros2_ws
source install/setup.bash

ros2 launch rm_driver rm_65_driver.launch.py
