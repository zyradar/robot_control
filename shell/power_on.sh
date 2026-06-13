#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
cd ../ros2_ws
source install/setup.bash

ros2 launch rm_driver rm_65_driver.launch.py &

cd "$ROOT" || exit 1
source install/setup.bash

python src/camera/vision/pixel_from_mouse.py
# &表示后台运行
