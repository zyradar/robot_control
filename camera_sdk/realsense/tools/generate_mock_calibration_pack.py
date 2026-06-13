# coding=utf-8
"""
生成可供 compute_in_hand.py 冒烟测试用的最小数据集：
- eye_hand_data/dataYYYYMMDD/poses.txt
- 同目录 1.jpg .. N.jpg（合成棋盘图，可被 findChessboardCorners 检出）

用法（在项目根目录）:
  python tools/generate_mock_calibration_pack.py
"""
from __future__ import annotations

import os
import random
from datetime import datetime

import cv2
import numpy as np

# 与 config.yaml 一致
XX, YY = 11, 8
NUM_VIEWS = 15
IMG_W, IMG_H = 640, 480


def make_flat_board_texture() -> np.ndarray:
    """12x9 方格，对应 11x8 内角点。"""
    cols, rows = XX + 1, YY + 1
    cell = 48
    h, w = rows * cell, cols * cell
    img = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = 0
    return img


def warp_board_view(base_small: np.ndarray, rng: random.Random, idx: int) -> np.ndarray:
    """仿射旋转/缩放/平移（不用强透视），便于 OpenCV 检出棋盘内角点。"""
    h0, w0 = base_small.shape[:2]
    ang = float(rng.uniform(-38, 38) + (idx % 9) * 3.5)
    scale = float(rng.uniform(0.82, 1.22))
    cx, cy = w0 / 2.0, h0 / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, scale)
    M[0, 2] += rng.uniform(-55, 55)
    M[1, 2] += rng.uniform(-45, 45)
    warped = cv2.warpAffine(
        base_small,
        M,
        (IMG_W, IMG_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped


def synthetic_poses(n: int, rng: random.Random) -> list[list[float]]:
    """末端相对基坐标的合成位姿（米 + 弧度），旋转分量足够大以利 Tsai。"""
    poses: list[list[float]] = []
    for i in range(n):
        u = i / max(n - 1, 1)
        x = 0.35 + 0.15 * rng.uniform(-1, 1)
        y = 0.05 * rng.uniform(-1, 1) + 0.08 * np.sin(u * np.pi)
        z = 0.42 + 0.06 * rng.uniform(-1, 1)
        rx = rng.uniform(-0.55, 0.55) + 0.25 * np.sin(u * 2 * np.pi)
        ry = rng.uniform(-0.45, 0.45) + 0.35 * np.cos(u * 1.7 * np.pi)
        rz = rng.uniform(-0.9, 0.9) + 0.4 * u
        poses.append([x, y, z, float(rx), float(ry), float(rz)])
    return poses


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    today = datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(root, "eye_hand_data", f"data{today}")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(42)
    texture = make_flat_board_texture()
    base = cv2.resize(texture, (560, 420))

    ok_count = 0
    attempts = 0
    while ok_count < NUM_VIEWS and attempts < NUM_VIEWS * 40:
        attempts += 1
        gray = warp_board_view(base, rng, ok_count)
        ret, _ = cv2.findChessboardCorners(
            gray,
            (XX, YY),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ret:
            continue
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        path = os.path.join(out_dir, f"{ok_count + 1}.jpg")
        cv2.imwrite(path, bgr)
        ok_count += 1

    if ok_count < NUM_VIEWS:
        raise RuntimeError(f"未能生成足够可检测棋盘图，仅 {ok_count}/{NUM_VIEWS}")

    poses = synthetic_poses(ok_count, rng)
    poses_path = os.path.join(out_dir, "poses.txt")
    with open(poses_path, "w", encoding="utf-8") as f:
        for p in poses:
            f.write(",".join(f"{v:.9f}" for v in p) + "\n")

    print(f"已写入: {poses_path}（{ok_count} 行）")
    print(f"已写入: {out_dir}/1.jpg .. {ok_count}.jpg")


if __name__ == "__main__":
    main()
