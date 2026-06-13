# coding=utf-8
"""
检验手眼标定采集数据质量：检测棋盘内角点并回写到图片，便于人工核对。

用法:
  python src/camera/calibration/check_data.py
  python src/camera/calibration/check_data.py --data-dir camera_data/eye_hand_data/calibration_data20260602
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from camera_sdk.realsense.libs.log_setting import CommonLog

DEFAULT_DATA_DIR = os.path.join(
    _PROJECT_ROOT, "camera_data", "eye_hand_data", "calibration_data20260602"
)
DEFAULT_OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "camera_data", "calibration_check")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "calibration_config.yaml")
CORNER_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def load_checkerboard_size(config_path: str = CONFIG_PATH) -> tuple[int, int]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    args = data.get("checkerboard_args") or {}
    return int(args["XX"]), int(args["YY"])


def iter_calibration_images(data_dir: str) -> list[str]:
    """按 0.jpg, 1.jpg, ... 顺序收集标定图片路径。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"标定数据目录不存在: {data_dir}")

    jpg_count = len([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    image_files = []
    for i in range(jpg_count + 1):
        image_file = os.path.join(data_dir, f"{i}.jpg")
        if os.path.exists(image_file):
            image_files.append(image_file)
    return image_files


def check_chessboard_corners(
    data_dir: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    config_path: str = CONFIG_PATH,
    logger: CommonLog | None = None,
) -> dict:
    """
    检测 data_dir 内标定图片的棋盘角点，将标注结果保存到 output_root/<会话名>/。

    返回统计信息字典。
    """
    if logger is None:
        logger = CommonLog(logging.getLogger(__name__))

    xx, yy = load_checkerboard_size(config_path)
    board_size = (xx, yy)
    session_name = os.path.basename(os.path.normpath(data_dir))
    output_dir = os.path.join(output_root, session_name)
    os.makedirs(output_dir, exist_ok=True)

    image_files = iter_calibration_images(data_dir)
    if not image_files:
        raise RuntimeError(f"目录中未找到标定图片: {data_dir}")

    results = []
    found_count = 0
    missing_count = 0

    for image_file in image_files:
        basename = os.path.basename(image_file)
        stem = os.path.splitext(basename)[0]
        logger.info(f"角点检查: 读 {image_file}")

        img = cv2.imread(image_file)
        if img is None:
            logger.warning(f"无法读取图片: {image_file}")
            missing_count += 1
            results.append(
                {
                    "image": basename,
                    "found": False,
                    "error": "read_failed",
                    "output": None,
                }
            )
            continue

        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), CORNER_CRITERIA)
            cv2.drawChessboardCorners(vis, board_size, corners2, ret)
            status_text = "CORNER OK"
            status_color = (0, 220, 0)
            found_count += 1
        else:
            status_text = "CORNER NOT FOUND"
            status_color = (0, 0, 255)
            missing_count += 1

        cv2.putText(
            vis,
            f"{basename} | {status_text} | board={board_size}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        save_path = os.path.join(output_dir, f"{stem}_corners.jpg")
        cv2.imwrite(save_path, vis)
        logger.info(f"角点检查图已保存: {save_path}")

        results.append(
            {
                "image": basename,
                "found": bool(ret),
                "output": os.path.basename(save_path),
            }
        )

    summary = {
        "data_dir": os.path.abspath(data_dir),
        "output_dir": os.path.abspath(output_dir),
        "board_size": {"XX": xx, "YY": yy},
        "total": len(image_files),
        "found": found_count,
        "missing": missing_count,
        "pass_rate": round(found_count / len(image_files), 4) if image_files else 0.0,
        "details": results,
    }

    summary_path = os.path.join(output_dir, "check_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"角点检查完成: 共 {summary['total']} 张, "
        f"成功 {found_count} 张, 失败 {missing_count} 张, "
        f"通过率 {summary['pass_rate']:.1%}"
    )
    logger.info(f"检查摘要已写入: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检验手眼标定采集数据的棋盘角点质量")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="待检查的标定图片目录",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help="角点标注图片输出根目录",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="棋盘格参数配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = CommonLog(logging.getLogger(__name__))
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_root = os.path.abspath(args.output_dir)

    summary = check_chessboard_corners(
        data_dir=data_dir,
        output_root=output_root,
        config_path=args.config,
        logger=logger,
    )

    print("\n=== 角点检验结果 ===")
    print(f"数据目录: {summary['data_dir']}")
    print(f"输出目录: {summary['output_dir']}")
    print(f"棋盘尺寸 (XX, YY): ({summary['board_size']['XX']}, {summary['board_size']['YY']})")
    print(f"总计: {summary['total']} 张 | 成功: {summary['found']} 张 | 失败: {summary['missing']} 张")
    print(f"通过率: {summary['pass_rate']:.1%}")

    if summary["missing"] > 0:
        failed = [item["image"] for item in summary["details"] if not item["found"]]
        print(f"未检测到角点的图片: {', '.join(failed)}")


if __name__ == "__main__":
    main()
