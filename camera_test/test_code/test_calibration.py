# coding=utf-8
"""
眼在手上自动标定流程：结合 collect_data.py 的采集方式与 compute_in_hand.py 的计算。

采集（与 collect_data 一致）：
  - RealSense 彩色 640×480；显示与存盘均为 2 倍缩放图。
  - 仅当按下 s 时请求机械臂位姿；位姿成功则追加 poses.txt 并保存 {序号}.jpg（序号按成功次数递增）。

按键：
  s      保存一帧标定数据（与 collect_data 相同触发方式；界面仍会提示是否检测到棋盘角点）
  C / c  用当前会话目录已采集数据执行 compute_in_hand，将旋转矩阵、四元数 (x,y,z,w)、平移 x,y,z (m)
         写入 calibration_results/hand_eye_时间戳.yaml
  ESC    关闭全部窗口、停止 RealSense 流、关闭 socket 并退出（不写 YAML、不执行末尾标定）

用法：
  python test.py                      # 默认成功保存满 15 张后结束采集循环
  python test.py --target 12          # 成功保存满 12 张后结束
  python test.py --compute-only eye_hand_data/data20260512   # 仅用已有目录计算并打印（不写 YAML）
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from scipy.spatial.transform import Rotation as R

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from collect_data import send_cmd
from libs.auxiliary import create_folder_with_date, get_ip, popup_message
from libs.log_setting import CommonLog

import compute_in_hand

logger_ = CommonLog(logging.getLogger(__name__))

SCALING_FACTOR = 2.0
CALIBRATION_RESULTS_DIR = os.path.join(_ROOT, "calibration_results")


def _load_checkerboard_size():
    with open(os.path.join(_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    args = data.get("checkerboard_args") or {}
    return int(args["XX"]), int(args["YY"])


def _run_hand_eye_and_save_yaml(session_dir: str) -> str:
    """
    调用 compute_in_hand.func，将旋转矩阵、四元数 (scipy xyzw)、平移 x,y,z (m) 写入 YAML。
    返回生成的 YAML 绝对路径。
    """
    rotation_matrix, translation_vector = compute_in_hand.func(session_dir)
    quat = R.from_matrix(rotation_matrix).as_quat()
    x, y, z = (float(v) for v in translation_vector.flatten().tolist())
    qx, qy, qz, qw = (float(v) for v in quat.tolist())

    os.makedirs(CALIBRATION_RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(CALIBRATION_RESULTS_DIR, f"hand_eye_{stamp}.yaml")
    rel_data = os.path.relpath(os.path.abspath(session_dir), _ROOT).replace("\\", "/")

    payload = {
        "hand_eye": {
            "data_dir": rel_data,
            "rotation_matrix": np.asarray(rotation_matrix).tolist(),
            "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw},
            "translation_m": {"x": x, "y": y, "z": z},
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, default_flow_style=None)

    logger_.info(f"手眼结果已写入: {out_path}")
    logger_.info(f"旋转矩阵:\n{rotation_matrix}")
    logger_.info(f"平移 (m): x={x}, y={y}, z={z}")
    logger_.info(f"四元数 (x,y,z,w scipy 约定): {quat}")
    return out_path


def auto_collect(session_dir: str, client: socket.socket, target_count: int) -> tuple[int, str]:
    """
    仅当按下 s 时拉取位姿并保存（与 collect_data：socket、poses.txt、2× 缩放存图一致）。

    返回 (成功保存张数, 结束原因): ``target`` 为达到目标张数；``esc`` 为用户按 ESC。
    """
    xx, yy = _load_checkerboard_size()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(cfg)
    except Exception as e:
        logger_.error(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")
        raise SystemExit(1) from e

    count = 0
    cv2.namedWindow("Capture_Video", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            vis = cv2.resize(
                color_image, None, fx=SCALING_FACTOR, fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (xx, yy), None)

            k = cv2.waitKey(30) & 0xFF

            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                cv2.drawChessboardCorners(vis, (xx, yy), corners2, found)

            if k == ord("s"):
                socket_command = '{"command": "get_current_arm_state"}'
                state, pose = send_cmd(client, socket_command)
                logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else pose}')
                if not state:
                    logger_.warning(f"获取机械臂位姿失败：{pose}")
                else:
                    count += 1
                    pose_path = os.path.join(session_dir, "poses.txt")
                    with open(pose_path, "a+", encoding="utf-8") as f:
                        f.write(f'{",".join(str(i) for i in pose)}\n')
                    img_path = os.path.join(session_dir, f"{count}.jpg")
                    cv2.imwrite(img_path, vis)
                    logger_.info(f"===采集第{count}次数据！ -> {img_path}")

                    if count >= target_count:
                        logger_.info("已达到目标采集张数，结束采集循环。")
                        return count, "target"

            if k == ord("c") or k == ord("C"):
                if count < 3:
                    logger_.warning("当前有效采集不足 3 组，无法手眼解算；请继续采集后再按 C。")
                else:
                    try:
                        _run_hand_eye_and_save_yaml(session_dir)
                    except Exception as e:
                        logger_.error(f"按 C 标定失败: {e}", exc_info=True)

            if k == 27:
                logger_.info("用户按 ESC：将关闭窗口与相机流并退出程序。")
                return count, "esc"

            tip = f"n={count}/{target_count}  s保存  C标定+YAML  ESC退出"
            tip = tip + ("  | board OK" if found else "  | 无棋盘")
            cv2.putText(
                vis,
                tip,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 220, 0) if found else (0, 140, 255),
                2,
            )
            cv2.imshow("Capture_Video", vis)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def _run_compute(data_dir: str):
    """仅计算并打日志，不写 YAML（供 --compute-only）。"""
    rotation_matrix, translation_vector = compute_in_hand.func(data_dir)
    quat = R.from_matrix(rotation_matrix).as_quat()
    x, y, z = translation_vector.flatten()
    logger_.info(f"旋转矩阵:\n{rotation_matrix}")
    logger_.info(f"平移向量:\n{translation_vector}")
    logger_.info(f"四元数 (x,y,z,w 为 scipy 默认序): {quat}")
    logger_.info(f"平移分量 x={x}, y={y}, z={z}")
    return rotation_matrix, translation_vector


def main():
    parser = argparse.ArgumentParser(description="一键手眼标定：采集 + compute_in_hand")
    parser.add_argument("--target", type=int, default=15, help="按 s 成功保存满该张数后结束采集循环")
    parser.add_argument("--min", dest="min_count", type=int, default=8, help="少于该张数时结束会提示建议多采")
    parser.add_argument(
        "--compute-only",
        metavar="DIR",
        default=None,
        help="跳过相机与采集，仅对已有目录执行 compute_in_hand（不写 calibration_results YAML）",
    )
    args = parser.parse_args()

    if args.compute_only:
        _run_compute(args.compute_only)
        return

    # robot_ip = get_ip()
    # logger_.info(f"robot_ip: {robot_ip}")
    # if not robot_ip:
    #     popup_message("提醒", "机械臂 IP 未 ping 通")
    #     sys.exit(1)

    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect((robot_ip, 8080))
    # send_cmd(client, '{"command":"set_change_work_frame","frame_name":"Base"}', get_pose=False)

    session_dir = create_folder_with_date()
    logger_.info(f"本次数据目录: {os.path.abspath(session_dir)}")

    # count, reason = auto_collect(session_dir, client, args.target)
    # client.close()

    # if reason == "esc":
    #     logger_.info("已按 ESC 退出：窗口与相机流已关闭，未执行程序末尾标定。")
    #     sys.exit(0)

    # if count < args.min_count:
    #     logger_.warning(f"本次仅成功采集 {count} 张，建议不少于 {args.min_count} 张；可在采集中按 C 写 YAML 或使用 --compute-only。")

    # if count < 3:
    #     logger_.error("有效采集少于 3 张，无法在采集中按 C 完成手眼解算。")
    #     sys.exit(1)

    logger_.info("采集循环已正常结束。手眼 YAML 仅在采集中按「C」成功时写入 calibration_results。")


if __name__ == "__main__":
    main()
