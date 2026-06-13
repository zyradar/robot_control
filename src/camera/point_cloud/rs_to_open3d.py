# coding=utf-8
import json
import logging,os
import socket
import time
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

# 直接运行本脚本时加入路径：项目根（camera_sdk 包）+ realsense 目录（libs.* 相对导入）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from camera_sdk.realsense.libs.log_setting import CommonLog
from camera_sdk.realsense.libs.auxiliary import create_folder_with_date, get_ip, popup_message

cam0_origin_path = create_folder_with_date() # 提前建立好的存储照片文件的目录


logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

def callback(frame, type: str, snapshot_hook=None, color_mode_rgb_ref=None):

    scaling_factor = 2.0
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧

    if snapshot_hook is not None and k == ord("t"):
        snapshot_hook()

    if color_mode_rgb_ref is not None and (k == ord("c") or k == ord("C")):
        color_mode_rgb_ref[0] = not color_mode_rgb_ref[0]
        mode = "RGB纹理" if color_mode_rgb_ref[0] else "深度伪彩色(JET)"
        logger_.info(f"点云着色切换 [C]: {mode}")

    if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串
        socket_command = '{"command": "get_current_arm_state"}'
        state,pose = send_cmd(client,socket_command)
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')

        if state:
            filename = os.path.join(cam0_origin_path,"poses.txt")
            with open(filename, 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]      # x,y,z,rx,ry,rz
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            image_path = os.path.join(cam0_origin_path,f"{str(count)}.jpg")
            cv2.imwrite(image_path , cv_img)
            logger_.info(f"===采集第{count}次数据！")

        count += 1

    else:
        pass


def send_cmd(client, cmd, get_pose=True):
    """
    发送命令到机械臂并可选择性地获取姿态(pose)数据

    参数:
    client: socket客户端连接
    cmd: 要发送的命令字符串或JSON字符串
    get_pose: 是否需要获取pose数据

    返回:
    如果get_pose为True，返回tuple (状态, pose或错误信息)
    如果get_pose为False，返回布尔值表示命令是否成功发送
    """
    client.send(cmd.encode('utf-8'))

    if not get_pose:
        response = client.recv(1024).decode('utf-8')
        logger_.info(f"response:{response}")
        return True

    time.sleep(0.1)
    response = client.recv(4096).decode('utf-8')  # 增大接收缓冲区
    logger_.info(f'response:{response}')

    try:
        decoder = json.JSONDecoder()
        data_list = []
        index = 0
        # 分割并解析所有可能的JSON对象
        while index < len(response):
            try:
                # 跳过空白字符
                while index < len(response) and response[index].isspace():
                    index += 1
                if index >= len(response):
                    break
                obj, idx = decoder.raw_decode(response[index:])
                data_list.append(obj)
                index += idx
            except json.JSONDecodeError as e:
                logger_.error(f"JSON解析错误：{str(e)}")
                break

        # 寻找最后一个包含目标状态的响应
        target_data = None
        for data in reversed(data_list):
            if data.get("state") == "current_arm_state":
                target_data = data
                break

        if not target_data:
            return False, "未找到有效的机械臂状态响应"

        # 检查错误码
        if target_data["arm_state"]["err"] != [0]:
            return False, f"机械臂报错: {target_data['arm_state']['err']}"

        # 转换单位
        pose_raw = target_data["arm_state"]["pose"]
        pose_converted = [
            pose_raw[0] / 1000000,  # x: 0.001mm → m
            pose_raw[1] / 1000000,  # y: 0.001mm → m
            pose_raw[2] / 1000000,  # z: 0.001mm → m
            pose_raw[3] / 1000,    # rx: 0.001rad → rad
            pose_raw[4] / 1000,    # ry: 0.001rad → rad
            pose_raw[5] / 1000     # rz: 0.001rad → rad
        ]

        return True, pose_converted

    except json.JSONDecodeError:
        return False, "JSON解析错误"
    except KeyError as e:
        return False, f"响应缺少关键字段: {str(e)}"
    except Exception as e:
        return False, f"处理响应时发生错误: {str(e)}"


DEBUG_SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "open3d")


def vertex_colors_depth_jet(verts_xyz):
    """按相机 Z（深度方向）做 JET 伪彩色，与 RealSense/pyglet 深度彩虹风格接近。返回 (N,3) float64 RGB [0,1]。"""
    z = verts_xyz[:, 2].astype(np.float64)
    z_pos = z[z > 0]
    if z_pos.size > 0:
        lo, hi = np.percentile(z_pos, [2, 98])
        if hi <= lo:
            lo, hi = float(z_pos.min()), float(z_pos.max())
    else:
        lo, hi = 0.0, 1.0
    t = np.clip((z - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    t_u8 = (t * 255).astype(np.uint8).reshape(-1, 1)
    jet_bgr = cv2.applyColorMap(t_u8, cv2.COLORMAP_JET).reshape(-1, 3)
    return jet_bgr[:, ::-1].astype(np.float64) / 255.0


def save_open3d_debug_bundle(depth_frame, color_frame, color_image_bgr, rs_points_obj, pcd_o3d):
    """
    保存深度 / 彩色 / Open3D 点云 / RealSense 导出 PLY，便于与 RealSense Viewer、pyglet 示例对比。
    在 Capture_Video 窗口激活时按键盘【t】触发。
    """
    os.makedirs(DEBUG_SAVE_ROOT, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DEBUG_SAVE_ROOT, f"snapshot_{ts}")

    depth_u16 = np.asanyarray(depth_frame.get_data())
    cv2.imwrite(f"{base}_depth_u16.png", depth_u16)

    valid_d = depth_u16[depth_u16 > 0]
    if valid_d.size > 0:
        lo, hi = np.percentile(valid_d, [2, 98])
        vis = np.clip(depth_u16.astype(np.float32), lo, hi)
        vis = ((vis - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(depth_u16, dtype=np.uint8)
    vis_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{base}_depth_colormap.png", vis_bgr)

    cv2.imwrite(f"{base}_color_bgr.png", color_image_bgr)

    o3d.io.write_point_cloud(f"{base}_open3d_rgb.ply", pcd_o3d)

    np.save(f"{base}_depth_u16.npy", depth_u16)

    try:
        rs_points_obj.export_to_ply(f"{base}_realsense_tex.ply", color_frame)
    except Exception as ex:
        logger_.warning(f"RealSense export_to_ply 失败（可忽略用于对比）: {ex}")

    n_pts = np.asarray(pcd_o3d.points).shape[0]
    meta = (
        f"aligned_depth_to_color: yes (rs.align)\n"
        f"depth_shape: {depth_u16.shape}\n"
        f"color_shape: {color_image_bgr.shape}\n"
        f"depth_valid_pixels: {int(np.sum(depth_u16 > 0))}\n"
        f"open3d_num_points: {n_pts}\n"
    )
    with open(f"{base}_meta.txt", "w", encoding="utf-8") as mf:
        mf.write(meta)

    logger_.info(f"调试数据已写入: {base}_* （目录 {DEBUG_SAVE_ROOT}）")


#
def displayD435():
    # RealSense数据流
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 开启深度+彩色数据流
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    # config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    try:
        # 启动相机数据流
        pipeline.start(config)

        # 深度对齐到彩色（减少 RGB / 深度错位，利于纹理与点云一致）
        align_to_color = rs.align(rs.stream.color)
        
        # 创建点云对象，用于存储深度数据，点云生成器
        pc = rs.pointcloud()
        
        # Open3D可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='PointCloud')

        color_mode_rgb = [True]  # True=彩色纹理；False=深度JET（与 pyglet [c] 类似）

        def _toggle_pointcloud_color(vis_obj):
            color_mode_rgb[0] = not color_mode_rgb[0]
            mode = "RGB纹理" if color_mode_rgb[0] else "深度伪彩色(JET)"
            logger_.info(f"点云着色切换 [C]: {mode}")
            return False

        if hasattr(vis, "register_key_callback"):
            vis.register_key_callback(ord("C"), _toggle_pointcloud_color)
            vis.register_key_callback(ord("c"), _toggle_pointcloud_color)
        
        # 点云容器
        point_cloud = o3d.geometry.PointCloud()
        
        first_frame = True
    except Exception as e:
        logger_.error_(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")

        sys.exit(1)

    global count
    count = 1

    logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")
    logger_.info("按 [C] 切换点云：RGB纹理 / 深度伪彩色(JET)；Capture_Video 或 PointCloud 窗口均可")

    try:
        cv2.namedWindow("Capture_Video", cv2.WINDOW_NORMAL)
        while True:
            # 获取一帧数据并对齐（深度图重投影到彩色分辨率，与 color_image 像素对齐）
            success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
            # frames = align_to_color.process(frames)
            if not success:
                continue

            depth = frames.get_depth_frame()                    # 已与彩色对齐的深度帧
            color_frame = frames.get_color_frame()              # 彩色RGB帧
            if not depth or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())     # 彩色BGR图像转numpy数组

            # 点云生成器，将深度数据映射到彩色图像上（须先于 calculate）
            pc.map_to(color_frame)
            points = pc.calculate(depth)

            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            h, w = color_image.shape[0], color_image.shape[1]
            u = np.clip((texcoords[:, 0] * w).astype(np.int32), 0, w - 1)
            v_pix = np.clip((texcoords[:, 1] * h).astype(np.int32), 0, h - 1)

            bgr = color_image[v_pix, u]
            rgb = bgr[:, ::-1].astype(np.float64) / 255.0

            valid = np.isfinite(verts).all(axis=1) & (verts[:, 2] > 0)
            verts = verts[valid]
            rgb = rgb[valid]

            if color_mode_rgb[0]:
                display_rgb = rgb
            else:
                display_rgb = vertex_colors_depth_jet(verts)

            point_cloud.points = o3d.utility.Vector3dVector(verts)
            point_cloud.colors = o3d.utility.Vector3dVector(display_rgb)

            if first_frame:
                vis.add_geometry(point_cloud)
                first_frame = False

            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()

            callback(
                color_image,
                "img",
                snapshot_hook=lambda: save_open3d_debug_bundle(
                    depth, color_frame, color_image, points, point_cloud
                ),
                color_mode_rgb_ref=color_mode_rgb,
            )
#            callback(np_image, "depth")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    displayD435()
