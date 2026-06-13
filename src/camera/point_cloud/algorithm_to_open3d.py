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

# OpenCV 窗口内按键：m 切换点云来源；e 导出 Open3D 点云 PLY；x 导出 RealSense export_to_ply
_O3D_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

def callback(frame, ui_state=None):

    scaling_factor = 2.0
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧

    if ui_state is not None:
        if k == ord("m"):
            ui_state["viz"] = "realsense" if ui_state["viz"] == "open3d" else "open3d"
            src = "Open3D(RGBD)" if ui_state["viz"] == "open3d" else "RealSense(rs.pointcloud)"
            logger_.info(f"点云显示已切换为: {src}")
        elif k == ord("e"):
            ui_state["req_o3d_export"] = True
        elif k == ord("x"):
            ui_state["req_rs_export"] = True

    if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串

        socket_command = '{"command": "get_current_arm_state"}'
        state,pose = send_cmd(client,socket_command)
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
        if state:

            filename = os.path.join(cam0_origin_path,"poses.txt")

            with open(filename, 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
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


def _rs_intrinsics_to_o3d_pinhole(intr) -> o3d.camera.PinholeCameraIntrinsic:
    """RealSense 视频流内参 → Open3D 针孔模型（不依赖 rs.pointcloud）。"""
    return o3d.camera.PinholeCameraIntrinsic(
        int(intr.width),
        int(intr.height),
        float(intr.fx),
        float(intr.fy),
        float(intr.ppx),
        float(intr.ppy),
    )


def pointcloud_from_aligned_bgr_depth_o3d(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    pinhole: o3d.camera.PinholeCameraIntrinsic,
    depth_scale_m_per_unit: float,
    depth_trunc_m: float = 8.0,
) -> o3d.geometry.PointCloud:
    """
    用 Open3D 针孔反投影由对齐后的彩色图 + 深度图生成点云。
    depth_u16 为 RealSense Z16 原始值；depth_scale_m_per_unit 来自 depth_sensor.get_depth_scale()。
    """
    if color_bgr.shape[:2] != depth_u16.shape[:2]:
        raise ValueError(
            f"对齐后彩色与深度尺寸需一致，当前 color {color_bgr.shape[:2]} vs depth {depth_u16.shape[:2]}"
        )
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_u16.astype(np.uint16))

    # Open3D: 深度(米) = 原始深度 / depth_scale；RS 的 scale 为「每单位米数」，故 depth_scale = 1/scale
    o3d_depth_scale = 1.0 / float(depth_scale_m_per_unit)

    """合成RGBD图像"""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=o3d_depth_scale,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )

    """生成点云"""
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)         # 用针孔模型对每个有效深度像素做反投影(U,V) + depth = (x,y,z)
    # 与 Open3D 可视化常用坐标系一致（相机 Z 朝前 → 右手系展示）
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])   # 将点云坐标系转换为与 Open3D 可视化常用坐标系一致
    return pcd


def pointcloud_rs_aligned_to_o3d(
    pc: rs.pointcloud,
    depth_frame,
    color_frame,
    color_bgr: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    使用 RealSense rs.pointcloud 计算顶点 + map_to 彩色纹理，再转为 Open3D 点云用于同窗口对比。
    对坐标施加与 pointcloud_from_aligned_bgr_depth_o3d 相同的翻转，便于与 Open3D 分支对照。
    """
    """点云计算"""
    points = pc.calculate(depth_frame)      # 将深度像素坐标转换为世界坐标
    pc.map_to(color_frame)                  # 将点云与彩色纹理对齐

    w, h = depth_frame.get_width(), depth_frame.get_height()

    verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
    texcoords = np.asarray(points.get_texture_coordinates(2))

    verts_flat = verts.reshape(-1, 3)
    tex_flat = texcoords.reshape(-1, 2)

    """把归一化(u,v)映射到彩色纹理的像素坐标"""
    ch, cw = color_bgr.shape[0], color_bgr.shape[1]
    u = np.clip((tex_flat[:, 0] * cw).astype(np.int32), 0, cw - 1)
    v_pix = np.clip((tex_flat[:, 1] * ch).astype(np.int32), 0, ch - 1)

    bgr = color_bgr[v_pix, u]
    rgb = bgr[:, ::-1].astype(np.float64) / 255.0

    valid = np.isfinite(verts_flat).all(axis=1) & (verts_flat[:, 2] > 0)
    v = verts_flat[valid]
    v_flip = ( _O3D_FLIP @ v.T).T

    """生成点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v_flip)
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid])
    return pcd


def displayD435():

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        logger_.error_(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")

        sys.exit(1)

    depth_sensor = profile.get_device().first_depth_sensor()    # 获取深度传感器句柄
    depth_scale_m = depth_sensor.get_depth_scale()      # 获取深度单位比例
    align = rs.align(rs.stream.color)
    pc_rs = rs.pointcloud()

    global count
    count = 1

    ui_state = {
        "viz": "open3d",
        "req_o3d_export": False,
        "req_rs_export": False,
    }

    logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")
    logger_.info(
        "默认点云: Open3D(RGBD)。Capture_Video 焦点下 [m] 与 RealSense(rs.pointcloud) 切换显示；"
        "[e] 导出当前帧 Open3D 点云 PLY；[x] 导出当前帧 RealSense 纹理 PLY（export_to_ply）。"
    )

    """创建Open3D可视化窗口"""
    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window(window_name="HandEye D435 (点云对比 / Open3D 默认)")
    pcd_o3d = o3d.geometry.PointCloud()
    first_o3d = True

    try:
        while True:
            """获取深度与彩色帧"""
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_u16 = np.asanyarray(depth_frame.get_data())
            color_intr = rs.video_stream_profile(color_frame.profile).get_intrinsics()      # 对齐深度与彩色尺寸,用彩色流内参做open3d反投影与RGBD一致
            pinhole = _rs_intrinsics_to_o3d_pinhole(color_intr)                            # 将RealSense内参转换为Open3D针孔模型

            """"每帧点云计算"""
            if ui_state["viz"] == "realsense":
                pcd_show = pointcloud_rs_aligned_to_o3d(
                    pc_rs, depth_frame, color_frame, color_image
                )
            else:
                pcd_o3d_frame = pointcloud_from_aligned_bgr_depth_o3d(
                    color_image, depth_u16, pinhole, depth_scale_m
                )
                pcd_show = pcd_o3d_frame
            pcd_o3d.points = pcd_show.points
            pcd_o3d.colors = pcd_show.colors

            if first_o3d:
                vis_o3d.add_geometry(pcd_o3d)
                first_o3d = False
            else:
                vis_o3d.update_geometry(pcd_o3d)
            vis_o3d.poll_events()
            vis_o3d.update_renderer()

            callback(color_image, ui_state)

            if ui_state["req_o3d_export"]:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(cam0_origin_path, f"frame_open3d_{ts}.ply")
                o3d.io.write_point_cloud(path, pcd_o3d_frame)
                logger_.info(f"已导出 Open3D 点云: {path}")
                ui_state["req_o3d_export"] = False
            if ui_state["req_rs_export"]:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(cam0_origin_path, f"frame_realsense_{ts}.ply")
                rs_pts = pc_rs.calculate(depth_frame)
                pc_rs.map_to(color_frame)
                rs_pts.export_to_ply(path, color_frame)
                logger_.info(f"已导出 RealSense PLY: {path}")
                ui_state["req_rs_export"] = False

    finally:
        try:
            vis_o3d.destroy_window()
        except Exception:
            pass
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # robot_ip = get_ip()

    # logger_.info(f'robot_ip:{robot_ip}')

    # if robot_ip:

    #     client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     client.connect((robot_ip, 8080))
    #     socket_command = '{"command":"set_change_work_frame","frame_name":"Base"}'
    #     send_cmd(client,socket_command,get_pose = False)

    # else:

    #     popup_message("提醒", "机械臂ip没有ping通")
    #     sys.exit(1)

    displayD435()
