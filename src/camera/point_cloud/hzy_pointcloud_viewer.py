# coding=utf-8

import math
import cv2
import numpy as np
import pyrealsense2 as rs

# =========================
# 初始化 RealSense
# =========================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 点云对象
pc = rs.pointcloud()

# =========================
# 相机视角参数
# =========================

yaw = 0
pitch = 0
translation = np.array([0, 0, -1], dtype=np.float32)

# 输出图像
out = np.zeros((480, 640, 3), dtype=np.uint8)

print("按键控制：")
print("W/S : 上下旋转")
print("A/D : 左右旋转")
print("Q/E : 缩放")
print("ESC : 退出")

# =========================
# 旋转矩阵
# =========================

def rotation_matrix(axis, theta):

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))

    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)

    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d),     2*(b*d+a*c)],
        [2*(b*c+a*d),     a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c),     2*(c*d+a*b),     a*a+d*d-b*b-c*c]
    ])

# =========================
# 主循环
# =========================

try:
    cv2.namedWindow("PointCloud Viewer", cv2.WINDOW_NORMAL)
    cv2.namedWindow("color_frame", cv2.WINDOW_NORMAL)
    while True:

        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # RGB图
        color_image = np.asanyarray(color_frame.get_data())

        # 点云映射到RGB
        pc.map_to(color_frame)

        # 生成点云
        points = pc.calculate(depth_frame)

        # 获取顶点
        verts = np.asanyarray(points.get_vertices())
        verts = verts.view(np.float32).reshape(-1, 3)

        # 获取纹理坐标
        texcoords = np.asanyarray(points.get_texture_coordinates())
        texcoords = texcoords.view(np.float32).reshape(-1, 2)

        # 清空画布
        out.fill(0)

        # =========================
        # 旋转矩阵
        # =========================

        Rx = rotation_matrix((1, 0, 0), pitch)
        Ry = rotation_matrix((0, 1, 0), yaw)

        R = Ry @ Rx

        # =========================
        # 遍历点云
        # =========================

        for i in range(0, len(verts), 2):

            v = verts[i]

            if v[2] <= 0:
                continue

            # 旋转
            v = R @ v

            # 平移
            v += translation

            if v[2] <= 0:
                continue

            # 投影
            x = int((v[0] / v[2]) * 320 + 320)
            y = int((v[1] / v[2]) * 320 + 240)

            if 0 <= x < 640 and 0 <= y < 480:

                # RGB纹理
                u, vv = texcoords[i]

                tx = min(max(int(u * 640), 0), 639)
                ty = min(max(int(vv * 480), 0), 479)

                color = color_image[ty, tx]

                out[y, x] = color

        # 显示
        cv2.imshow("PointCloud Viewer", out)
        cv2.imshow("color_frame", color_image)

        # =========================
        # 键盘控制
        # =========================

        key = cv2.waitKey(1)

        if key == 27:
            break

        elif key == ord('a'):
            yaw -= 0.1

        elif key == ord('d'):
            yaw += 0.1

        elif key == ord('w'):
            pitch -= 0.1

        elif key == ord('s'):
            pitch += 0.1

        elif key == ord('q'):
            translation[2] += 0.1

        elif key == ord('e'):
            translation[2] -= 0.1

finally:

    pipeline.stop()
    cv2.destroyAllWindows()