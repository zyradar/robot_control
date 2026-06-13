import numpy as np
import pyrealsense2 as rs
import os, yaml
import cv2, sys
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

class VisionSolove():
    """视觉解算"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.default_cameraRT = _PROJECT_ROOT + "/camera_data/calibration_results/calibration_data20260528.yaml"
        self.default_intrinsics = None
        self.default_R = None
        self.default_t = None
        self.default_quaternion = None

    def get_default_camera_params(self):
        """
        获取相机外参（旋转矩阵和平移向量）和相机内参
        """
        self.default_intrinsics = self.get_default_intrinsics()
        self.default_R, self.default_t, self.default_quaternion = self.get_default_cameraRT()

    def get_default_intrinsics(self):
        """
        获取相机内参
        """
        profile = self.pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        return intrinsics

    def get_default_cameraRT(self):
        """
        获取相机外参（旋转矩阵和平移向量）
        """
        print("默认相机外参路径: ", self.default_cameraRT, "\n")
        # 获取旋转矩阵、位移矩阵
        try:
            with open(self.default_cameraRT, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            default_R = data["rotation_matrix"]
            default_t = data["translation_vector"]
            default_quaternion = data["quaternion_xyzw"]
        except:
            print("未找到相机外参,请先进行相机标定")
            return None, None, None
        return default_R, default_t, default_quaternion

    def get_fixed_pose(self, default_position, pose=None):
        """
        获取固定位姿
        """
        if pose is None:
            # 只约束末端朝向
            # z = -normal
            # z = z / np.linalg.norm(z)
            z = np.array([0, 0, -1])
            ref = np.array([1, 0, 0])
            x = np.cross(ref, z)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.column_stack((x, y, z))
            quat = Rotation.from_matrix(R).as_quat()

            # 姿态强约束
            # quat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()

            quaternion = [quat[0], quat[1], quat[2], quat[3]]
            position = default_position
        else:
            quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            position = [pose.position.x, pose.position.y, pose.position.z]
        return position, quaternion

    def get_roi_pointcloud(self, u, v, depth_frame, size=20):
        """
        从点击像素(u,v)生成局部点云
        size: ROI半径（20 -> 40x40区域）
        """
        intrinsics = self.default_intrinsics
        points = []

        for du in range(-size, size):
            for dv in range(-size, size):
                uu = u + du
                vv = v + dv
                if uu < 0 or vv < 0:
                    continue

                Z = depth_frame.get_distance(uu, vv)
                if Z <= 0:
                    continue

                # 像素 -> 相机3D点
                P = rs.rs2_deproject_pixel_to_point(intrinsics, [uu, vv], Z)
                points.append(P)

        return np.array(points)
        
    def tool_to_base(self, pose_msg):
        """
        geometry_msgs/Pose -> 4x4 齐次矩阵
        构建机械臂末端坐标系到基座坐标系的齐次变换矩阵（机械臂位姿矩阵）
        """
        # 构建旋转矩阵
        qx = pose_msg.orientation.x
        qy = pose_msg.orientation.y
        qz = pose_msg.orientation.z
        qw = pose_msg.orientation.w
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

        # 构建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        return T

    def pixel_to_camera(self, u, v, depth_frame, intrinsics=None):
        """
        像素点转换为相机坐标系下坐标
        u, v: 像素坐标
        depth_image: 深度图 (单位: meters)
        intrinsics: RealSense intrinsics    （相机内参）
        """
        # 获取深度值
        Z = depth_frame.get_distance(u, v)
        if Z <= 0:
            print("该点无有效深度")
            return None

        # 获取相机内参
        if intrinsics is None:
            intrinsics = self.default_intrinsics

        # 2D --> 3D(像素点转换为相机坐标系下的三维点)
        P_camera = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], Z)
        return P_camera

    def cam_to_base(self, P_camera, arm_pose, R=None, t=None):
        """
        相机坐标系转换为世界坐标系
        P_camera: 相机坐标系下的三维点
        arm_pose: 机械臂位姿
        R: 旋转矩阵
        t: 平移向量
        return: 基座坐标系下的三维点
        """
        # 构建机械臂末端坐标系到基座坐标系的齐次变换矩阵（机械臂位姿矩阵）
        # tool --> base
        T_base_tool = self.tool_to_base(arm_pose)
        
        # 使用默认外参矩阵
        if R is None or t is None:
            R, t = self.default_R, self.default_t

        # 构建相机坐标系到世界坐标系的齐次变换矩阵
        # camera --> tool
        T_tool_camera = np.eye(4)
        T_tool_camera[:3,:3] = np.array(R)
        T_tool_camera[:3,3] = np.array(t).flatten()

        # 将相机坐标系下的三维点转换为齐次坐标
        P_camera_pixel = np.append(P_camera, 1)

        # 齐次坐标转换为世界坐标系下的三维点
        P_base = T_base_tool @ T_tool_camera @ P_camera_pixel

        # 查看世界坐标系变换数据
        print(f"T_tool_camera @ P_camera_pixel: {(T_tool_camera @ P_camera_pixel).astype(float).tolist()}")
        print(f"T_inv @ P_camera_pixel: {(np.linalg.inv(T_tool_camera) @ P_camera_pixel).astype(float).tolist()}")
        # print(f"T_base_tool @ T_tool_camera @ P_camera_pixel: {P_base.astype(float).tolist()}")

        return P_base[:3].astype(float).tolist()

    def pixel_to_base(self, u, v, depth_frame, R=None, t=None, intrinsics=None):
        """
        像素点转换为世界坐标系
        u, v: 像素坐标
        depth_image: 深度图 (单位: meters)
        intrinsics: RealSense intrinsics
        R: 旋转矩阵
        t: 平移向量
        return: 世界坐标系下的三维点
        """
        # 获取相机内参
        if intrinsics is None:
            intrinsics = self.default_intrinsics
            
        # 使用默认外参矩阵
        if R is None or t is None:
            R, t = self.default_R, self.default_t

        # 像素点转换为相机坐标系下的三维点
        P_camera = self.pixel_to_camera(u, v, depth_frame, intrinsics)
        if P_camera is None:
            return None

        # 相机坐标系下的三维点转换为世界坐标系下的三维点
        P_world = self.cam_to_base(np.array(P_camera), R, t)
        return P_world.astype(float).tolist()

def mouse_callback(event, x, y, flags, param):
    global current_color_image
    global current_depth_frame

    # 鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:

        print(f"\n点击像素坐标: x={x}, y={y}")

        # # 获取BGR颜色值
        # bgr = current_color_image[y, x]
        # print(f"BGR值: {bgr}")

        # 像素点转换为世界坐标系下的三维点
        P_world = vision_solve.pixel_to_base(x, y, current_depth_frame)
        print(f"P_world: {P_world}")

current_color_image = None
current_depth_frame = None

if __name__ == "__main__":
    # 相机流配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"相机连接异常：{e}")
        raise SystemExit(1) from e

    vision_solve = VisionSolove(pipeline)
    vision_solve.get_default_camera_params()

    cv2.namedWindow("depth_image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("color_image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("color_image", mouse_callback)

    try:
        while cv2.waitKey(1) != 27:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            current_depth_frame = depth_frame
            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            current_color_image = color_image
            # cv2.imshow("depth_image", depth_image)
            cv2.imshow("color_image", color_image)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    


    