from tkinter import N
import cv2, logging, os, sys
import pyrealsense2 as rs
import numpy as np
import open3d
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from pixel_to_worlld import VisionSolove
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from src.rm_ros_control.rm_ros_control.publisher_cmd import CmdPubNode
from src.rm_ros_control.rm_ros_control.subscription_status import StatusSubNode
from arm_sdk.RM65_B1.arm_limit import Joint_init_position

class PixelFormMouse():
    def __init__(self, pipeline, config):
        self.pipeline = pipeline
        self.config = config
        self.vision_solve = VisionSolove(pipeline)
        self.customize_x = Joint_init_position["arm_zero_init"]["position"][0]
        self.customize_y = Joint_init_position["arm_zero_init"]["position"][1]
        self.customize_z = Joint_init_position["arm_zero_init"]["position"][2]
        self.color_frame = None
        self.depth_frame = None
        self.color_image = None
        self.depth_image = None
        self.click_point = None
        self.status_sub_node = None
        self.Arm_control_node = None
        self.key = None
        self.test_cnt = 1
        self.arm_type = "arm_action_init"
        self.base_type = "on_weiyi"
        self.arm_recover_type = "arm_action_init"

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数
        """
        # 鼠标左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            print(f"\n点击像素坐标: x={x}, y={y}")
            self.vision_to_movejp(x, y)

    def show_pointcloud(self, points):
        """
        显示点云
        """
        if len(points) == 0:
            print("空点云")
            return

        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(points)
        pc.paint_uniform_color([1, 0.7, 0])     # 上色（统一白色）

        # 坐标轴
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        open3d.visualization.draw_geometries([pc, coord])

    def vision_to_movejp(self, x, y):
        # # 获取BGR颜色值
        # bgr = self.color_image[y, x]
        # print(f"BGR值: {bgr}")

        # 获取当前机械臂位姿矩阵
        status_pose = self.status_sub_node.get_current_pose()
        if status_pose is None:
            return
        print("status_pose: ", status_pose)
        position = [self.customize_x, self.customize_y, self.customize_z]
        P_world, quaternion = self.vision_solve.get_fixed_pose(position, status_pose)

        # 获取相机坐标系坐标
        P_camera = self.vision_solve.pixel_to_camera(x, y, self.depth_frame)
        print(f"P_camera: {P_camera}")
        if P_camera is None:
            return

        # 获取世界坐标系坐标
        P_world = self.vision_solve.cam_to_base(P_camera, status_pose)
        print(f"P_world: {P_world}")

        # 获取局部点云
        points_cam = self.vision_solve.get_roi_pointcloud(x, y, self.depth_frame, size=20)
        self.show_pointcloud(points_cam)


        # 获取末端向下位姿
        # P_world, quaternion = self.vision_solve.get_fixed_pose(position)
        # print(f"P_world: {P_world}\nquaternion: {quaternion}")  

        # 获取轨迹规划
        # if P_world is not None:
        #     P_world = [P_world[2], P_world[1], P_world[0]]
        #     quaternion, position = self.vision_solve.get_fixed_pose(position)
        #     self.Arm_control_node.vision_to_movejp(P_world, quaternion)
        #     self.customize_x, self.customize_y, self.customize_z = P_world[0:3]
    
    def customize_control(self, key, level=None, pixel=None):
        """自定义控制"""
        # 自定义控制
        # if key != -1:
        #     print(f"key down {chr(key)}!!!")
        control_flag = False
        if key == ord('w'):
            self.customize_z += 0.05
            control_flag = True
        elif key == ord('s'):
            self.customize_z -= 0.05
            control_flag = True
        elif key == ord('a'):
            self.customize_y -= 0.05
            control_flag = True
        elif key == ord('d'):
            self.customize_y += 0.05
            control_flag = True
        elif key == ord('q'):
            self.customize_x -= 0.05
            control_flag = True
        elif key == ord('e'):
            self.customize_x += 0.05
            control_flag = True
        elif key == ord('z'):
            self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_zero_init"], self.base_type)
            self.arm_recover_type = "arm_zero_init"
        elif key == ord('x'):
            self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_user_init"], self.base_type)
            self.arm_recover_type = "arm_user_init"
        elif key == ord('c'):
            self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position[self.arm_type], self.base_type)
            self.arm_recover_type = self.arm_type
        elif key == ord('j') or key == ord('k'):
            control_flag = True
        # elif key == ord('v'):
        #     self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_auto_init"])
        elif key == 'vision':
            self.vision_to_movejp(pixel[0], pixel[1])
            # if self.test_cnt == 1:
            #     self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_tool_1"])
            #     self.test_cnt += 1
            # else:
            #     self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_tool_2"])
            #     self.test_cnt = 1
        elif key == 'action':
            self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_action_init"], self.base_type)
            self.arm_recover_type = "arm_action_init"

        # 无控制指令，返回
        if control_flag is False:
            self.customize_x, self.customize_y, self.customize_z = Joint_init_position[self.arm_recover_type]["position"]
            return

        # 获取当前机械臂位姿
        status_pose = self.status_sub_node.get_current_pose()
        if status_pose is None:
            return
        
        # 获取自定义轨迹规划
        position = [self.customize_x, self.customize_y, self.customize_z]
        P_world, quaternion = self.vision_solve.get_fixed_pose(position, status_pose)
        # P_world, quaternion = self.vision_solve.get_fixed_pose(position)
        if P_world is not None:
            if key == ord('j'):
                quaternion[3] -= 0.01
            elif key == ord('k'):
                quaternion[3] += 0.01
            print(f"P_world: {position}\nquaternion: {quaternion}")
            self.Arm_control_node.vision_to_movejp(position, quaternion, level)

    def process(self):
        """
        执行体
        """
        # 相机流配置
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.pipeline.start(self.config)
            self.vision_solve.get_default_camera_params()
        except Exception as e:
            print(f"相机连接异常：{e}")
            raise SystemExit(1) from e
        cv2.namedWindow("color_image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("color_image", self.mouse_callback)

        # 创建机械臂控制节点
        rclpy.init()
        self.Arm_control_node = CmdPubNode('pixel_from_mouse_node')
        # self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_auto_init"])

        # 创建状态订阅节点
        self.status_sub_node = StatusSubNode('status_sub_node')
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.status_sub_node)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros_thread.start()

        try:
            while self.key != 27:
                frames = self.pipeline.wait_for_frames()
                self.color_frame = frames.get_color_frame()
                self.depth_frame = frames.get_depth_frame()
                if not self.color_frame or not self.depth_frame:
                    continue
                
                self.color_image = np.asanyarray(self.color_frame.get_data())
                # self.depth_image = np.asanyarray(self.depth_frame.get_data())

                cv2.imshow("color_image", self.color_image)
                self.Arm_control_node.img_to_msg(self.color_image)
                # cv2.imshow("depth_image", self.depth_image)
                self.key = cv2.waitKey(1)
                self.customize_control(self.key)

                key, level = self.status_sub_node.get_current_webcmd()
                if key is not None:
                    self.customize_control(key, level)

                cmd, pixel = self.status_sub_node.get_current_webact()
                if cmd is not None:
                    self.customize_control(cmd, None, pixel)
        finally:
            # self.Arm_control_node.auto_calibration_MoveJ(Joint_init_position["arm_user_init"])
            print("退出程序!!!")
            self.close()

    def run(self):
        self.process()

    def close(self):
        self.executor.shutdown()
        self.status_sub_node.destroy_node()
        self.pipeline.stop()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pixel_form_mouse = PixelFormMouse(rs.pipeline(), rs.config())
    pixel_form_mouse.run()

