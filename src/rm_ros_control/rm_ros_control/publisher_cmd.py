import rclpy
import random
import sys, os
import cv2
from rclpy.node import Node
from rm_ros_interfaces.msg import *
from sensor_msgs.msg import *
from cv_bridge import CvBridge

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from arm_sdk.RM65_B1.arm_limit import *

"""
ros2 topic pub --once /rm_driver/movej_cmd rm_ros_interfaces/msg/Movej "{joint: [0.1, -0.6, 1.8432, 0.0, 1.125, 0.3],
 speed: 50, block: true, trajectory_connect: 0, dof: 6}"
ros2 topic pub --once /rm_driver/movej_p_cmd rm_ros_interfaces/msg/Movejp  \
"{pose=(geometry_msgs.msg.Point(x=0.0084, y=0.2876, z=0.19466),  \
geometry_msgs.msg.Quaternion(x=0.6845920085906982, y=-0.7048959732055664, z=-0.12933099269866943, w=-0.13314400613307953)), \
 speed=50, trajectory_connect=0, block=True}"
"""

class CmdPubNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)     # 继承父类的构造函数
        self.get_logger().info(f'{node_name} is open!')
        self.grea_level= [20, 50, 80]
        self.init_publisher()

    def init_publisher(self):
        """
        初始化话题发布接口
        """
        self.movej_publisher_ = self.create_publisher(Movej, 'rm_driver/movej_cmd', 10)
        self.movejp_publisher_ = self.create_publisher(Movejp, 'rm_driver/movej_p_cmd', 10)
        self.image_publisher_ = self.create_publisher(CompressedImage, 'xinwei/image_cmd', 10)

    def auto_calibration_MoveJ(self, pose=None, base_type="on_weiyi"):
        # 计算随机位姿
        if pose is None:
            pose ={}
            for i in [0, 1, 3, 5]:
                pose[Joint_name[i]] = random.uniform(Auto_calibration_limit[base_type][Joint_name[i]][0], Auto_calibration_limit[base_type][Joint_name[i]][1])
            pose[Joint_name[2]] = (Auto_calibration_sum_rad[base_type] - pose[Joint_name[1]]) * random.random()
            pose[Joint_name[4]] = (Auto_calibration_sum_rad[base_type] - pose[Joint_name[1]] - pose[Joint_name[2]]) * random.random()
                
        # 构建消息结构
        msg = Movej()
        msg.joint = [pose[Joint_name[0]], pose[Joint_name[1]], pose[Joint_name[2]], pose[Joint_name[3]], pose[Joint_name[4]], pose[Joint_name[5]]]  # 弧度
        msg.speed = self.grea_level[0]
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self.movej_publisher(msg)

    def vision_to_movejp(self, P_world, quaternion, level=None):
        """
        视觉控制
        视觉坐标系转换为轨迹规划
        P_world: 世界坐标系下的三维点
        quaternion: 四元数
        return: movejp.msg
        """
        msg = Movejp()
        msg.pose.position.x = P_world[0]
        msg.pose.position.y = P_world[1]
        msg.pose.position.z = P_world[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        msg.speed = self.grea_level[level - 1] if level is not None else self.grea_level[1]
        msg.block = True
        # print(f"轨迹规划: {msg}")
        self.movejp_publisher(msg)

    def img_to_msg(self, img):
        """
        相机流话题发布接口
        """
        # msg = CvBridge().cv2_to_imgmsg(img, encoding='bgr8')
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = buffer.tobytes()

        self.image_publisher(msg)

    def movej_publisher(self, msg):
        self.movej_publisher_.publish(msg)
        self.get_logger().info(f'publish:{msg}')

    def movejp_publisher(self, msg):
        self.movejp_publisher_.publish(msg)
        self.get_logger().info(f'publish:{msg}')

    def image_publisher(self, msg):
        # image_publisher_ = self.create_publisher(Image, 'xinwei/image_cmd', 10)  # 发布话题
        self.image_publisher_.publish(msg)

def main():
    rclpy.init()
    node = CmdPubNode('cmd_pub')
    rclpy.spin(node)
    rclpy.shutdown()



