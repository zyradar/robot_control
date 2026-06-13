from tkinter import N
import rclpy
import sys, os
from rclpy.node import Node
from rm_ros_interfaces.msg import *
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from threading import Lock
import copy, json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

class StatusSubNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)     # 继承父类的构造函数
        self.get_logger().info(f'{node_name} is open!')
        self.JointState = None
        self.Pose = None
        self.Image = None
        self.keymsg = None
        self.keydata = None
        self.actmsg = None
        self.actdata = None
        self.pose_lock = Lock()
        self.start_get_state()

    def start_get_state(self):
        self.JointState_subscriber_ = self.create_subscription(JointState, "joint_states", self.JointState_callback, 10)
        self.Pose_subscriber_ = self.create_subscription(Pose, "rm_driver/udp_arm_position", self.Pose_callback, 10)
        self.Webcmd_subscriber_ = self.create_subscription(String, "/arm/command", self.webcmd_callback, 10)
        self.Webact_subscriber_ = self.create_subscription(String, "/arm/pixel_click", self.webact_callback, 10)
        # self.Image_subscriber_ = self.create_subscription(Image, "xinwei/image_cmd", self.Image_callback, 10)

    def webact_callback(self, msg):
        with self.pose_lock:
            self.actmsg = copy.deepcopy(msg)

    def webcmd_callback(self, msg):
        with self.pose_lock:
            self.keymsg = copy.deepcopy(msg)

    def JointState_callback(self, msg):
        with self.pose_lock:
            self.JointState = copy.deepcopy(msg)

    def Pose_callback(self, msg):
        with self.pose_lock:
            self.Pose = copy.deepcopy(msg)

    def Image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='jpeg')
        with self.pose_lock:
            self.Image = copy.deepcopy(img)

    def get_current_pose(self):
        with self.pose_lock:
            if self.Pose is None:
                return None
            return copy.deepcopy(self.Pose)

    def get_current_image(self):
        with self.pose_lock:
            if self.Image is None:
                return None
            return copy.deepcopy(self.Image)

    def get_current_webcmd(self):
        with self.pose_lock:
            if self.keymsg is None:
                return None, None
            msg_data = json.loads(self.keymsg.data)
            try:
                if self.keydata is not None:
                    if self.keydata["ts"] == msg_data["ts"]:
                        return None, None
                self.keydata = msg_data
                key = self.keydata['content']['cmd']
                level = int(self.keydata['content']['speed'])
                return key, level
            except Exception as e:
                self.get_logger().warning(f"'/arm/command'接口数据异常: {e}\n当前数据包: {self.keymsg.data}")
                return None, None

    def get_current_webact(self):
        with self.pose_lock:
            if self.actmsg is None:
                return None, None
            msg_data = json.loads(self.actmsg.data)
            try:
                if self.actdata is not None:
                    if self.actdata["ts"] == msg_data["ts"]:
                        return None, None
                self.actdata = msg_data
                cmd = self.actdata['content']['cmd']
                point = None
                if cmd == 'vision':
                    pixel = self.actdata['content']['pixel']
                    point = [pixel['x'], pixel['y']]
                return cmd, point
            except Exception as e:
                self.get_logger().warning(f"'/arm/pixel_click'接口数据异常: {e}\n当前数据包: {self.keymsg.data}")
                return None, None

    def get_current_jointstate(self):
        with self.pose_lock:
            if self.JointState is None:
                return None
            return copy.deepcopy(self.JointState)

def main():
    rclpy.init()
    node = StatusSubNode('joint_state_sub')
    rclpy.spin(node)
    rclpy.shutdown()