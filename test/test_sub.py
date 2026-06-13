import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


class test_sub(Node):
    def __init__(self, node_name):
        super().__init__(node_name)     # 继承父类的构造函数
        self.get_logger().info(f'{node_name} is open!')
        self.Image = None
        self.start_get_state()

    def start_get_state(self):
        self.Image_subscriber_ = self.create_subscription(Image, "xinwei/image_cmd", self.Image_callback, 10)

    def Image_callback(self, msg):
        # 直接发送图片
        # img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # jpeg图片压缩
        np_img = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        cv2.imshow("image", img)
        cv2.waitKey(1)
        

if __name__ == "__main__":
    rclpy.init()
    node = test_sub('test_sub')
    rclpy.spin(node)
    rclpy.shutdown()