# coding=utf-8
import json, yaml
import sys, time
import logging, os
import cv2, socket
import rclpy
import numpy as np
import pyrealsense2 as rs

# 直接运行本脚本时加入路径：项目根（camera_sdk 包）+ realsense 目录（libs.* 相对导入）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REALSENSE_ROOT = os.path.join(_PROJECT_ROOT, "camera_sdk", "realsense")
for _path in (_PROJECT_ROOT, _REALSENSE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.rm_ros_control.rm_ros_control.publisher_cmd import CmdPubNode
from camera_sdk.realsense.libs.log_setting import CommonLog
from camera_sdk.realsense.libs.auxiliary import create_folder_with_date, get_ip, popup_message, find_latest_data_folder
from scipy.spatial.transform import Rotation as R
from camera_sdk.realsense.save_poses import poses_main
from arm_sdk.RM65_B1.arm_limit import Joint_init_position


class AutomaticCalibration():
    def __init__(self, pipeline_: rs.pipeline, config_: rs.config):
        self.calibration_data_path = create_folder_with_date()
        # self.calibration_data_path = "C:/HZYSoft/Project/Preview/XINWEI_RealMan/camera_data/eye_hand_data/calibration_data20260602"
        # self.calibration_data_path = "/home/nvidia/projects/XINWEI_RealMan/camera_data/eye_hand_data/calibration_data2026060201"
        self.logger_ = CommonLog(logging.getLogger(__name__))
        self.cali_img_count = 0
        self.scaling_factor = 2.0
        self.pipeline = pipeline_
        self.config = config_
        self.key = None
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.arm_init = "arm_action_init"
        self.base_type = "on_weiyi"

    def get_calibration_board(self):
        with open(os.path.join(os.path.dirname(__file__), "calibration_config.yaml"), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        self.XX = data.get("checkerboard_args").get("XX") #标定板的中长度对应的角点的个数
        self.YY = data.get("checkerboard_args").get("YY") #标定板的中宽度对应的角点的个数
        self.L = data.get("checkerboard_args").get("L")   #标定板一格的长度  单位为米

    def calibrate_camera_func(self):
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

        # 获取标定板角点的位置
        objp = np.zeros((self.XX * self.YY, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.XX, 0:self.YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        objp = self.L * objp

        obj_points, img_points = [], []     # 存储3D, 2D点

        images_num = [f for f in os.listdir(self.calibration_data_path) if f.endswith('.jpg')]
        imgsave_path = os.path.dirname(os.path.dirname(self.calibration_data_path)) + "/calibration_check/" + os.path.basename(self.calibration_data_path)
        os.makedirs(os.path.dirname(imgsave_path), exist_ok=True)
        os.makedirs(imgsave_path, exist_ok=True)
        for i in range(0, len(images_num) + 1):   # 标定好的图片在images_path路径下，从0.jpg到x.jpg
            image_file = os.path.join(self.calibration_data_path,f"{i}.jpg")

            if os.path.exists(image_file):
                self.logger_.info(f'读 {image_file}')

                img = cv2.imread(image_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, (self.XX, self.YY), None)

                if ret:
                    obj_points.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                    cv2.drawChessboardCorners(img, (self.XX, self.YY), corners2, ret)
                    if [corners2]:
                        img_points.append(corners2)
                    else:
                        img_points.append(corners)

                cv2.imwrite(os.path.join(imgsave_path, f"{i}.jpg"), img)
                self.logger_.info(f"保存到{imgsave_path}")
        N = len(img_points)
        if N == 0:
            raise RuntimeError(
                "所有图片均未检测到棋盘内角点，无法调用 calibrateCamera。"
                "请核对 config.yaml 中 XX、YY 是否与 OpenCV 约定一致（每行/列的内角点个数，不是方格数），"
                "并检查图像是否完整包含标定板、光照、模糊与棋盘方向。"
            )
        else:
            self.logger_.info(f'标定总计：{len(images_num)}张， 成功检测角点：{N}张， 未检测到角点：{len(images_num) - N}张')

        # 标定,得到图案在相机坐标系下的位姿
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        # logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵1
        # logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        
        # 机器人末端在基座标系下的位姿
        poses_main(os.path.join(self.calibration_data_path, "poses.txt"))
        # poses_main("C:/HZYSoft/Project/Preview/hand_eye_calibration/eye_hand_data/data20260512/poses.txt")

        csv_file = os.path.join(self.calibration_data_path, "RobotToolPose.csv")
        print("csv_file:",csv_file)
        tool_pose = np.loadtxt(csv_file,delimiter=',')

        # 获取手眼标定旋转矩阵与平移矩阵
        R_tool, t_tool = [], []
        for i in range(int(N)):
            R_tool.append(tool_pose[0:3,4*i:4*i+3])
            t_tool.append(tool_pose[0:3,4*i+3])
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

        return R,t

    def save_calibration_results(self):
        rotation_matrix, translation_vector = self.calibrate_camera_func()
        
        # 将旋转矩阵转换为四元数
        rotation = R.from_matrix(rotation_matrix)   
        quaternion = rotation.as_quat()
        x, y, z = translation_vector.flatten()

        results_dir = os.path.join(os.path.dirname(os.path.dirname(self.calibration_data_path)), "calibration_results")
        os.makedirs(results_dir, exist_ok=True)
        yaml_path = os.path.join(results_dir, f"{os.path.basename(self.calibration_data_path)}.yaml")
        payload = {
            "checkerboard_args":{"XX": self.XX, "YY": self.YY, "L": self.L},
            "rotation_matrix": np.asarray(rotation_matrix).tolist(),
            "translation_vector": translation_vector.flatten().tolist(),
            "quaternion_xyzw": quaternion.tolist(),
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        self.logger_.info(f"标定结果已写入: {yaml_path}")

    def send_cmd(self, client, cmd, get_pose=True):
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
            self.logger_.info(f"response:{response}")
            return True

        time.sleep(0.1)
        response = client.recv(4096).decode('utf-8')  # 增大接收缓冲区
        self.logger_.info(f'response:{response}')

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
                    self.logger_.error(f"JSON解析错误：{str(e)}")
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

    def process(self):
        """连接机械臂"""
        robot_ip = get_ip()
        self.logger_.info(f'robot_ip:{robot_ip}')
        if robot_ip:
            self.client.connect((robot_ip, 8080))
            socket_command = '{"command":"set_change_work_frame","frame_name":"Base"}'
            self.send_cmd(self.client, socket_command, get_pose = False)
        else:
            popup_message("提醒", "机械臂ip没有ping通")
            sys.exit(1)

        """连接相机"""
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cv2.namedWindow("rgb_image", cv2.WINDOW_NORMAL)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            self.logger_.error_(f"相机连接异常：{e}")
            popup_message("提醒", "相机连接异常")
            sys.exit(1)

        """创建机械臂控制节点"""
        rclpy.init()
        Arm_control_node = CmdPubNode('cmd_pub')
        Arm_control_node.auto_calibration_MoveJ(Joint_init_position[self.arm_init], self.base_type)

        """手眼标定"""
        try:
            self.logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")
            while self.key != 27:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                
                cv_img = cv2.resize(color_image, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
                cv2.imshow("rgb_image", cv_img)  # 窗口显示，显示名为 Capture_Video

                if self.key == ord('s'):  # 若检测到按键 ‘s’，打印字符串
                    socket_command = '{"command": "get_current_arm_state"}'
                    state, pose = self.send_cmd(self.client,socket_command)
                    self.logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
                    if state:
                        filename = os.path.join(self.calibration_data_path, "poses.txt")

                        with open(filename, 'a+') as f:
                            # 将列表中的元素用空格连接成一行
                            pose_ = [str(i) for i in pose]
                            new_line = f'{",".join(pose_)}\n'
                            # 将新行附加到文件的末尾
                            f.write(new_line)

                        image_path = os.path.join(self.calibration_data_path,f"{str(self.cali_img_count)}.jpg")
                        print("image_path:", image_path, "\nself.calibration_data_path:", self.calibration_data_path, "\n")
                        cv2.imwrite(image_path , cv_img)
                        self.logger_.info(f"===采集第{self.cali_img_count}次数据！")
                        self.cali_img_count += 1
                        if self.cali_img_count % 1 == 0:
                            Arm_control_node.auto_calibration_MoveJ()
                elif self.key == ord('c'):
                    self.get_calibration_board()
                    self.save_calibration_results()
                    break

                self.key = cv2.waitKey(30)  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        finally:
            self.logger_.error(f"程序退出！！")
            Arm_control_node.auto_calibration_MoveJ(Joint_init_position[self.arm_init], self.base_type)
            self.close()

    def close(self):
        self.pipeline.stop()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        sys.exit(0)

    def run(self):
        self.process()

if __name__ == "__main__":
    automatic_calibration = AutomaticCalibration(rs.pipeline(), rs.config())
    automatic_calibration.run()