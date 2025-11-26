#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import socket
import json
import base64
import time
import struct
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import threading
import math
import pickle
import os
import csv
import h5py
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

from naviai_udp_comm.msg import TeleoperationUDPRaw # type:ignore
from publish_action import set_target

SERVER_IP = "120.48.58.215"
SERVER_PORT = 1590

HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

RECORD_DATA = True

class DataRecorder:
    def __init__(self, data_dir: str = "trajectory_data"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(data_dir, f"{timestamp}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.tcp_pose_file = os.path.join(self.data_dir, f"tcp_pose_data.csv")
        self.target_pose_file = os.path.join(self.data_dir, f"target_pose_data.csv")
        self.control_command_file = os.path.join(self.data_dir, f"control_command_data.csv")
        self.network_actions_file = os.path.join(self.data_dir, f"network_actions.h5")
        self.init_csv_files()

        self.lock = threading.Lock()
        self.network_actions_buffer = []
        self.buffer_counter = 0

    def init_csv_files(self):
        tcp_headers = [
            'timestamp',
            'left_pos_x', 'left_pos_y', 'left_pos_z',
            'left_quat_x', 'left_quat_y', 'left_quat_z', 'left_quat_w',
            'right_pos_x', 'right_pos_y', 'right_pos_z',
            'right_quat_x', 'right_quat_y', 'right_quat_z', 'right_quat_w'
        ]

        target_headers = [
            'timestamp',
            'left_target_pos_x', 'left_target_pos_y', 'left_target_pos_z',
            'left_target_quat_x', 'left_target_quat_y', 'left_target_quat_z', 'left_target_quat_w',
            'left_target_lin_vel_x', 'left_target_lin_vel_y', 'left_target_lin_vel_z',
            'left_target_ang_vel_x', 'left_target_ang_vel_y', 'left_target_ang_vel_z',
            'right_target_pos_x', 'right_target_pos_y', 'right_target_pos_z',
            'right_target_quat_x', 'right_target_quat_y', 'right_target_quat_z', 'right_target_quat_w',
            'right_target_lin_vel_x', 'right_target_lin_vel_y', 'right_target_lin_vel_z',
            'right_target_ang_vel_x', 'right_target_ang_vel_y', 'right_target_ang_vel_z'
        ]

        control_headers = [
            'timestamp',
            'left_cmd_lin_vel_x', 'left_cmd_lin_vel_y', 'left_cmd_lin_vel_z',
            'left_cmd_ang_vel_x', 'left_cmd_ang_vel_y', 'left_cmd_ang_vel_z',
            'left_hand_joint_1', 'left_hand_joint_2', 'left_hand_joint_3',
            'left_hand_joint_4', 'left_hand_joint_5', 'left_hand_joint_6',
            'right_cmd_lin_vel_x', 'right_cmd_lin_vel_y', 'right_cmd_lin_vel_z',
            'right_cmd_ang_vel_x', 'right_cmd_ang_vel_y', 'right_cmd_ang_vel_z',
            'right_hand_joint_1', 'right_hand_joint_2', 'right_hand_joint_3',
            'right_hand_joint_4', 'right_hand_joint_5', 'right_hand_joint_6'
        ]

        with open(self.tcp_pose_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(tcp_headers)

        with open(self.target_pose_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(target_headers)

        with open(self.control_command_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(control_headers)

    def record_tcp_pose(self, timestamp: float,
                       left_tcp_pose: List[float],
                       right_tcp_pose: List[float]):
        with self.lock:
            with open(self.tcp_pose_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp] + left_tcp_pose + right_tcp_pose
                writer.writerow(row)

    def record_target_pose(self, timestamp: float,
                          left_target_pose: Optional[List[float]] = None,
                          left_target_velocity: Optional[List[float]] = None,
                          right_target_pose: Optional[List[float]] = None,
                          right_target_velocity: Optional[List[float]] = None):
        if left_target_pose is None:
            left_target_pose = [0.0] * 7  # 位置+四元数
        if left_target_velocity is None:
            left_target_velocity = [0.0] * 6  # 线速度+角速度
        if right_target_pose is None:
            right_target_pose = [0.0] * 7
        if right_target_velocity is None:
            right_target_velocity = [0.0] * 6

        with self.lock:
            with open(self.target_pose_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp] + left_target_pose + left_target_velocity + right_target_pose + right_target_velocity
                writer.writerow(row)

    def record_control_command(self, timestamp: float,
                             left_control_velocity: Optional[List[float]] = None,
                             left_hand_joints: Optional[List[float]] = None,
                             right_control_velocity: Optional[List[float]] = None,
                             right_hand_joints: Optional[List[float]] = None):
        if left_control_velocity is None:
            left_control_velocity = [0.0] * 6  # 线速度+角速度
        if left_hand_joints is None:
            left_hand_joints = [0.0] * 6  # 6个手部关节
        if right_control_velocity is None:
            right_control_velocity = [0.0] * 6
        if right_hand_joints is None:
            right_hand_joints = [0.0] * 6

        with self.lock:
            with open(self.control_command_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp] + list(left_control_velocity) + list(left_hand_joints) + list(right_control_velocity) + list(right_hand_joints)
                writer.writerow(row)

    def record_network_actions(self, timestamp: float, actions: List[np.ndarray] = None):
        if actions is None:
            actions = []
        with self.lock:
            action_data = {
                'timestamp': timestamp,
                'actions': actions
            }
            self.network_actions_buffer.append(action_data)
            self.flush_network_actions()

    def flush_network_actions(self):
        if self.network_actions_buffer:
            with h5py.File(self.network_actions_file, 'a') as f:
                for i, action_data in enumerate(self.network_actions_buffer):
                    group = f.create_group(f'timestamp_{self.buffer_counter + i}')
                    group.attrs['timestamp'] = action_data['timestamp']
                    action_group = group.create_group('left_actions')
                    for j, action_matrix in enumerate(action_data['actions']):
                        action_group.create_dataset(f'action_{j}', data=action_matrix)
            self.buffer_counter += len(self.network_actions_buffer)
            self.network_actions_buffer = []

    def close(self):
        self.flush_network_actions()


class TrajectoryInterpolator:
    def __init__(self):
        self.spline = None
        self.t_min = None
        self.t_max = None
        self.t0 = None

    def update_actions(self, actions: List[Dict[str, Any]]):
        if not actions:
            return
        times_abs = np.array([a["timestamp"] for a in actions], dtype=float)
        values = np.array([a["values"] for a in actions], dtype=float)
        self.t0 = times_abs[0]
        times = times_abs - self.t0
        self.spline = CubicSpline(times, values, axis=0)
        self.t_min = times[0]
        self.t_max = times[-1]

    def get_action(self, t_query: float) -> Dict[str, Any]:
        if self.spline is None:
            return {"timestamp": t_query, "values": None}
        t_rel = np.clip(t_query - self.t0, self.t_min, self.t_max)
        values = self.spline(t_rel)
        velocity = self.spline(t_rel, 1)
        return {"timestamp": t_query, "values": values, "velocity": velocity}


class DiffusionInferenceNode:
    def __init__(self):
        rospy.init_node("diffusion_inference_node", anonymous=True, log_level=rospy.DEBUG)
        self.bridge = CvBridge()

        self.observation_left_tcp_pose_in_chest = [0.0] * 7
        self.observation_right_tcp_pose_in_chest = [0.0] * 7
        self.observation_joints_hand_left = [0.0] * 6
        self.observation_joints_hand_right = [0.0] * 6
        self.observation_joints_arm_right = [0.0] * 7
        self.observation_joints_arm_left = [0.0] * 7
        self.observation_joints_head = [0.0] * 2
        self.observation_joints_waist = [0.0] * 3

        self.observation_images_head_left_rgb = None
        self.observation_images_head_right_rgb = None
        self.observation_images_wrist_left_rgb = None
        self.observation_images_wrist_right_rgb = None
        self.observation_images_chest_rgb = None

        self.target_publisher = rospy.Publisher("/teleoperation_ctrl_cmd/recv_raw", TeleoperationUDPRaw, queue_size=10)
        self.target_tcp_publisher = [rospy.Publisher(name, Float32MultiArray, queue_size=10) for name in ("left_arm_vel_cmd", "right_arm_vel_cmd")]
        self.pose_control_error_publisher = [rospy.Publisher(name, Float32MultiArray, queue_size=10) \
            for name in ("/vla_interence_left_tcp_pose_control_error", "/vla_interence_right_tcp_pose_control_error")]

        rospy.Subscriber("/right_arm_tcp_pose", Pose, self.right_arm_tcp_pose_cb, queue_size=10)
        rospy.Subscriber("/left_arm_tcp_pose", Pose, self.left_arm_tcp_pose_cb, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self.joint_states_cb, queue_size=10)
        rospy.Subscriber("/hand_joint_states", JointState, self.hand_joint_states_cb, queue_size=10)
        rospy.Subscriber("/img/CAM_A/image_raw", Image, self.camera_head_left_cb, queue_size=10)
        rospy.Subscriber("/img/CAM_B/image_raw", Image, self.camera_head_right_cb, queue_size=10)
        rospy.Subscriber("/realsense_up/color/image_raw", Image, self.camera_chest_cb, queue_size=10)
        rospy.Subscriber("/img/left_wrist/image_raw", Image, self.camera_wrist_left_cb, queue_size=10)
        rospy.Subscriber("/img/right_wrist/image_raw", Image, self.camera_wrist_right_cb, queue_size=10)

        # rospy.wait_for_service("robotHandJointSwitch")
        # self.hand_service = rospy.ServiceProxy("robotHandJointSwitch", RobotHandJointSrv)
        # rospy.wait_for_service("left_arm_movel_service")
        # rospy.wait_for_service("right_arm_movel_service")
        # self.movel_service = [rospy.ServiceProxy("left_arm_movel_service", MoveL),
        #                       rospy.ServiceProxy("right_arm_movel_service", MoveL)]

        self.establish_connection()

        self.run_flag = True
        self.timer_event = threading.Event()
        self.action_lock = threading.Lock()
        self.timer_event.clear()
        self.start_send_time = 0

        self.action_recv_time = [0, 0]
        self.action_list = [None, None] # [Last, New]
        self.action_interpolator = [TrajectoryInterpolator(), TrajectoryInterpolator()]

        self.last_control_error = [[np.array([0.0] * 3), np.array([0.0] * 3)], [np.array([0.0] * 3), np.array([0.0] * 3)]]
        self.last_control_pose = [np.array([0.0] * 7), np.array([0.0] * 7)]
        self.control_pose_feedback_new_data_rate = 0.5

        if RECORD_DATA:
            self.record_data = DataRecorder()

        # Start inference thread
        rospy.loginfo("Start TCP inference thread.")
        threading.Thread(target=self.run_inference_loop, daemon=True).start()
        # Move to startup pose
        # startup_neck_target = [[-0.1, 0.05], [-0.2, 0.1]]
        # startup_arm_target = [[[0.8, 0.6, 1.3, -1.2, -0.4, 0.3, 0.2],
        #                        [0.62, -0.7, -1.2, -1.25, -0.27, -0.2, 0.05]],
        #                       [[0.85, 0.9, 0.9, -1.8, -0.05, -0.35, -0.15],
        #                        [1.1, -0.6, -1.5, -1.6, -0.05, -0.85, 0.55]]]
        # startup_hand_target = [[[0.0] * 6, [0.0] * 6], [[0.0] * 6, [0.0] * 6]]
        # startup_waist_target = [[0, 0.15], [0, 0.15]]
        startup_neck_target = [[0.0, 0.0], [0.0, 0.0]]
        startup_arm_target = [[[0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                               [0.6, -0.55, -1.0, -1.1, 0.4, 0, 0]],
                              [[0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                               [0.6, -0.55, -1.0, -1.1, 0.4, 0, 0]]]
        startup_hand_target = [[[0.0] * 6, [-1, 1, 0, 0, 0, 0]], [[0.0] * 6, [-1, 1, 0, 0, 0, 0]]]
        startup_waist_target = [[0, 0.0], [0, 0.0]]
        assert len(startup_neck_target) == len(startup_arm_target) and len(startup_arm_target) == len(startup_hand_target) \
            and len(startup_hand_target) == len(startup_waist_target)
        startup_len = len(startup_hand_target)
        for i in range(startup_len):
            rospy.loginfo(f"[{i}/{startup_len}]Preparing the startup actions, Moving to startup pose...")
            for _ in range(25):
                time.sleep(0.1)
                self.target_publisher.publish(set_target(True, True, True,
                                                    startup_neck_target[i], startup_arm_target[i],
                                                    startup_hand_target[i], startup_waist_target[i]))
            time.sleep(1)
        # Start control threads
        rospy.loginfo("Start control thread.")
        threading.Thread(target=self.move_robot_timer_thread, daemon=True).start()
        threading.Thread(target=self.move_robot_thread, daemon=True).start()
        rospy.spin()

    def __del__(self) -> None:
        self.run_flag = False
        if hasattr(self, "sock"):
            self.sock.close()

    def establish_connection(self) -> None:
        start_wait = time.time()
        max_wait = 30
        retry_interval = 2
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5)
                self.sock.connect((SERVER_IP, SERVER_PORT))
                rospy.loginfo(f"Connected to server {SERVER_IP}:{SERVER_PORT}")
                self.sock.settimeout(None)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1<<20)  # 1MB buffer
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1<<20)
                break
            except (ConnectionRefusedError, ConnectionAbortedError, socket.timeout) as e:
                if time.time() - start_wait > max_wait:
                    raise TimeoutError(f"Could not connect within {max_wait} seconds.") from e
                rospy.logwarn(f"Connection failed: {e}, retrying in {retry_interval}s...")
                time.sleep(retry_interval)

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            rospy.logerr(f"[encode_image] Encode failed: {e}")
            return ""

    @staticmethod
    def pose_to_list(msg: Pose) -> List[float]:
        pos, ori = msg.position, msg.orientation
        return [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    @staticmethod
    def joints_slice(msg: JointState, slc: Optional[slice] = None) -> None:
        return msg.position[slc] if slc else msg.position

    @staticmethod
    def list_to_pose(tcp_pose: List[float]) -> Pose:
        pose = Pose()
        if len(tcp_pose) == 7:
            pose.position.x, pose.position.y, pose.position.z = tcp_pose[0:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = tcp_pose[3:7]
        elif len(tcp_pose) == 6:
            pose.position.x, pose.position.y, pose.position.z = tcp_pose[0:3]
            rotvec = np.array(tcp_pose[3:6])
            angle = np.linalg.norm(rotvec)
            if angle < 1e-8:
                q = [0, 0, 0, 1]
            else:
                axis = rotvec / angle
                c = np.cos(angle / 2)
                s = np.sin(angle / 2)
                q = [axis[0]*s, axis[1]*s, axis[2]*s, c]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        else:
            raise ValueError(f"tcp_pose must be length 6 or 7, got {len(tcp_pose)}")
        return pose

    def image_resize(self, msg: Image, size: Tuple[int]=(IMAGE_WIDTH, IMAGE_HEIGHT)) -> np.ndarray:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return cv2.resize(bgr, size)

    def joint_states_cb(self, msg: JointState) -> None:
        pos = msg.position
        self.observation_joints_arm_right = pos[0:7]
        self.observation_joints_arm_left = pos[7:14]
        self.observation_joints_head = pos[14:16]
        self.observation_joints_waist = pos[16:19]

    def right_arm_tcp_pose_cb(self, msg: Pose) -> None:
        self.observation_right_tcp_pose_in_chest = self.pose_to_list(msg)

    def left_arm_tcp_pose_cb(self, msg: Pose) -> None:
        self.observation_left_tcp_pose_in_chest = self.pose_to_list(msg)

    def hand_joint_states_cb(self, msg: JointState) -> None:
        self.observation_joints_hand_left = self.joints_slice(msg, slice(0, 6))
        self.observation_joints_hand_right = self.joints_slice(msg, slice(6, 12))

    def camera_head_left_cb(self, msg: Image) -> None:
        self.observation_images_head_left_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_head_right_cb(self, msg: Image) -> None:
        self.observation_images_head_right_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_chest_cb(self, msg: Image) -> None:
        self.observation_images_chest_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_wrist_left_cb(self, msg: Image) -> None:
        self.observation_images_wrist_left_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_wrist_right_cb(self, msg: Image) -> None:
        self.observation_images_wrist_right_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def pack_msg(self) -> str:
        return json.dumps({
            "head_left_rgb": self.encode_image(self.observation_images_head_left_rgb),
            "head_right_rgb": self.encode_image(self.observation_images_head_right_rgb),
            "wrist_left_image": self.encode_image(self.observation_images_wrist_left_rgb),
            "wrist_right_image": self.encode_image(self.observation_images_wrist_right_rgb),
            "chest_rgb": self.encode_image(self.observation_images_chest_rgb),

            "left_tcp_pose_in_chest": self.observation_left_tcp_pose_in_chest,
            "right_tcp_pose_in_chest": self.observation_right_tcp_pose_in_chest,

            "left_arm_joint_angles": self.observation_joints_arm_left,
            "right_arm_joint_angles": self.observation_joints_arm_right,
            "left_hand_joints": self.observation_joints_hand_left,
            "right_hand_joints": self.observation_joints_hand_right,
            "waist_angles": self.observation_joints_waist,
            "neck_angles": self.observation_joints_head,

            'timestamp': time.time()
        })

    def recv_packet(self) -> dict:
        header = self.recv_all(len(HEADER))
        if header != HEADER:
            raise ValueError(f"Invalid header: {header}")
        length_bytes = self.recv_all(PACK_LEN_INDICATOR_LEN)
        body_len = struct.unpack("!I", length_bytes)[0]
        body_bytes = self.recv_all(body_len)
        # print(f"Real inference time: {(time.time() - self.start_send_time) * 1000}ms")
        return json.loads(body_bytes.decode("utf-8"))

    def inference_remote(self) -> Optional[Dict]:
        if self.sock is None:
            self.establish_connection()
        try:
            body = self.pack_msg().encode("utf-8")
            body_len = len(body)
            length_bytes = struct.pack("!I", body_len)
            packet = HEADER + length_bytes + body
            self.start_send_time = time.time()
            self.sock.sendall(packet)
            pack = self.recv_packet()
            # print(f"Inference time including unpack: {(time.time() - self.start_send_time) * 1000}ms")
            return pack
        except (ConnectionError, BrokenPipeError):
            rospy.logwarn("[Client] Connection lost. Reconnecting...")
            self.sock.close()
            self.establish_connection()
            try:
                self.sock.sendall(packet)
                return self.recv_packet()
            except Exception as e:
                rospy.logerr(f"[Client] Failed after reconnect: {e}")
                return None
        except Exception as e:
            rospy.logerr(f"[Client] Communication failed: {e}")
            return None

    def recv_all(self, length: int) -> bytes:
        buf = b""
        while len(buf) < length:
            chunk = self.sock.recv(length - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            buf += chunk
        return buf

    def call_hand_joint_service(self, is_left: bool, hand_joint: List[float]) -> None:
        try:
            self.hand_service(0 if is_left else 1, hand_joint)
        except rospy.ServiceException as e:
            rospy.logerr(f"Call hand control service failed: {e}")

    def call_tcp_in_chest_service(self, is_left: bool, pose: Pose) -> None:
        try:
            self.movel_service[0 if is_left else 1](pose, True)
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveL failed for {'left' if is_left else 'right'} arm: {e}")

    def check_data_have_received(self) -> bool:
        if not all([self.observation_images_head_left_rgb is not None,
                    self.observation_images_head_right_rgb is not None,
                    self.observation_images_chest_rgb is not None]):
            rospy.logwarn_throttle(1, "Waiting for image data...")
            return False
        if not all([sum(self.observation_right_tcp_pose_in_chest),
                    sum(self.observation_left_tcp_pose_in_chest),
                    sum(self.observation_joints_hand_right),
                    sum(self.observation_joints_hand_left)]):
            rospy.logwarn_throttle(1, "Waiting for robot data...")
            return False
        return True

    def run_inference_loop(self) -> None:
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if not self.check_data_have_received():
                rate.sleep()
                continue
            try:
                t1 = time.time()
                response = self.inference_remote()
                rospy.logdebug(f"Inference cost time: {(time.time() - t1) * 1000}ms")
            except Exception as e:
                rospy.logerr(f"Error during inference: {e}")
                rate.sleep()
                continue
            if not response:
                continue
            actions = response.get("predicted_action")
            observation = response.get("observation")
            if not actions:
                rospy.logerr(f"Invalid response from server: {response}")
                continue
            action_length = len(actions[0])
            if action_length in (13, 28) and observation:
                pass
            else:
                rospy.logerr(f"Invalid action length {action_length} or missing observation")
            new_action = []
            for i, action in enumerate(actions):
                act_arr = np.asarray(action, dtype=np.float32)  # shape (28,)
                # neck = act_arr[26:28] + np.asarray(observation["neck_angles"], dtype=np.float32)
                # left_arm = act_arr[0:7] + np.asarray(observation["left_arm_joint_angles"], dtype=np.float32)
                # right_arm = act_arr[13:20] + np.asarray(observation["right_arm_joint_angles"], dtype=np.float32)
                # left_hand = act_arr[7:13] + np.asarray(observation["left_hand_joints"], dtype=np.float32)
                # right_hand = act_arr[20:26] + np.asarray(observation["right_hand_joints"], dtype=np.float32)
                # values = np.concatenate([neck, left_arm, right_arm, left_hand, right_hand]).astype(np.float32)
                new_action.append({
                    "timestamp": float(response["timestamp"]) + float(i) * (1 / 30) * 3,
                    "values": act_arr
                })
            with self.action_lock:
                self.action_list = [self.action_list[1], new_action]
                self.action_recv_time = [self.action_recv_time[1], time.time()]
            if hasattr(self, "record_data"):
                self.record_data.record_network_actions(response["timestamp"], actions)
            time.sleep(1)

    def move_robot_timer_thread(self) -> None:
        while self.run_flag:
            time.sleep((1 / 60) / 1.05)
            self.timer_event.set()

    def blend_actions(self, old_action: Dict[str, Any], new_action: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        return (1 - alpha) * old_action["values"] + alpha * new_action["values"]

    @staticmethod
    def norm_quaternions(quat: Union[List[float], np.ndarray]) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        # assert norm > 0.05, f"Quaternions should be vector of length larger than 0.05, get {norm} from quat {quat}"
        return np.array(quat) / norm

    def move_robot_thread(self) -> None:
        while self.run_flag:
            self.timer_event.wait()
            self.timer_event.clear()
            # Interpolate actions
            with self.action_lock:
                action = self.action_list.copy()
                recv_time = self.action_recv_time[1]
            if (action[0] is None) and (action[1] is None):
                rospy.logwarn_throttle(1, "No valid action get.")
                continue
            if action[1] is not None:
                self.action_interpolator[1].update_actions(action[1])
            now_time = time.time()
            new_action = self.action_interpolator[1].get_action(now_time)
            if new_action["values"] is None:
                rospy.logwarn(f"Get new action: {action[1]}, unable to interpolate.")
                continue
            new_vals = new_action["values"]
            new_velocities = new_action["velocity"]
            if action[0] is not None:
                self.action_interpolator[0].update_actions(action[0])
                last_action = self.action_interpolator[0].get_action(now_time)
                if last_action["values"] is None:
                    last_vals = new_action["values"]
                    last_velocities = new_action["velocity"]
                else:
                    last_vals = last_action["values"]
                    last_velocities = last_action["velocity"]
            else:
                last_vals = new_action["values"]
                last_velocities = new_action["velocity"]
            # Blend actions
            T = 0.5
            t_diff = now_time - recv_time
            if t_diff <= 0:
                alpha = 0.0
            elif t_diff >= T:
                alpha = 1.0
            else:
                alpha = 0.5 * (1.0 - math.cos(math.pi * (t_diff / T)))
            final_vals = (1.0 - alpha) * last_vals + alpha * new_vals
            final_velocities = (1.0 - alpha) * last_velocities + alpha * new_velocities
            # Move robot
            if len(final_vals) == 28:
                self.set_target_using_joint_angles(True, True, True,
                                                #    final_vals[26:28].tolist(),
                                                   [-0.2, 0.1],
                                                   [final_vals[0:7].tolist(), final_vals[13:20].tolist()],
                                                   [final_vals[7:13].tolist(), final_vals[20:26].tolist()],
                                                   [0, 0, 0, 0])
            elif len(final_vals) == 13:
                last_target_ori = None
                orientation_velocity = self.calculate_delta_rotation(Rotation.from_quat(self.norm_quaternions(last_target_ori)),
                                                               Rotation.from_quat(self.norm_quaternions(final_vals[3:7]))).as_rotvec() / (time.time() - last_tick) * 0.5 \
                                                                   if last_target_ori is not None else np.array([0.0] * 6)
                self.set_target_using_tcp_pose(False, True, True,
                                                [0, 0],
                                                # [self.observation_left_tcp_pose_in_chest, [0.35, -0.45, 0.2, -0.34813003552579475, -0.5676744441664578, 0.5052014138136429, 0.5489287160331552]],
                                                [self.observation_left_tcp_pose_in_chest, final_vals[0:7].tolist()],
                                                [self.observation_joints_hand_left, final_vals[7:13].tolist()],
                                                [0, 0, 0, 0],
                                                [[0.0] * 6, final_velocities[7:10].tolist() + orientation_velocity.tolist()])
                if hasattr(self, "record_data"):
                    self.record_data.record_target_pose(time.time(), self.observation_left_tcp_pose_in_chest,
                                                        [0.0] * 6, final_vals[0:7].tolist(),
                                                        final_velocities[7:10].tolist() + orientation_velocity.tolist())
                    self.record_data.record_tcp_pose(time.time(), self.observation_left_tcp_pose_in_chest, self.observation_right_tcp_pose_in_chest)
                last_tick = time.time()
                last_target_ori = final_vals[3:7]
            else:
                rospy.logfatal(f"Invalid length: {len(final_vals)}")

    @staticmethod
    def calculate_delta_rotation(a: Rotation, b: Rotation) -> Rotation: # Rotate from a to b
        return b * a.inv()

    def set_target_using_tcp_pose(self, neck_target_valid: bool, arm_target_valid: bool, hand_target_valid: bool,
                                  neck_target: List[float], arm_target: List[List[float]], hand_target: List[List[float]],
                                  waist_target: List[float], arm_velocity_target: Optional[List[List[float]]] = None) -> None:
        self.target_publisher.publish(set_target(neck_target_valid, False, hand_target_valid,
                                                 neck_target, [[0.0] * 7 for _ in range(2)],
                                                 hand_target, waist_target))
        if arm_target_valid:
            curent_tcp_poses = [self.observation_left_tcp_pose_in_chest, self.observation_right_tcp_pose_in_chest]
            target_velocities = [None, None]
            if arm_velocity_target is None:
                arm_velocity_target = [[0.0] * 6 for _ in range(2)]
            for i in range(2):
                last_control_error = self.last_control_error[i]
                self.last_control_pose[i] = self.last_control_pose[i] * (1 - self.control_pose_feedback_new_data_rate) + \
                    np.array(curent_tcp_poses[i]) * self.control_pose_feedback_new_data_rate
                translation_error = (np.array(arm_target[i][:3]) - np.array(self.last_control_pose[i][:3]))
                rotation_error = self.calculate_delta_rotation(Rotation.from_quat(self.norm_quaternions(self.last_control_pose[i][3:])),
                                                               Rotation.from_quat(self.norm_quaternions(arm_target[i][3:]))).as_rotvec()
                translation_speed_target = translation_error * 60 * 0.2 + (translation_error - last_control_error[0]) * 60 * 0.0
                rotation_speed_target = rotation_error * 60 * 0.15 + (rotation_error - last_control_error[1]) * 60 * 0.1
                if np.linalg.norm(translation_speed_target) > 0.25:
                    translation_speed_target = translation_speed_target / np.linalg.norm(translation_speed_target) * 0.25
                translation_speed_target += np.array(arm_velocity_target[i][:3])
                if np.linalg.norm(rotation_speed_target) > np.pi / 6:
                    rotation_speed_target = rotation_speed_target / np.linalg.norm(rotation_speed_target) * np.pi / 6
                rotation_speed_target += np.array(arm_velocity_target[i][3:6])
                target_tcp = Float32MultiArray()
                target_tcp.data = translation_speed_target.tolist() + rotation_speed_target.tolist()
                self.target_tcp_publisher[i].publish(target_tcp)
                pose_error = Float32MultiArray()
                pose_error.data = translation_error.tolist() + rotation_error.tolist()
                self.pose_control_error_publisher[i].publish(pose_error)
                self.last_control_error[i] = [translation_error, rotation_error]
                target_velocities[i] = target_tcp.data
            if hasattr(self, "record_data"):
                self.record_data.record_control_command(time.time(), target_velocities[0],
                                                        hand_target[0], target_velocities[1],
                                                        hand_target[1])

    def set_target_using_joint_angles(self, neck_target_valid: bool, arm_target_valid: bool, hand_target_valid: bool,
                                      neck_target: List[float], arm_target: List[List[float]], hand_target: List[List[float]],
                                      waist_target: List[float]) -> None:
        self.target_publisher.publish(set_target(neck_target_valid, arm_target_valid, hand_target_valid,
                                                 neck_target, arm_target, hand_target, waist_target))


if __name__ == "__main__":
    try:
        DiffusionInferenceNode()
    except rospy.ROSInterruptException:
        pass
