#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import time
import math
import base64
import socket
import struct
import threading
import numpy as np

from typing import List, Optional, Tuple, Dict, Any, Union
from scipy.spatial.transform import Rotation

import rospy                                  # type: ignore
from std_msgs.msg import Float32MultiArray    # type: ignore
from sensor_msgs.msg import Image, JointState # type: ignore
from geometry_msgs.msg import Pose            # type: ignore
from cv_bridge import CvBridge                # type: ignore

from naviai_udp_comm.msg import TeleoperationUDPRaw # type: ignore
from robot_uplimb_pkg.msg import servoL             # type: ignore
from publish_action import set_target
from record_data import DataRecorder
from interpolators import TrajectoryInterpolator, QuaternionTrajectoryInterpolator


HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MOVE_USE_SERVOL = True

class DiffusionInferenceNode:
    def __init__(self):
        self.parse_args()
        assert self.args.chunk_fps > 0
        assert self.args.inter_inference_sleep_time > 0

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
        if MOVE_USE_SERVOL:
            self.target_tcp_publisher = rospy.Publisher("/arm_servol", servoL, queue_size=10)
        else:
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

        self.establish_connection()

        self.run_flag = True
        self.timer_event = threading.Event()
        self.action_lock = threading.Lock()
        self.timer_event.clear()
        self.start_send_time = 0

        self.action_recv_time = [0, 0]
        self.action_list = [None, None] # [Last, New]
        self.action_interpolator = [TrajectoryInterpolator(), TrajectoryInterpolator()]
        self.quat_interpolator = [[QuaternionTrajectoryInterpolator(), QuaternionTrajectoryInterpolator()],
                                  [QuaternionTrajectoryInterpolator(), QuaternionTrajectoryInterpolator()]]

        self.last_control_error = [[np.array([0.0] * 3), np.array([0.0] * 3)], [np.array([0.0] * 3), np.array([0.0] * 3)]]
        self.last_control_pose = [np.array([0.0] * 7), np.array([0.0] * 7)]
        self.control_pose_feedback_new_data_rate = 0.5

        if self.args.record_data:
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
        startup_arm_target = [[[-0.6, 0.55, -1.0, -1.1, 0.4, 0.0, 0],
                               [0.6, -0.55, -1.0, -1.1, 0.4, 0, 0]],
                              [[-0.6, 0.55, -1.0, -1.1, 0.4, 0.0, 0],
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
        if hasattr(self, "record_data"):
            self.record_data.close()

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(description='Naviai Diffusion Policy Server')

        parser.add_argument('--server_ip', type=str, default="120.48.58.215",
                        help='Server IP address')
        parser.add_argument('--server_port', type=int, default=2857,
                        help='Server port number')

        parser.add_argument('--prompt', type=str, default="do something",
                        help='Server port number')

        parser.add_argument('--record_data', action='store_true', default=False,
                        help='Enable data recording')
        parser.add_argument('--no-record_data', dest='record_data', action='store_false',
                        help='Disable data recording')
        parser.add_argument('--chunk_fps', type=int, default=10,
                        help='Frames per second for chunk recording')
        parser.add_argument('--joint_target', action='store_true', default=True,
                        help='Use joint target mode')
        parser.add_argument('--no-joint_target', dest='joint_target', action='store_false',
                        help='Disable joint target mode')
        parser.add_argument('--inter_inference_sleep_time', type=float, default=0.3,
                        help='Sleep time between inferences in seconds')
        parser.add_argument('--control-left', dest='control_left_arm', action='store_true', default=False,
                        help='Enable left mode')
        parser.add_argument('--control-right', dest='control_left_arm', action='store_false',
                            help='Enable right mode')

        self.args = parser.parse_args()

    def establish_connection(self) -> None:
        start_wait = time.time()
        max_wait = 30
        retry_interval = 2
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5)
                self.sock.connect((self.args.server_ip, self.args.server_port))
                rospy.loginfo(f"Connected to server {self.args.server_ip}:{self.args.server_port}")
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
        t = time.time()
        return json.dumps({
            # "head_left_rgb": self.encode_image(self.observation_images_head_left_rgb),
            # "head_right_rgb": self.encode_image(self.observation_images_head_right_rgb),
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

            'timestamp': t,
            'prompt': self.args.prompt
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
        # if not all([self.observation_images_head_left_rgb is not None,
        #             self.observation_images_head_right_rgb is not None,
        #             self.observation_images_chest_rgb is not None]):
        if not all([self.observation_images_wrist_left_rgb is not None,
                    self.observation_images_wrist_right_rgb is not None,
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
            if not actions:
                rospy.logerr(f"Invalid response from server: {response}")
                continue
            action_length = len(actions[0])
            if action_length in (13, 28):
                pass
            else:
                rospy.logerr(f"Invalid action length {action_length}")
            new_action = []
            for i, action in enumerate(actions):
                act_arr = np.asarray(action, dtype=np.float32)
                new_action.append({
                    "timestamp": float(response["timestamp"]) + float(i + 1) * (1 / self.args.chunk_fps),
                    "values": act_arr
                })
            with self.action_lock:
                self.action_list = [self.action_list[1], new_action]
                self.action_recv_time = [self.action_recv_time[1], time.time()]
            if hasattr(self, "record_data"):
                self.record_data.record_network_actions(response["timestamp"], actions)
            time.sleep(self.args.inter_inference_sleep_time)

    def move_robot_timer_thread(self) -> None:
        while self.run_flag:
            time.sleep((1 / 60) / 1.05)
            self.timer_event.set()

    @staticmethod
    def norm_quaternions(quat: Union[List[float], np.ndarray]) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        # assert norm > 0.05, f"Quaternions should be vector of length larger than 0.05, get {norm} from quat {quat}"
        return np.array(quat) / norm

    @staticmethod
    def get_blend_rate(t_diff: float) -> float:
        T = 0.5
        if t_diff <= 0:
            return 0.0
        elif t_diff >= T:
            return 1.0
        else:
            return 0.5 * (1.0 - math.cos(math.pi * (t_diff / T)))

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
                # self.quat_interpolator[1][0].update_actions(action[1], slice(16, 20))
                self.quat_interpolator[1][1].update_actions(action[1], slice(3, 7))
            now_time = time.time()
            new_action = self.action_interpolator[1].get_action(now_time)
            quat_action_new = [{"velocity": np.zeros(3)},
                               self.quat_interpolator[1][1].get_action(now_time)]
                            #    self.quat_interpolator[1][1].get_action(now_time)]
            new_action["values"][3:7] = quat_action_new[1]["values"]
            # new_action["values"][16:20] = quat_action_new[0]["values"]
            quat_velocity_new = [action["velocity"] for action in quat_action_new]
            if (new_action["values"] is None) or (new_action["velocity"] is None):
                rospy.logwarn(f"Get new action: {action[1]}, unable to interpolate.")
                continue
            new_vals = new_action["values"]
            new_velocities = new_action["velocity"]
            quat_velocity_last = [np.zeros(3), np.zeros(3)]
            if action[0] is not None:
                self.action_interpolator[0].update_actions(action[0])
                last_action = self.action_interpolator[0].get_action(now_time)
                quat_action_last = [{"velocity": np.zeros(3)},
                                    self.quat_interpolator[1][1].get_action(now_time)]
                                    # self.quat_interpolator[1][1].get_action(now_time)]
                if quat_action_last[1]["values"] is not None:
                    last_action["values"][3:7] = quat_action_last[1]["values"]
                else:
                    last_action = None
                # last_action["values"][16:20] = quat_action_last[0]["values"]
                quat_velocity_last = [action["velocity"] for action in quat_action_last]
            else:
                last_action = None
            last_vals = last_action["values"] if last_action and last_action["values"] is not None else new_action["values"]
            last_velocities = last_action["velocity"] if last_action and last_action["velocity"] is not None else new_action["velocity"]
            quat_velocity_last = (quat_velocity_new if last_action else quat_velocity_last).copy()
            # Blend actions
            alpha = self.get_blend_rate(now_time - recv_time)
            final_vals = (1.0 - alpha) * last_vals + alpha * new_vals
            final_velocities = (1.0 - alpha) * last_velocities + alpha * new_velocities
            tcp_ori_velocities = [(1.0 - alpha) * quat_vel_last + alpha * quat_vel_new if (quat_vel_last is not None) and (quat_vel_new is not None) else np.zeros(3) \
                for quat_vel_last, quat_vel_new in zip(quat_velocity_last, quat_velocity_new)]
            # Move robot
            if len(final_vals) == 28:
                self.set_target_using_joint_angles(True, True, True,
                                                   [-0.2, 0.1],
                                                   [final_vals[0:7].tolist(), final_vals[13:20].tolist()],
                                                   [final_vals[7:13].tolist(), final_vals[20:26].tolist()],
                                                   [0, 0, 0, 0])
            elif len(final_vals) == 13:
                if self.args.joint_target:
                    if self.args.control_left_arm:
                        self.set_target_using_joint_angles(True, True, True,
                                                   [-0.2, 0.1],
                                                   [final_vals[0:7].tolist(), self.observation_joints_arm_right],
                                                   [final_vals[7:13].tolist(), self.observation_joints_hand_right],
                                                   [0, 0.16263192860378695, 0, 0])
                    else:
                        self.set_target_using_joint_angles(True, True, True,
                                                    [-0.2, 0.1],
                                                    [self.observation_joints_arm_left, final_vals[0:7].tolist()],
                                                    [self.observation_joints_hand_left, final_vals[-6:].tolist()],
                                                    [0, 0.16263192860378695, 0, 0])
                else:
                    if not self.args.control_left_arm:
                        self.set_target_using_tcp_pose(False, True, True,
                                                        [0, 0],
                                                        [self.observation_left_tcp_pose_in_chest, final_vals[0:7].tolist()],
                                                        [self.observation_joints_hand_left, final_vals[7:13].tolist()],
                                                        [0, 0, 0, 0],
                                                        # [[0.0] * 6, list(final_velocities[0:3]) + list(tcp_ori_velocities[0])])
                                                        [[0.0] * 6, [0.0] * 6])
                    else:
                        self.set_target_using_tcp_pose(False, True, True,
                                                        [0, 0],
                                                        [final_vals[0:7].tolist(), self.observation_right_tcp_pose_in_chest],
                                                        [final_vals[7:13].tolist(), self.observation_joints_hand_right],
                                                        [0, 0, 0, 0],
                                                        # [list(final_velocities[0:3]) + list(tcp_ori_velocities[0]), [0.0] * 6])
                                                        [[0.0] * 6, [0.0] * 6])
                    if hasattr(self, "record_data"):
                        self.record_data.record_target_pose(time.time(), self.observation_left_tcp_pose_in_chest,
                                                            [0.0] * 6, final_vals[0:7].tolist(),
                                                            list(final_velocities[0:3]) + list(tcp_ori_velocities[0]))
                        if self.args.joint_target:
                            self.record_data.record_tcp_pose(time.time(), self.observation_joints_arm_left,
                                                                self.observation_joints_arm_right)
                        else:
                            self.record_data.record_tcp_pose(time.time(), self.observation_left_tcp_pose_in_chest,
                                                             self.observation_right_tcp_pose_in_chest)
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
            if MOVE_USE_SERVOL:
                msg = servoL()
                msg.dt = 1 / 60
                msg.leftArmValid = False
                msg.rightArmValid = arm_target_valid
                msg.leftArmTargetPos.x = self.observation_left_tcp_pose_in_chest[0]
                msg.leftArmTargetPos.y = self.observation_left_tcp_pose_in_chest[1]
                msg.leftArmTargetPos.z = self.observation_left_tcp_pose_in_chest[2]
                msg.leftArmTargetQuat.x = self.observation_left_tcp_pose_in_chest[3]
                msg.leftArmTargetQuat.y = self.observation_left_tcp_pose_in_chest[4]
                msg.leftArmTargetQuat.z = self.observation_left_tcp_pose_in_chest[5]
                msg.leftArmTargetQuat.z = self.observation_left_tcp_pose_in_chest[6]
                msg.rightArmTargetPos.x = self.observation_left_tcp_pose_in_chest[0]
                msg.rightArmTargetPos.y = self.observation_right_tcp_pose_in_chest[1]
                msg.rightArmTargetPos.z = self.observation_right_tcp_pose_in_chest[2]
                msg.rightArmTargetQuat.x = self.observation_right_tcp_pose_in_chest[3]
                msg.rightArmTargetQuat.y = self.observation_right_tcp_pose_in_chest[4]
                msg.rightArmTargetQuat.z = self.observation_right_tcp_pose_in_chest[5]
                msg.rightArmTargetQuat.z = self.observation_right_tcp_pose_in_chest[6]
                self.target_tcp_publisher.publish(msg)
            else:
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
