#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import socket
import json
import base64
import time
import struct
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import threading
import math
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

from naviai_udp_comm.msg import TeleoperationUDPRaw
from publish_action import set_target
from record_data import DataRecorder
from interpolators import TrajectoryInterpolator, QuaternionTrajectoryInterpolator


SERVER_IP = "192.168.217.143"
SERVER_PORT = 8000

HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

RECORD_DATA = True


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
        # velocities
        self.observation_velocity_arm_left = [0.0] * 7
        self.observation_velocity_arm_right = [0.0] * 7
        self.observation_velocity_head = [0.0] * 2
        self.observation_velocity_waist = [0.0] * 3
        # force
        self.observation_left_force = [0.0] * 6
        self.observation_right_force = [0.0] * 6
        # tf
        self.chest_in_eye = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.observation_images_head_left_rgb = None
        self.observation_images_head_right_rgb = None
        self.observation_images_wrist_left_rgb = None
        self.observation_images_wrist_right_rgb = None
        self.observation_images_chest_rgb = None

        self.target_publisher = rospy.Publisher("/teleoperation_ctrl_cmd/recv_raw", TeleoperationUDPRaw, queue_size=10)
        self.target_tcp_publisher = [rospy.Publisher(name, Float32MultiArray, queue_size=10) for name in ("left_arm_vel_cmd", "right_arm_vel_cmd")]

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
        self.quat_interpolator = [[QuaternionTrajectoryInterpolator(), QuaternionTrajectoryInterpolator()],
                                  [QuaternionTrajectoryInterpolator(), QuaternionTrajectoryInterpolator()]]
        
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
                               [0.3, -0.65, -1.1, -1.2, 0.8, 0.1, 0]],
                              [[0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                               [0.3, -0.65, -1.1, -1.2, 0.8, 0.1, 0]]]
        startup_hand_target = [[[0.0] * 6, [-1, 1, 0, 0, 0, 0]], [[0.0] * 6, [-1, 1, 0, 0, 0, 0]]]
        startup_waist_target = [[0, 0], [0, 0]]
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
        # joint velocities
        v = msg.velocity
        self.observation_velocity_arm_right = v[0: 7]
        self.observation_velocity_arm_left = v[7: 14]
        self.observation_velocity_head = v[14: 16]
        self.observation_velocity_waist = v[16: 19]

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

    def camera_wrist_left_cb(self, msg: Image) -> None:
        self.observation_images_wrist_left_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_wrist_right_cb(self, msg: Image) -> None:
        self.observation_images_wrist_right_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_chest_cb(self, msg: Image) -> None:
        self.observation_images_chest_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def pack_msg(self) -> str:
        return json.dumps({
            # "head_left_rgb": self.encode_image(self.observation_images_head_left_rgb),
            # "head_right_rgb": self.encode_image(self.observation_images_head_right_rgb),
            "chest_rgb": self.encode_image(self.observation_images_chest_rgb),
            "wrist_right_image": self.encode_image(self.observation_images_wrist_right_rgb),

            "left_tcp_pose_in_chest": self.observation_left_tcp_pose_in_chest,
            "right_tcp_pose_in_chest": self.observation_right_tcp_pose_in_chest,

            "left_arm_joint_angles": self.observation_joints_arm_left,
            "right_arm_joint_angles": self.observation_joints_arm_right,
            "left_hand_joints": self.observation_joints_hand_left,
            "right_hand_joints": self.observation_joints_hand_right,
            "waist_angles": self.observation_joints_waist,
            "neck_angles": self.observation_joints_head,
            # arm joints vel
            "left_arm_joint_velocities": self.observation_velocity_arm_left,
            "right_arm_joint_velocities": self.observation_velocity_arm_right,
            # head & waist joints vel
            "chest_joint_velocities": self.observation_velocity_waist,
            "head_joint_velocities": self.observation_velocity_head,
            # force
            "left_hand_force": self.observation_left_force,
            "right_hand_force": self.observation_right_force,
            # tf
            "chest_in_eye": self.chest_in_eye.tolist(),

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

    def send_msg(self) -> Optional[Dict]:
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
                response = self.send_msg()
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
            if action_length in (13, 28):
                pass
            else:
                rospy.logerr(f"Invalid action length {action_length}")
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
                    "timestamp": float(response["timestamp"]) + float(i) * (1 / 30),
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
            time.sleep((1 / 50) / 1.05)
            self.timer_event.set()

    def blend_actions(self, old_action: Dict[str, Any], new_action: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        return (1 - alpha) * old_action["values"] + alpha * new_action["values"]

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
            if action[0] is not None:
                self.action_interpolator[0].update_actions(action[0])
                last_action = self.action_interpolator[0].get_action(now_time)
                if last_action["values"] is None:
                    last_vals = new_action["values"]
                else:
                    last_vals = last_action["values"]
            else:
                last_vals = new_action["values"]
            # Blend actions
            T = 0.3
            t_diff = now_time - recv_time
            if t_diff <= 0:
                alpha = 0.0
            elif t_diff >= T:
                alpha = 1.0
            else:
                alpha = 0.5 * (1.0 - math.cos(math.pi * (t_diff / T)))
            final_vals = (1.0 - alpha) * last_vals + alpha * new_vals
            # Move robot
            if len(final_vals) == 28:
                self.set_target_using_joint_angles(False, True, True,
                                                #    final_vals[26:28].tolist(),
                                                   [-0.0, 0.0],
                                                   [[0, 0.6, 0.5, -0.05, 0, 0.0, 0], final_vals[0:7].tolist()],
                                                   [[0.0] * 6, final_vals[7:13].tolist()],
                                                   [0, 0.0, 0, 0])
            elif len(final_vals) == 13:
                # self.set_target_using_tcp_pose(False, True, True,
                #                                 [0, 0],
                #                                 # [self.observation_left_tcp_pose_in_chest, [0.35, -0.45, 0.2, -0.34813003552579475, -0.5676744441664578, 0.5052014138136429, 0.5489287160331552]],
                #                                 [self.observation_left_tcp_pose_in_chest, final_vals[0:7].tolist()],
                #                                 [self.observation_joints_hand_left, final_vals[7:13].tolist()],
                #                                 [0, 0, 0, 0])
                final_vals[8] = 1.5
                self.set_target_using_joint_angles(False, True, True,
                                                #    final_vals[26:28].tolist(),
                                                   [-0.0, 0.0],
                                                   [[0, 0.6, 0.5, -0.05, 0, 0.0, 0], final_vals[0:7].tolist()],
                                                   [[0.0] * 6, final_vals[7:13].tolist()],
                                                   [0, 0.0, 0, 0])
                if hasattr(self, "record_data"):
                    self.record_data.record_target_pose(time.time(), [0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                                                        [0.0] * 6, final_vals[0:7].tolist(),
                                                        [0.0] * 6)
            else:
                rospy.logfatal(f"Invalid length: {len(final_vals)}")

    @staticmethod
    def calculate_delta_rotation(a: Rotation, b: Rotation) -> Rotation: # Rotate from a to b
        return b * a.inv()

    def set_target_using_tcp_pose(self, neck_target_valid: bool, arm_target_valid: bool, hand_target_valid: bool,
                                  neck_target: List[float], arm_target: List[List[float]], hand_target: List[List[float]],
                                  waist_target: List[float]) -> None:
        self.target_publisher.publish(set_target(neck_target_valid, False, hand_target_valid,
                                                 neck_target, [[0.0] * 7 for _ in range(2)],
                                                 hand_target, waist_target))
        if arm_target_valid:
            curent_tcp_poses = [self.observation_left_tcp_pose_in_chest, self.observation_right_tcp_pose_in_chest]
            for i in range(2):
                translation_speed = (np.array(arm_target[i][:3]) - np.array(curent_tcp_poses[i][:3])) * 50 * 0.1
                rotation_speed = self.calculate_delta_rotation(Rotation.from_quat(curent_tcp_poses[i][3:]),
                                                               Rotation.from_quat(arm_target[i][3:])).as_rotvec() * 50 * 0.2
                if np.linalg.norm(translation_speed) > 0.1:
                    translation_speed = translation_speed / np.linalg.norm(translation_speed) * 0.1
                if np.linalg.norm(rotation_speed) > np.pi / 6:
                    rotation_speed = rotation_speed / np.linalg.norm(rotation_speed) * np.pi / 6
                target_tcp = Float32MultiArray()
                target_tcp.data = translation_speed.tolist() + rotation_speed.tolist()
                self.target_tcp_publisher[i].publish(target_tcp)

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
