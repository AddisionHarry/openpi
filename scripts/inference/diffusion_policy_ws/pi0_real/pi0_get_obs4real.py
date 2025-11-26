#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# source devel/setup.bash


import rospy
import cv2
import socket
import json
import base64
import tf
import time
import struct
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import threading
import math
from scipy.interpolate import CubicSpline

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, WrenchStamped
from tf2_msgs.msg import TFMessage

from naviai_udp_comm.msg import TeleoperationUDPRaw
from publish_action import set_target
from record_data import DataRecorder

# 120.48.58.215:5381
SERVER_IP = "120.48.58.215"
SERVER_PORT = 5381

HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

RECORD_DATA = True


class TrajectoryInterpolator:
    def __init__(self):
        self.spline = None
        self.t_min = None
        self.t_max = None

    def update_actions(self, actions: List[Dict[str, Any]]):
        if not actions:
            return
        times = np.array([a["timestamp"] for a in actions], dtype=float)   # shape: (N,)
        values = np.array([a["values"] for a in actions], dtype=float)     # shape: (N, D)
        self.spline = CubicSpline(times, values, axis=0)

        self.t_min = times[0]
        self.t_max = times[-1]

    def get_action(self, t_query: float) -> Dict[str, Any]:
        if self.spline is None:
            return {"timestamp": t_query, "values": None}
        t_query = np.clip(t_query, self.t_min, self.t_max)
        values = self.spline(t_query)
        return {"timestamp": t_query, "values": values}


class DiffusionInferenceNode:
    def __init__(self):
        rospy.init_node("diffusion_inference_node", anonymous=True, log_level=rospy.DEBUG)
        self.bridge = CvBridge()
        # images
        self.observation_images_head_left_rgb = None
        self.observation_images_head_right_rgb = None
        self.observation_images_wrist_left_rgb = None
        self.observation_images_wrist_right_rgb = None
        self.observation_images_chest_rgb = None

        # joints positions
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
        # eef
        self.observation_left_tcp_pose_in_chest = [0.0] * 7
        self.observation_right_tcp_pose_in_chest = [0.0] * 7
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

        self.target_publisher = rospy.Publisher("/teleoperation_ctrl_cmd/recv_raw", TeleoperationUDPRaw, queue_size=10)

        # eef poses
        rospy.Subscriber("/right_arm_tcp_pose", Pose, self.right_arm_tcp_pose_cb, queue_size=10)
        rospy.Subscriber("/left_arm_tcp_pose", Pose, self.left_arm_tcp_pose_cb, queue_size=10)
        # arm, head, waist joints positions and velocities
        rospy.Subscriber("/joint_states", JointState, self.joint_states_cb, queue_size=10)
        # hand joints positions
        rospy.Subscriber("/hand_joint_states", JointState, self.hand_joint_states_cb, queue_size=10)
        # images
        rospy.Subscriber("/img/CAM_A/image_raw", Image, self.camera_head_left_cb, queue_size=10)
        rospy.Subscriber("/img/CAM_B/image_raw", Image, self.camera_head_right_cb, queue_size=10)
        rospy.Subscriber("/realsense_up/color/image_raw", Image, self.camera_chest_cb, queue_size=10)
        rospy.Subscriber("/img/left_wrist/image_raw", Image, self.camera_wrist_left_cb, queue_size=10)
        rospy.Subscriber("/img/right_wrist/image_raw", Image, self.camera_wrist_right_cb, queue_size=10)
        # force sensor
        rospy.Subscriber('/force_sensor_extra_6D_left', WrenchStamped, self.left_force_sub_cb, queue_size=10)
        rospy.Subscriber('/force_sensor_extra_6D_right', WrenchStamped, self.right_force_sub_cb, queue_size=10)
        # tf
        rospy.Subscriber("/tf", TFMessage, self.tf_callback)

        # rospy.wait_for_service("robotHandJointSwitch")
        # self.hand_service = rospy.ServiceProxy("robotHandJointSwitch", RobotHandJointSrv)
        # rospy.wait_for_service("left_arm_movel_service")
        # rospy.wait_for_service("right_arm_movel_service")
        # self.movel_service = [rospy.ServiceProxy("left_arm_movel_service", MoveL),
        #                       rospy.ServiceProxy("right_arm_movel_service", MoveL)]

        if RECORD_DATA:
            self.record_data = DataRecorder()

        self.establish_connection()

        self.run_flag = True
        self.timer_event = threading.Event()
        self.action_lock = threading.Lock()
        self.timer_event.clear()
        self.start_send_time = 0

        self.action_recv_time = [0, 0]
        self.action_list = [None, None] # [Last, New]
        self.action_interpolator = [TrajectoryInterpolator(), TrajectoryInterpolator()]

        # Start inference thread
        rospy.loginfo("Start TCP inference thread.")
        threading.Thread(target=self.run_inference_loop, daemon=True).start()
        # Move to startup pose
        startup_neck_target = [[0.0, 0.0], [0.0, 0.0]]
        startup_arm_target = [[[0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                               [1, -1, -1.1, -1.3, 0.15, 0, 0]],
                              [[0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                               [-0.5, -1.2, 0.3, -1.35, 0.75, -0.15, 0]]]
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
        v = msg.velocity

        # joint positions
        self.observation_joints_arm_right = pos[0:7]
        self.observation_joints_arm_left = pos[7:14]
        self.observation_joints_head = pos[14:16]
        self.observation_joints_waist = pos[16:19]

        # joint velocities
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

    def camera_chest_cb(self, msg: Image) -> None:
        self.observation_images_chest_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_wrist_left_cb(self, msg: Image) -> None:
        self.observation_images_wrist_left_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def camera_wrist_right_cb(self, msg: Image) -> None:
        self.observation_images_wrist_right_rgb = cv2.cvtColor(self.image_resize(msg), cv2.COLOR_BGR2RGB)

    def left_force_sub_cb(self, msg):
        # print('left_force_sub_callback')
        msg = msg.wrench
        force = msg.force
        torque = msg.torque

        self.observation_left_force = [force.x, force.y, force.z, torque.x, torque.y, torque.z]

    def right_force_sub_cb(self, msg):
        # print('right_force_sub_callback')
        msg = msg.wrench
        force = msg.force
        torque = msg.torque

        self.observation_right_force = [force.x, force.y, force.z, torque.x, torque.y, torque.z]

    def tf_callback(self, msg):
        try:
            if self.tf_listener.canTransform('CHEST', 'HEAD', rospy.Time(0)):
                (chest_in_head_trans, chest_in_head_rot) = self.tf_listener.lookupTransform('HEAD', 'CHEST',
                                                                                            rospy.Time(0))

                chest_in_head_ros_pose = Pose()
                chest_in_head_ros_pose.position.x = chest_in_head_trans[0] * 1000
                chest_in_head_ros_pose.position.y = chest_in_head_trans[1] * 1000
                chest_in_head_ros_pose.position.z = chest_in_head_trans[2] * 1000
                chest_in_head_ros_pose.orientation.x = chest_in_head_rot[0]
                chest_in_head_ros_pose.orientation.y = chest_in_head_rot[1]
                chest_in_head_ros_pose.orientation.z = chest_in_head_rot[2]
                chest_in_head_ros_pose.orientation.w = chest_in_head_rot[3]
                chest_in_head = self.ros_pose_to_matrix(chest_in_head_ros_pose)

                # TODO
                self.chest_in_eye = chest_in_head
                # self.chest_in_eye[0][3] -= 74.1
                # self.chest_in_eye[1][3] -= 35  # left eye
                # self.chest_in_eye[2][3] -= 71.5

                # tcp_in_base = np.dot(chest_in_base, _tcp_in_chest)

            else:
                rospy.logwarn_throttle(5.0, f"Waiting for transform between CHEST and HEAD")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "Could not compute relative transform.")

    def pack_msg(self) -> str:
        # print("head_left_rgb type:", type(self.observation_images_head_left_rgb))
        # print("head_right_rgb type:", type(self.observation_images_head_right_rgb))
        # print("chest_rgb type:", type(self.observation_images_chest_rgb))
        # print("left_arm_joint_angles type:", type(self.observation_joints_arm_left))
        # print("right_arm_joint_angles type:", type(self.observation_joints_arm_right))
        # print("left_hand_joints type:", type(self.observation_joints_hand_left))
        # print("right_hand_joints type:", type(self.observation_joints_hand_right))
        # print("chest_angles type:", type(self.observation_joints_waist))
        # print("head_angles type:", type(self.observation_joints_head))
        # print("left_arm_joint_velocities type:", type(self.observation_velocity_arm_left))
        # print("right_arm_joint_velocities type:", type(self.observation_velocity_arm_right))
        # print("chest_joint_velocities type:", type(self.observation_velocity_waist))
        # print("head_joint_velocities type:", type(self.observation_velocity_head))
        # print("left_tcp_pose_in_chest type:", type(self.observation_left_tcp_pose_in_chest))
        # print("right_tcp_pose_in_chest type:", type(self.observation_right_tcp_pose_in_chest))
        # print("left_hand_force type:", type(self.observation_left_force))
        # print("right_hand_force type:", type(self.observation_right_force))
        # print("chest_in_eye type:", type(self.chest_in_eye))
        # print("timestamp type:", type(int(time.time() * 1000)))
        return json.dumps({
            # images
            # "head_left_rgb": self.encode_image(self.observation_images_head_left_rgb),
            # "head_right_rgb": self.encode_image(self.observation_images_head_right_rgb),
            "chest_rgb": self.encode_image(self.observation_images_chest_rgb),
            "wrist_right_image": self.encode_image(self.observation_images_wrist_right_rgb),
            # arm joints pos
            "left_arm_joint_angles": self.observation_joints_arm_left,
            "right_arm_joint_angles": self.observation_joints_arm_right,
            # hand joints pos
            "left_hand_joints": self.observation_joints_hand_left,
            "right_hand_joints": self.observation_joints_hand_right,
            # neck and waist joints pos
            "chest_angles": self.observation_joints_waist, # waist
            "head_angles": self.observation_joints_head,
            # arm joints vel
            "left_arm_joint_velocities": self.observation_velocity_arm_left,
            "right_arm_joint_velocities": self.observation_velocity_arm_right,
            # head & waist joints vel
            "chest_joint_velocities": self.observation_velocity_waist,
            "head_joint_velocities": self.observation_velocity_head,
            # eef
            "left_tcp_pose_in_chest": self.observation_left_tcp_pose_in_chest,
            "right_tcp_pose_in_chest": self.observation_right_tcp_pose_in_chest,
            # force
            "left_hand_force": self.observation_left_force,
            "right_hand_force": self.observation_right_force,
            # tf
            # "chest_in_eye": self.chest_in_eye.tolist(),
            # others
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
            print(np.array(actions))
            print(len(actions))
            # observation = response.get("observation")
            if not actions:
                rospy.logerr(f"Invalid response from server: {response}")
                continue
            action_length = len(actions[0])
            # if action_length == 28 and observation:
            #     pass
            # else:
            #     rospy.logerr(f"Invalid action length {action_length} or missing observation")
            new_action = []
            for i, action in enumerate(actions):
                act_arr = np.asarray(action, dtype=np.float32)
                new_action.append({
                    "timestamp": float(response["timestamp"]) + float(i + 1) * (1 / 10),
                    "values": act_arr
                })
            with self.action_lock:
                self.action_list = [self.action_list[1], new_action]
                self.action_recv_time = [self.action_recv_time[1], time.time()]
            if hasattr(self, "record_data"):
                self.record_data.record_network_actions(response["timestamp"], actions)
            time.sleep(1)
            # time.sleep(0.5)

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
                rospy.logwarn("No valid action get.")
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
            T = 0.4
            t_diff = now_time - recv_time
            if t_diff <= 0:
                alpha = 0.0
            elif t_diff >= T:
                alpha = 1.0
            else:
                alpha = 0.5 * (1.0 - math.cos(math.pi * (t_diff / T)))
            final_vals = (1.0 - alpha) * last_vals + alpha * new_vals
            # Move robot
            final_vals[8] = 1.5
            self.target_publisher.publish(set_target(
                                                    False, True, True, action[1][0]["values"][0:2].tolist(),
                                                    # False, False, False,
                                                    #  final_vals[0:2].tolist(),
                                                    #  [-0.2, 0.1],
                                                    #  [final_vals[2:9].tolist(), final_vals[9:16].tolist()],
                                                    #  [final_vals[16:22].tolist(), final_vals[22:28].tolist()],
                                                    # [action[1][-1]["values"][2:9].tolist(), action[1][-1]["values"][9:16].tolist()],
                                                    # [action[1][-1]["values"][16:22].tolist(), action[1][-1]["values"][22:28].tolist()],
                                                    [[0, 0.6, 0.5, -0.05, 0, 0.0, 0], final_vals[0:7].tolist()],
                                                    [[0.0] * 6, final_vals[-6:].tolist()],
                                                     [0, 0.0, 0, 0]))
            if hasattr(self, "record_data"):
                self.record_data.record_target_pose(time.time(), [0, 0.6, 0.5, -0.05, 0, 0.0, 0],
                                                    [0.0] * 6, final_vals[0:7].tolist(),
                                                    [0.0] * 6)



if __name__ == "__main__":
    try:
        DiffusionInferenceNode()
    except rospy.ROSInterruptException:
        pass
