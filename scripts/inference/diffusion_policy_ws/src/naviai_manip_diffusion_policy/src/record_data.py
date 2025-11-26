#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import h5py
import threading
import numpy as np

from datetime import datetime
from typing import List, Optional

class DataRecorder:
    def __init__(self, data_dir: str = "trajectory_data"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(data_dir, f"{timestamp}")
        os.makedirs(self.data_dir, exist_ok=True)

        self.tcp_pose_file = os.path.join(self.data_dir, "tcp_pose_data.csv")
        self.target_pose_file = os.path.join(self.data_dir, "target_pose_data.csv")
        self.control_command_file = os.path.join(self.data_dir, "control_command_data.csv")
        self.network_actions_file = os.path.join(self.data_dir, "network_actions.h5")

        self.init_csv_files()
        self.lock = threading.Lock()
        self.network_actions_buffer = []
        self.buffer_counter = 0

    def init_csv_files(self):
        tcp_headers = ["timestamp"] + \
            [f"left_pos_{c}" for c in "xyz"] + [f"left_quat_{c}" for c in "xyzw"] + \
            [f"right_pos_{c}" for c in "xyz"] + [f"right_quat_{c}" for c in "xyzw"]

        target_headers = ["timestamp"] + \
            [f"left_target_pos_{c}" for c in "xyz"] + [f"left_target_quat_{c}" for c in "xyzw"] + \
            [f"left_target_lin_vel_{c}" for c in "xyz"] + [f"left_target_ang_vel_{c}" for c in "xyz"] + \
            [f"right_target_pos_{c}" for c in "xyz"] + [f"right_target_quat_{c}" for c in "xyzw"] + \
            [f"right_target_lin_vel_{c}" for c in "xyz"] + [f"right_target_ang_vel_{c}" for c in "xyz"]

        control_headers = ["timestamp"] + \
            [f"left_cmd_lin_vel_{c}" for c in "xyz"] + [f"left_cmd_ang_vel_{c}" for c in "xyz"] + \
            [f"left_hand_joint_{i}" for i in range(1, 7)] + \
            [f"right_cmd_lin_vel_{c}" for c in "xyz"] + [f"right_cmd_ang_vel_{c}" for c in "xyz"] + \
            [f"right_hand_joint_{i}" for i in range(1, 7)]

        for file, headers in [
            (self.tcp_pose_file, tcp_headers),
            (self.target_pose_file, target_headers),
            (self.control_command_file, control_headers),
        ]:
            with open(file, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def record_tcp_pose(self, timestamp: float, left_tcp_pose: List[float], right_tcp_pose: List[float]):
        with self.lock, open(self.tcp_pose_file, "a", newline="") as f:
            csv.writer(f).writerow([timestamp] + left_tcp_pose + right_tcp_pose)

    def record_target_pose(self, timestamp: float,
                           left_target_pose: Optional[List[float]] = None,
                           left_target_velocity: Optional[List[float]] = None,
                           right_target_pose: Optional[List[float]] = None,
                           right_target_velocity: Optional[List[float]] = None):
        left_target_pose = left_target_pose or [0.0] * 7
        left_target_velocity = left_target_velocity or [0.0] * 6
        right_target_pose = right_target_pose or [0.0] * 7
        right_target_velocity = right_target_velocity or [0.0] * 6

        with self.lock, open(self.target_pose_file, "a", newline="") as f:
            row = [timestamp] + left_target_pose + left_target_velocity + right_target_pose + right_target_velocity
            csv.writer(f).writerow(row)

    def record_control_command(self, timestamp: float,
                               left_control_velocity: Optional[List[float]] = None,
                               left_hand_joints: Optional[List[float]] = None,
                               right_control_velocity: Optional[List[float]] = None,
                               right_hand_joints: Optional[List[float]] = None):
        left_control_velocity = left_control_velocity or [0.0] * 6
        left_hand_joints = left_hand_joints or [0.0] * 6
        right_control_velocity = right_control_velocity or [0.0] * 6
        right_hand_joints = right_hand_joints or [0.0] * 6

        with self.lock, open(self.control_command_file, "a", newline="") as f:
            row = [timestamp] + list(left_control_velocity) + list(left_hand_joints) + \
                  list(right_control_velocity) + list(right_hand_joints)
            csv.writer(f).writerow(row)

    def record_network_actions(self, timestamp: float, actions: List[np.ndarray] = None):
        actions = actions or []
        with self.lock:
            self.network_actions_buffer.append({"timestamp": timestamp, "actions": actions})
            self.flush_network_actions()

    def flush_network_actions(self):
        if not self.network_actions_buffer:
            return
        with h5py.File(self.network_actions_file, "a") as f:
            for i, action_data in enumerate(self.network_actions_buffer):
                group = f.create_group(f"timestamp_{self.buffer_counter + i}")
                group.attrs["timestamp"] = action_data["timestamp"]
                action_group = group.create_group("left_actions")
                for j, action_matrix in enumerate(action_data["actions"]):
                    action_group.create_dataset(f"action_{j}", data=action_matrix)
        self.buffer_counter += len(self.network_actions_buffer)
        self.network_actions_buffer = []

    def close(self):
        self.flush_network_actions()
