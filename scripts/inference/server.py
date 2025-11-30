#!/usr/bin/env python3
"""
Diffusion Policy Inference Server for NAVIAI Humanoid
This script runs a TCP server that receives images and robot states from a client,
runs a trained Diffusion Policy model to predict the next action, and sends the action back.
"""
import traceback
import socket
import json
import base64
import argparse
import time
import struct
from collections import deque
from typing import Tuple, Dict
from openpi.training import config
from openpi.policies import policy_config

import torch
import cv2
import dill
# import hydra
import numpy as np
import sys
# from utils.websocket_client_policy import WebsocketClientPolicy



from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir))

# from diffusion_policy.workspace.train_diffusion_unet_image_isaaclab_workspace import TrainDiffusionUnetImageIsaacLabWorkspace
# from rotation_transformer import RotationTransformer

HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4
TEST_SAVE_IMAGES = False

# ------------------------- Helper Functions -------------------------
def parse_image_from_message(image_msg: str, device: str, image_format='rgb', dtype='float') -> torch.Tensor:
    """
    Decode base64 image message into a torch tensor suitable for model input.
    """
    image_data = base64.b64decode(image_msg)
    np_arr = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_format == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if dtype == 'float':
        image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).to(device).unsqueeze(0).permute(0, 3, 1, 2)
    return image_tensor

def parse_image_from_message_np(image_msg: str, device: str, image_format='rgb', dtype='float') -> torch.Tensor:
    """
    Decode base64 image message into a torch tensor suitable for model input.
    """
    # base64 -> bytes -> np.uint8 buffer -> BGR
    image_data = base64.b64decode(image_msg)
    np_arr = np.frombuffer(image_data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode base64 image")
    if image_format == 'rgb':
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.uint8)

def parse_pose_from_message(pose_msg: list, device: str, w_first=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pose list [pos + quat] into torch tensors (pos, quat).
    Supports w-first or w-last quaternion formats.
    """
    pos = torch.tensor(pose_msg[:3], dtype=torch.float32, device=device)
    if w_first:
        quat = torch.tensor(pose_msg[3:7], dtype=torch.float32, device=device)
    else:
        quat = torch.tensor([pose_msg[6], pose_msg[3], pose_msg[4], pose_msg[5]], dtype=torch.float32, device=device)
    return pos.unsqueeze(0), quat.unsqueeze(0)

def stack_last_n_obs_dict(all_obs: deque, n_steps: int) -> dict:
    """
    Stack the last n_steps of observations into a single batch dictionary.
    Pads with the earliest observation if fewer than n_steps available.
    """
    assert len(all_obs) > 0
    # all_obs = list(all_obs)
    obs_list = list(all_obs)
    result = {
        key: torch.zeros(
            list(obs_list[-1][key].shape)[0:1] + [n_steps] + list(obs_list[-1][key].shape)[1:],
            dtype=obs_list[-1][key].dtype,
        ).to(obs_list[-1][key].device)
        for key in obs_list[-1]
    }
    start_idx = -min(n_steps, len(obs_list))
    for key in obs_list[-1]:
        result[key][:, start_idx:] = torch.cat([obs[key][:, None] for obs in obs_list[start_idx:]], dim=1)
        if n_steps > len(obs_list):
            if start_idx == -1:
                result[key][:, :start_idx] = result[key][:, start_idx:]
            else:
                result[key][:, :start_idx] = result[key][:, start_idx:start_idx + 1]
    return result

# ------------------------- Argument Parsing -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Policy Server for NAVIAI Humanoid")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Server port to listen")
    parser.add_argument("--config_name", type=str, default="pi05_zjhumanoid_grasp_can", help="Target configuration name")
    parser.add_argument("--model_path", type=str, required=True, default="/root/workspace/openpi/checkpoints/pi0_grasp_chips_217_10hz/my_experiment/13000",
                        help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="data/eval_naviai", help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (cpu or cuda)")
    return parser.parse_args()

def recv_all(conn: socket.socket, length: int) -> bytes:
    buf = b""
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        buf += chunk
    return buf

def recv_packet(conn: socket.socket) -> dict:
    header = recv_all(conn, len(HEADER))
    if header != HEADER:
        raise ValueError(f"Invalid header: {header}")
    length_bytes = recv_all(conn, PACK_LEN_INDICATOR_LEN)
    body_len = struct.unpack("!I", length_bytes)[0]
    body_bytes = recv_all(conn, body_len)
    return json.loads(body_bytes.decode("utf-8"))

def unparse_observation(message: Dict, device: str) -> Dict:
    # parse image
    # head_right_image = parse_image_from_message_np(message['head_right_rgb'], device, image_format='rgb', dtype='float')
    # head_left_image = parse_image_from_message_np(message['head_left_rgb'], device, image_format='rgb', dtype='float')
    chest_image = parse_image_from_message_np(message['chest_rgb'], device, image_format='rgb', dtype='float')
    right_wrist_image = parse_image_from_message_np(message['wrist_right_image'], device, image_format='rgb', dtype='float')
    left_wrist_image = parse_image_from_message_np(message['wrist_left_image'], device, image_format='rgb', dtype='float')
    # arm joints pos
    left_arm_joints_pos = np.array(message['left_arm_joint_angles']).reshape(-1)
    right_arm_joints_pos = np.array(message['right_arm_joint_angles']).reshape(-1)
    # hand joints pos
    left_hand_joints_pos = np.array(message['left_hand_joints']).reshape(-1)
    right_hand_joints_pos = np.array(message['right_hand_joints']).reshape(-1)
    # head & waist joints pos
    # neck_joints_pos = np.array(message['head_angles']).reshape(-1)
    waist_joints_pos = np.array(message['waist_angles']).reshape(-1)[:2]
    # arm joints velocity
    # left_arm_joints_vel = np.array(message['left_arm_joint_velocities']).reshape(-1)
    # right_arm_joints_vel = np.array(message['right_arm_joint_velocities']).reshape(-1)
    # head & waist joints velocity
    # neck_joints_vel = np.array(message['head_joint_velocities']).reshape(-1)
    # waist_joints_vel = np.array(message['chest_joint_velocities']).reshape(-1)
    # eef
    left_tcp_pose_in_chest = np.array(message['left_tcp_pose_in_chest']).reshape(-1)
    right_tcp_pose_in_chest = np.array(message['right_tcp_pose_in_chest']).reshape(-1)
    # forces
    # left_hand_force = np.array(message['left_hand_force']).reshape(-1)
    # right_hand_force = np.array(message['right_hand_force']).reshape(-1)
    # tcp chest in eye
    # chest_in_eye = np.array(message['chest_in_eye']).reshape(-1)

    timestamp = message['timestamp']

    return {
        "observation/images/chest_rgb": chest_image,
        "observation/images/left_wrist_rgb": left_wrist_image,
        "observation/images/right_wrist_rgb": right_wrist_image,
        "observation/end_effector/left_tcp": left_tcp_pose_in_chest,
        "observation/end_effector/right_tcp": right_tcp_pose_in_chest,
        "observation/left_arm_joint_position": left_arm_joints_pos,
        "observation/right_arm_joint_position": right_arm_joints_pos,
        "observation/left_hand_joint_position": left_hand_joints_pos,
        "observation/right_hand_joint_position": right_hand_joints_pos,
        "observation/waist_joint_position": waist_joints_pos,
        "prompt":"do something",
        "timestamp": timestamp,
    }

def pad_to_dim(x, dim):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] >= dim:
        return x[..., :dim]
    pad = np.zeros((dim - x.shape[-1],), dtype=np.float32)
    return np.concatenate([x, pad], axis=-1)



def main():
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    args = parse_args()
    # pi0 fast
    cfg = config.get_config(args.config_name)
    print("Start loading model from checkpoint.")
    policy = policy_config.create_trained_policy(cfg, args.model_path)
    assert policy is not None
    print("Loaded model from checkpoint...")
    # Save the images to check whether the TCP works correctly
    if TEST_SAVE_IMAGES:
        image_save_dir = Path(args.output_dir) / "saved_images"
        image_save_dir.mkdir(parents=True, exist_ok=True)
        saved_image_count = 0
    # Inference
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((args.host, args.port))
        s.listen(1)
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, addr = s.accept()
            print(f"[Server] Connected by {addr}")
            try:
                while True:
                    message = recv_packet(conn)
                    process_start_time = time.time()
                    if not message:
                        continue
                    try:
                        with torch.inference_mode():
                            # update new observation
                            obs = unparse_observation(message, args.device)
                            if TEST_SAVE_IMAGES and (saved_image_count < 10):
                                img = obs['right_image'][0].permute(1, 2, 0).cpu().numpy() * 255
                                img = img.astype(np.uint8)
                                img_path = image_save_dir / f"image_{saved_image_count+1:02d}.png"
                                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                                saved_image_count += 1
                            # Model inference
                            inference_start = time.time()
                            # action_dict = policy.predict_action(nobs)
                            action_dict = policy.infer(obs)

                            print(f"Time taken for diffusion policy inference: {(time.time() - inference_start) * 1000}ms")
                            # pack actions to response
                            real_actions = action_dict['actions']
                            response = json.dumps({
                                'predicted_action': real_actions.tolist(),
                                "timestamp": message["timestamp"]
                            })
                            body = response.encode("utf-8")
                            body_len = len(body)
                            length_bytes = struct.pack("!I", body_len)
                            packet = HEADER + length_bytes + body
                            conn.sendall(packet)
                    except Exception as e:
                        print(traceback.format_exc())
                        print("Error processing data:", e)
                    # print(f"Time taken for total server data process: {(time.time() - process_start_time) * 1000}ms")
            finally:
                conn.close()
                print(f"[Server] Connection from {addr} closed.")
                continue


if __name__ == '__main__':
    main()
