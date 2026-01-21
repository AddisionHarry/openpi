#!/usr/bin/env python3

import ast
import socket
import json
import argparse
import time
import struct
import pandas as pd
import numpy as np
from pathlib import Path


HEADER = b"NAVIAI_DIFFUSION_POLICY"
PACK_LEN_INDICATOR_LEN = 4

ACTION_IDX = {
    "left_arm_tcp":   (38, 45),
    "right_arm_tcp":  (45, 52),
    # "left_arm_joint": (7, 14),
    # "right_arm_joint":(0, 7),
    "left_arm_joint": (0, 7),
    "right_arm_joint":(7, 14),
    "left_hand":      (52, 58),
    "right_hand":     (58, 64),
    "waist":          (17, 19),
}

def parse_args():
    def parse_bool(s):
        return s.lower() in ("true", "1", "yes")

    parser = argparse.ArgumentParser(description="Pi Test Inference Server for NAVIAI Humanoid")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Server port to listen")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size to send each time")
    parser.add_argument("--dataset-action-fps", type=int, default=30, help="Action FPS of the dataset")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory to read dataset from")
    parser.add_argument("--episode-index", type=int, default=0, help="Default episode index to load")
    parser.add_argument("--use-arms", type=ast.literal_eval, default="[False, True]", help="Example: \"[False, True]\"")
    parser.add_argument("--use-waist-angles", type=parse_bool, default=False)
    parser.add_argument("--use-tcp-pose", type=parse_bool, default=False)
    return parser.parse_args()

def recv_packet(conn: socket.socket) -> dict:
    def recv_all(conn: socket.socket, length: int) -> bytes:
        buf = b""
        while len(buf) < length:
            chunk = conn.recv(length - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            buf += chunk
        return buf
    header = recv_all(conn, len(HEADER))
    if header != HEADER:
        raise ValueError(f"Invalid header: {header}")
    length_bytes = recv_all(conn, PACK_LEN_INDICATOR_LEN)
    body_len = struct.unpack("!I", length_bytes)[0]
    body_bytes = recv_all(conn, body_len)
    return json.loads(body_bytes.decode("utf-8"))

def load_action_names(dataset_root: str):
    info_path = Path(dataset_root) / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    return info["features"]["actions"]["names"]

def load_single_episode(dataset_root: str, episode_idx: int):
    root = Path(dataset_root)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    ep_file = f"episode_{episode_idx:06d}.parquet"
    matched = list(data_dir.glob(f"chunk-*/*{ep_file}"))
    if len(matched) == 0:
        raise FileNotFoundError(f"Episode file not found: {ep_file}")
    if len(matched) > 1:
        print("[Warning] Multiple files match episode index, use the first one.")
    pq = matched[0]
    print(f"[Dataset] Loading episode: {pq}")
    df = pd.read_parquet(pq)
    # Detect action columns
    if "action" in df.columns:
        acts = np.stack(df["action"].values)
    elif "actions" in df.columns:
        acts = np.stack(df["actions"].values)
    else:
        raise KeyError(f"Neither 'action' nor 'actions' exists in {pq}")
    print(f"[Dataset] Episode {episode_idx} length: {len(acts)} steps")
    return acts

def pack_real_action(action, use_arms, use_waist_angles, use_tcp_pose):
    segments = []
    if use_arms[0]:
        seg = "left_arm_tcp" if use_tcp_pose else "left_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.append(action[s:e])
        s, e = ACTION_IDX["left_hand"]
        segments.append(action[s:e])
    if use_arms[1]:
        seg = "right_arm_tcp" if use_tcp_pose else "right_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.append(action[s:e])
        s, e = ACTION_IDX["right_hand"]
        segments.append(action[s:e])
    if not use_arms[0] and not use_arms[1]:
        raise ValueError("At least one arm must be used.")
    if use_waist_angles:
        s, e = ACTION_IDX["waist"]
        segments.append(action[s:e])
    return np.concatenate(segments, axis=0).tolist()

def get_action_segment_names(action_names, use_arms, use_waist_angles, use_tcp_pose):
    segments = []
    if use_arms[0]:
        seg = "left_arm_tcp" if use_tcp_pose else "left_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.extend(action_names[s:e])
        s, e = ACTION_IDX["left_hand"]
        segments.extend(action_names[s:e])
    if use_arms[1]:
        seg = "right_arm_tcp" if use_tcp_pose else "right_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.extend(action_names[s:e])
        s, e = ACTION_IDX["right_hand"]
        segments.extend(action_names[s:e])
    if use_waist_angles:
        s, e = ACTION_IDX["waist"]
        segments.extend(action_names[s:e])
    return segments


def main():
    args = parse_args()

    episode_actions = load_single_episode(args.dataset_dir, args.episode_index)
    episode_len = len(episode_actions)
    segment_names = get_action_segment_names(
        load_action_names(args.dataset_dir),
        args.use_arms,
        args.use_waist_angles,
        args.use_tcp_pose
    )
    print(f"Packing data with the following settings: {segment_names}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((args.host, args.port))
        s.listen(1)
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, addr = s.accept()
            print(f"[Server] Connected by {addr}")
            base_timestamp = None
            last_recv_walltime = time.time()
            try:
                conn.settimeout(0.1)
                while True:
                    message = None
                    try:
                        message = recv_packet(conn)
                    except socket.timeout:
                        continue
                    recv_time = time.time()
                    time_since_last = recv_time - last_recv_walltime
                    last_recv_walltime = recv_time
                    if (time_since_last >= 10.0) or (base_timestamp is None):
                        print(f"[Server] {time_since_last:.1f}s no message, sending first frame actions.")
                        base_timestamp = "wait for startup"
                        action = pack_real_action(episode_actions[0], args.use_arms, args.use_waist_angles, args.use_tcp_pose)
                        actions_list = [action] * args.chunk_size
                        response = json.dumps({
                            "predicted_action": actions_list,
                            "timestamp": message["timestamp"],
                            "slow_move": True
                        })
                        print("[Server] Set to slow move.")
                        body = response.encode("utf-8")
                        conn.sendall(HEADER + struct.pack("!I", len(body)) + body)
                        continue
                    elif isinstance(base_timestamp, str):
                        print("[Server] Slow move finished.")
                        base_timestamp = message["timestamp"]
                    current_ts = message["timestamp"]
                    dt = current_ts - base_timestamp
                    idx = int(np.ceil(dt * args.dataset_action_fps))
                    if idx < 0:
                        idx = 0
                    window = []
                    for i in range(args.chunk_size):
                        real_idx = idx + i
                        if real_idx < episode_len:
                            action = pack_real_action(episode_actions[real_idx], args.use_arms, args.use_waist_angles, args.use_tcp_pose)
                            window.append(action)
                        else:
                            action = pack_real_action(episode_actions[-1], args.use_arms, args.use_waist_angles, args.use_tcp_pose)
                            window.append(action)
                    response = json.dumps({
                        "predicted_action": window,
                        "timestamp": message["timestamp"],
                        "slow_move": False
                    })
                    body = response.encode("utf-8")
                    conn.sendall(HEADER + struct.pack("!I", len(body)) + body)
            finally:
                conn.close()
                print(f"[Server] Connection from {addr} closed.")
                continue


if __name__ == '__main__':
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    main()
