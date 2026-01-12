#!/usr/bin/env python3
"""
Script: test_eval_dataset.py
Description:
    Evaluate a trained OpenPI policy on one or multiple episodes from a dataset.
    Supports saving episode data and predicted actions into H5 format, including
    stepwise MSE computation for each episode. Supports single episode evaluation
    (--episode-index) or batch evaluation of all episodes (--episode-all).

Usage Example:
    DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 uv run python3 eval_dataset.py \
        --dataset-dir <path/to/dataset> \
        --episode-index 0 \
        --model-path <path/to/checkpoint> \
        --config-name my_config \
        --output-path <path/to/output.h5> \
        --device cuda:0 \
        --use-arms "[False, True]" --use-waist-angles False --use-tcp-pose False
"""

import argparse
import ast
import json
from math import floor
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import cv2
import h5py
from openpi.training import config
from openpi.policies import policy_config
from openpi.policies.zj_humanoid_policy import make_zj_humanoid_example

ACTION_IDX = {
    "left_arm_tcp":   (38, 45),
    "right_arm_tcp":  (45, 52),
    # "left_arm_joint": (7, 14),
    # "right_arm_joint":(0, 7),
    "left_arm_joint": (0, 7),
    "right_arm_joint":(7, 14),
    "left_hand":      (52, 58),
    "right_hand":     (58, 64),
    "waist":          (16, 18),
}

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def parse_args():
    def parse_bool(s):
        return s.lower() in ("true", "1", "yes")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--episode-index", type=int, help="Single episode index to process")
    parser.add_argument("--episode-all", action="store_true", help="Process all episodes")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--inference-res-freq", type=float, default=30.0, help="Inference resolution frequency (Hz)")
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="do something")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--output-path", type=str, default="output.h5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-arms", type=ast.literal_eval, default="[False, True]", help="Example: \"[False, True]\"")
    parser.add_argument("--use-waist-angles", type=parse_bool, default=False)
    parser.add_argument("--use-tcp-pose", type=parse_bool, default=False)
    return parser.parse_args()

class EpisodeDataIterator:
    def __init__(self, dataset_root: str, episode_idx: int, batch_size: int = 1, cache_videos: bool = False):
        self.dataset_root = Path(dataset_root)
        self.episode_idx = episode_idx
        self.cache_videos = cache_videos
        self.batch_size = batch_size
        assert batch_size >= 1, "Batch size must be at least 1."

        # Load parquet data
        self.episode_file = self._find_episode_file()
        self.episode_df = pq.ParquetFile(self.episode_file).read().to_pandas()
        self.episode_len = len(self.episode_df)

        # Find video files
        self.video_paths = self._find_video_files()
        self.video_caps = {}
        self.video_frames = {}  # only used if cache_videos=True

        if self.cache_videos:
            self._cache_all_video_frames()

        self.current_step = 0

    def _find_episode_file(self) -> Path:
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
        ep_file_pattern = f"episode_{self.episode_idx:06d}.parquet"
        matched = list(data_dir.glob(f"chunk-*/*{ep_file_pattern}"))
        if not matched:
            raise FileNotFoundError(f"Episode file not found: {ep_file_pattern}")
        if len(matched) > 1:
            print("[Warning] Multiple files match episode index, use the first one.")
        return matched[0]

    def _find_video_files(self) -> dict:
        video_root = self.dataset_root / "videos"
        if not video_root.exists():
            raise FileNotFoundError(f"Video root folder not found: {video_root}")

        video_paths = {}
        ep_video_pattern = f"episode_{self.episode_idx:06d}.mp4"

        for chunk_dir in video_root.glob("chunk-*"):
            for cam_dir in chunk_dir.glob("observation.images.*"):
                cam_name = cam_dir.name.replace("observation.images.", "")
                video_file = cam_dir / ep_video_pattern
                if video_file.exists():
                    video_paths[cam_name] = str(video_file)
                else:
                    raise FileNotFoundError(f"Video file not found for {cam_name}: {video_file}")

        if not video_paths:
            raise FileNotFoundError(f"No video files found for episode {self.episode_idx:06d}")
        return video_paths

    def _cache_all_video_frames(self):
        # tqdm.write("[Info] Caching all video frames into memory...")
        for cam_name, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
                frames.append(frame)
            cap.release()
            self.video_frames[cam_name] = frames
            # tqdm.write(f"[Info] Cached {len(frames)} frames for camera {cam_name}")

    def _get_video_frame(self, cam_name: str, step: int) -> np.ndarray:
        if self.cache_videos:
            return self.video_frames[cam_name][step]

        if cam_name not in self.video_caps:
            cap = cv2.VideoCapture(self.video_paths[cam_name])
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.video_paths[cam_name]}")
            self.video_caps[cam_name] = cap

        cap = self.video_caps[cam_name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {step} from {cam_name} video")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    def _read_single_step(self, step: int) -> dict:
        if step < 0 or step >= self.episode_len:
            raise StopIteration(f"Step {step} out of range (max: {self.episode_len-1})")

        step_data = self.episode_df.iloc[step].to_dict()
        for cam_name in self.video_paths.keys():
            step_data[f"observation.images.{cam_name}"] = self._get_video_frame(cam_name, step)
        return step_data

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self) -> dict:
        if self.batch_size == 1:
            if self.current_step >= self.episode_len:
                for cap in self.video_caps.values():
                    cap.release()
                self.video_caps.clear()
                raise StopIteration
            step_data = self._read_single_step(self.current_step)
            self.current_step += 1
            return step_data
        else:
            if self.current_step >= self.episode_len:
                for cap in self.video_caps.values():
                    cap.release()
                self.video_caps.clear()
                raise StopIteration
            batch = []
            for _ in range(self.batch_size):
                if self.current_step >= self.episode_len:
                    break
                batch.append(self._read_single_step(self.current_step))
                self.current_step += 1
            if not batch:
                raise StopIteration
            return batch

    def __len__(self) -> int:
        return self.episode_len

    def __del__(self):
        for cap in self.video_caps.values():
            cap.release()


def unparse_observation(obs: dict, device: str, prompt: str) -> dict:
    head_image = np.array(obs['observation.images.head_rgb']).astype(np.uint8)
    chest_image = np.array(obs['observation.images.chest_rgb']).astype(np.uint8)
    right_wrist_image = np.array(obs['observation.images.right_wrist_rgb']).astype(np.uint8)
    left_wrist_image = np.array(obs['observation.images.left_wrist_rgb']).astype(np.uint8)

    left_arm_joints_pos = np.array(obs['observation.state'][slice(*ACTION_IDX["left_arm_joint"])], dtype=np.float32).reshape(-1)
    right_arm_joints_pos = np.array(obs['observation.state'][slice(*ACTION_IDX["right_arm_joint"])], dtype=np.float32).reshape(-1)
    left_hand_joints_pos = np.array(obs['observation.state'][slice(*ACTION_IDX["left_hand"])], dtype=np.float32).reshape(-1)
    right_hand_joints_pos = np.array(obs['observation.state'][slice(*ACTION_IDX["right_hand"])], dtype=np.float32).reshape(-1)
    waist_joints_pos = np.array(obs['observation.state'][slice(*ACTION_IDX["waist"])], dtype=np.float32).reshape(-1)
    left_tcp_pose_in_chest = np.array(obs['observation.state'][slice(*ACTION_IDX["left_arm_tcp"])], dtype=np.float32).reshape(-1)
    right_tcp_pose_in_chest = np.array(obs['observation.state'][slice(*ACTION_IDX["right_arm_tcp"])], dtype=np.float32).reshape(-1)

    return {
        "observation/images/head_rgb": head_image,
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
        "prompt": prompt,
    }

def pack_real_action(action: np.ndarray, use_arms: tuple[bool, bool], use_waist_angles: bool, use_tcp_pose: bool, axis=-1) -> np.ndarray:
    def slice_axis(arr, start, end, axis):
        # Construct slicing object for arbitrary axis
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(start, end)
        return arr[tuple(slicer)]

    segments = []
    if use_arms[0]:
        seg = "left_arm_tcp" if use_tcp_pose else "left_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.append(slice_axis(action, s, e, axis))
        s, e = ACTION_IDX["left_hand"]
        segments.append(slice_axis(action, s, e, axis))
    if use_arms[1]:
        seg = "right_arm_tcp" if use_tcp_pose else "right_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.append(slice_axis(action, s, e, axis))
        s, e = ACTION_IDX["right_hand"]
        segments.append(slice_axis(action, s, e, axis))
    if not use_arms[0] and not use_arms[1]:
        raise ValueError("At least one arm must be used.")
    if use_waist_angles:
        s, e = ACTION_IDX["waist"]
        segments.append(slice_axis(action, s, e, axis))
    return np.concatenate(segments, axis=axis).astype(np.float32)

def load_action_names_from_meta(dataset_dir: str) -> list[str]:
    meta_path = Path(dataset_dir) / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta info.json not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["features"]["actions"]["names"]

def load_dataset_frequency_from_meta(dataset_dir: str) -> float:
    meta_path = Path(dataset_dir) / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta info.json not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["fps"]

def pack_action_joint_names(
    action_names: list[str],
    use_arms: tuple[bool, bool],
    use_waist_angles: bool,
    use_tcp_pose: bool,
) -> list[str]:
    def slice_names(names, start, end):
        return names[start:end]
    segments = []
    if use_arms[0]:
        seg = "left_arm_tcp" if use_tcp_pose else "left_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.extend(slice_names(action_names, s, e))
        s, e = ACTION_IDX["left_hand"]
        segments.extend(slice_names(action_names, s, e))
    if use_arms[1]:
        seg = "right_arm_tcp" if use_tcp_pose else "right_arm_joint"
        s, e = ACTION_IDX[seg]
        segments.extend(slice_names(action_names, s, e))
        s, e = ACTION_IDX["right_hand"]
        segments.extend(slice_names(action_names, s, e))
    if not use_arms[0] and not use_arms[1]:
        raise ValueError("At least one arm must be used.")
    if use_waist_angles:
        s, e = ACTION_IDX["waist"]
        segments.extend(slice_names(action_names, s, e))
    return segments

def compute_stepwise_action_mse(all_pred_actions: np.ndarray, all_gt_actions: np.ndarray, chunk_length: int, resample_ratio: int = 1) -> np.ndarray:
    episode_len = all_gt_actions.shape[0]
    mse_per_step = np.zeros(episode_len, dtype=np.float32)
    for step in range(episode_len):
        valid_len = max(min(chunk_length, floor((episode_len - step) / resample_ratio)), 1)
        pred_chunk = all_pred_actions[step, :valid_len, :]
        gt_chunk = all_gt_actions[step:step + valid_len * resample_ratio:resample_ratio, 0, :]
        mse_per_step[step] = float(np.mean((pred_chunk - gt_chunk) ** 2))
    return mse_per_step

def main():
    args = parse_args()
    if args.episode_index is None and not args.episode_all:
        raise ValueError("Must specify at least --episode-index or --episode-all")

    if args.episode_all:
        data_dir = Path(args.dataset_dir) / "data"
        all_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        all_episode_indices = sorted({int(f.name.split("_")[1].split(".")[0]) for f in all_files})
    else:
        all_episode_indices = [args.episode_index]

    dataset_freq = load_dataset_frequency_from_meta(args.dataset_dir)
    resample_ratio = dataset_freq / args.inference_res_freq
    assert (resample_ratio >= 1.0) and (abs(resample_ratio - round(resample_ratio)) < 1e-3), "Inference frequency must be lower than or equal to dataset frequency by an integer factor."
    resample_ratio = round(resample_ratio)
    print(f"[Info] Dataset frequency: {dataset_freq} Hz, Inference frequency: {args.inference_res_freq} Hz, Resample ratio: {resample_ratio:.4f}")

    all_action_names = load_action_names_from_meta(args.dataset_dir)
    joint_names = pack_action_joint_names(
        all_action_names, args.use_arms, args.use_waist_angles, args.use_tcp_pose)

    pack_action = lambda x: pack_real_action(x, args.use_arms, args.use_waist_angles, args.use_tcp_pose)

    # Load model
    cfg = config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, args.model_path)
    print("[Model] Loaded trained policy.")
    first_obs = make_zj_humanoid_example(cfg.data.use_arms, cfg.data.use_tcp_pose,
                                         cfg.data.use_wrist_cameras, cfg.data.obs_use_waist_angles)
    first_data = policy.infer(first_obs)
    state_dim = len(joint_names)
    first_action = first_data['actions']
    action_shape = first_action.cpu().numpy().shape if isinstance(first_action, torch.Tensor) else first_action.shape
    print("[Model] Policy dry-run finished.")

    with h5py.File(args.output_path, 'w') as h5_file:
        config_grp = h5_file.create_group("config")
        config_grp.attrs["dataset_freq_hz"] = dataset_freq
        config_grp.attrs["inference_freq_hz"] = args.inference_res_freq
        config_grp.attrs["resample_ratio"] = resample_ratio
        config_grp.attrs["chunk_length"] = action_shape[0]
        config_grp.attrs["use_arms"] = [int(use_arm) for use_arm in args.use_arms]
        config_grp.attrs["use_waist_angles"] = int(args.use_waist_angles)
        config_grp.attrs["use_tcp_pose"] = int(args.use_tcp_pose)
        episode_mse_stats = []
        for ep_idx in tqdm(all_episode_indices, desc="Episodes", dynamic_ncols=True):
            ep_iterator = EpisodeDataIterator(args.dataset_dir, ep_idx, args.batch_size, cache_videos=True)
            episode_len = len(ep_iterator)
            grp = h5_file.create_group(f"episode_{ep_idx:06d}")
            steps_ds = grp.create_dataset('step', shape=(episode_len,), dtype=np.int32)
            str_dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "action_joint_names",
                data=np.array(joint_names, dtype=object),
                dtype=str_dt,
            )
            states_ds = grp.create_dataset('state', shape=(episode_len, state_dim), dtype=np.float32)
            actions_ds = grp.create_dataset('action', shape=(episode_len,) + action_shape, dtype=np.float32)
            gt_actions_ds = grp.create_dataset('gt_action', shape=(episode_len, state_dim), dtype=np.float32)
            mse_ds = grp.create_dataset('action_mse', shape=(episode_len,), dtype=np.float32)
            first_data = next(iter(ep_iterator))
            if args.batch_size == 1:
                for step, data in enumerate(ep_iterator):
                    obs = unparse_observation(data, args.device, args.prompt)
                    with torch.inference_mode():
                        action_dict = policy.infer(obs)
                    real_actions = action_dict['actions']
                    state = np.array(data['observation.state'], dtype=np.float32)
                    action = real_actions.cpu().numpy() if isinstance(real_actions, torch.Tensor) else np.array(real_actions, dtype=np.float32)
                    steps_ds[step] = step
                    states_ds[step] = pack_action(state)
                    actions_ds[step] = action
                    gt_actions_ds[step] = pack_action(np.array(data["actions"], dtype=np.float32))
            else:
                assert isinstance(first_data, list)
                assert len(first_data) <= args.batch_size
                global_step = 0
                for batch_data in ep_iterator:
                    obs_batch = [unparse_observation(step_data, args.device, args.prompt) for step_data in batch_data]
                    with torch.inference_mode():
                        action_dict = policy.infer_batch(obs_batch)
                    actions = action_dict["actions"]
                    if isinstance(actions, torch.Tensor):
                        actions = actions.cpu().numpy()
                    for i, step_data in enumerate(batch_data):
                        states_ds[global_step] = pack_action(step_data["observation.state"])
                        actions_ds[global_step] = actions[i]
                        gt = pack_action(step_data["actions"])
                        gt_actions_ds[global_step] = gt
                        mse_ds[global_step] = np.mean((actions[i] - gt) ** 2)
                        global_step += 1
            all_pred_actions = actions_ds[:]
            all_gt_actions = np.expand_dims(gt_actions_ds[:], axis=1)
            mse_ds[:] = compute_stepwise_action_mse(all_pred_actions, all_gt_actions, action_shape[0], resample_ratio=resample_ratio)
            valid_mse = mse_ds[:][~np.isnan(mse_ds[:])]
            q25, q50, q75 = np.percentile(valid_mse, [25, 50, 75])
            ep_stats = {
                "episode_idx": ep_idx,
                "mean": np.mean(valid_mse),
                "std": np.std(valid_mse),
                "min": np.min(valid_mse),
                "q25": q25,
                "median": q50,
                "q75": q75,
                "max": np.max(valid_mse)
            }
            episode_mse_stats.append(ep_stats)
        summary_grp = h5_file.create_group("episode_mse_summary")
        for key in ["episode_idx", "mean", "std", "min", "q25", "median", "q75", "max"]:
            summary_grp.create_dataset(key, data=[s[key] for s in episode_mse_stats])
        sorted_eps = sorted(episode_mse_stats, key=lambda x: x["mean"], reverse=True)[:5]
        print("\n=== Top 5 episodes by mean MSE ===")
        for s in sorted_eps:
            print(f"Episode {s['episode_idx']:06d}: mean MSE={s['mean']:.6f}")


if __name__ == '__main__':
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    main()
