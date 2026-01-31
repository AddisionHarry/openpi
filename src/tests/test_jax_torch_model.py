#!/usr/bin/env python3
"""
Compare Torch vs JAX Diffusion Policy on the same frame with identical noise.
Generate PyTorch model use:
  uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128/39999 \
    --config_name pi05_industrial_sorting_joint_20260126 \
    --output_path /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128/39999_torch
  cp -rf /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128/39999/assets \
      /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128/39999_torch
"""
import argparse
from pathlib import Path
import numpy as np

import cv2
import json
import pyarrow.parquet as pq

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
    "waist":          (17, 19),
}

class EpisodeDataIterator:
    def __init__(self, dataset_root: str, episode_index: int, batch_size: int = 1, cache_videos: bool = False):
        self.dataset_root = Path(dataset_root)
        self.episode_index = episode_index
        self.cache_videos = cache_videos
        self.batch_size = batch_size
        assert batch_size == 1, "Visualization typically requires batch_size=1 to map attention correctly."

        self.episode_file = self._find_episode_file()
        self.episode_df = pq.ParquetFile(self.episode_file).read().to_pandas()
        self.episode_len = len(self.episode_df)

        self.video_paths = self._find_video_files()
        self.video_caps = {}
        self.video_frames = {}

        self.episode_prompts = self._load_episode_prompts()
        if self.episode_index not in self.episode_prompts:
            raise ValueError(f"Prompt for episode {self.episode_index} not found in meta/episodes.jsonl")
        self.prompt = self.episode_prompts[self.episode_index]

        if self.cache_videos:
            self._cache_all_video_frames()

        self.current_step = 0

    def _find_episode_file(self) -> Path:
        data_dir = self.dataset_root / "data"
        ep_file_pattern = f"episode_{self.episode_index:06d}.parquet"
        matched = list(data_dir.glob(f"chunk-*/*{ep_file_pattern}"))
        if not matched:
            raise FileNotFoundError(f"Episode file not found in {data_dir}")
        return matched[0]

    def _load_episode_prompts(self):
        meta_file = self.dataset_root / "meta" / "episodes.jsonl"
        prompts = {}
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                idx = item["episode_index"]
                task = item.get("tasks", "")
                prompts[idx] = task
        return prompts

    def _find_video_files(self) -> dict:
        video_root = self.dataset_root / "videos"
        video_paths = {}
        ep_video_pattern = f"episode_{self.episode_index:06d}.mp4"

        for chunk_dir in video_root.glob("chunk-*"):
            for cam_dir in chunk_dir.glob("observation.images.*"):
                cam_name = cam_dir.name.replace("observation.images.", "")
                video_file = cam_dir / ep_video_pattern
                if video_file.exists():
                    video_paths[cam_name] = str(video_file)
        return video_paths

    def _cache_all_video_frames(self):
        for cam_name, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            self.video_frames[cam_name] = frames

    def _get_video_frame(self, cam_name: str, step: int) -> np.ndarray:
        if self.cache_videos:
            return self.video_frames[cam_name][step]

        if cam_name not in self.video_caps:
            self.video_caps[cam_name] = cv2.VideoCapture(self.video_paths[cam_name])

        cap = self.video_caps[cam_name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ret, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_single_step(self, step: int) -> dict:
        if step < 0 or step >= self.episode_len:
            raise StopIteration(f"Step {step} out of range (max: {self.episode_len-1})")
        step_data = self.episode_df.iloc[step].to_dict()
        for cam_name in self.video_paths.keys():
            step_data[f"observation.images.{cam_name}"] = self._get_video_frame(cam_name, step)
        step_data["prompt"] = self.prompt
        return step_data

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self) -> dict:
        data = self._read_single_step(self.current_step)
        self.current_step += 1
        return data

    def __len__(self) -> int:
        return self.episode_len

def unparse_observation(obs: dict, prompt: str) -> dict:
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

def load_frame(dataset_path: str, episode_idx: int, frame_idx: int) -> dict:
    iterator = EpisodeDataIterator(dataset_path, episode_idx, batch_size=1, cache_videos=False)
    for i, step_data in enumerate(iterator):
        if i == frame_idx:
            return unparse_observation(step_data, step_data["prompt"])
    raise IndexError(f"Frame {frame_idx} not found in episode {episode_idx}")

def main():
    parser = argparse.ArgumentParser(description="Torch vs JAX Diffusion Policy Comparison")
    parser.add_argument("--torch-model-path", type=str, required=True, help="Path to Torch model checkpoint")
    parser.add_argument("--jax-model-path", type=str, required=True, help="Path to JAX model checkpoint")
    parser.add_argument("--config-name", type=str, default="pi05_zjhumanoid_grasp_can", help="Target configuration name")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to test")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index within episode to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise")
    parser.add_argument("--num-runs", type=int, default=10)
    args = parser.parse_args()

    cfg = config.get_config(args.config_name)
    noise_shape = (50, 32)
    print("Start loading model from checkpoint.")
    policies = [policy_config.create_trained_policy(cfg, model_path) for model_path in
                [args.torch_model_path, args.jax_model_path]]
    assert policies is not None
    print("Loaded model from checkpoint, start first inference...")
    first_obs = make_zj_humanoid_example(cfg.data.use_arms, cfg.data.use_tcp_pose,
                                         cfg.data.use_wrist_cameras, cfg.data.obs_use_waist_angles)
    policies[0].infer(first_obs), policies[1].infer(first_obs)
    print("Finished first inference.")

    # load data and infer
    obs = load_frame(args.dataset_path, args.episode_idx, args.frame_idx)
    noise = np.random.RandomState(args.seed).randn(*noise_shape).astype(np.float32)
    torch_actions = []
    jax_actions = []

    for _ in range(args.num_runs):
        torch_out, jax_out = policies[0].infer(obs, noise=noise), policies[1].infer(obs, noise=noise)
        torch_actions.append(torch_out["actions"])
        jax_actions.append(jax_out["actions"])
    torch_actions = np.stack(torch_actions, axis=0)
    jax_actions   = np.stack(jax_actions, axis=0)
    torch_mean = torch_actions.mean(axis=0)
    jax_mean   = jax_actions.mean(axis=0)
    torch_intra = np.mean(np.sum((torch_actions - torch_mean) ** 2, axis=(-1, -2)))
    jax_intra = np.mean(np.sum((jax_actions - jax_mean) ** 2, axis=(-1, -2)))
    inter = np.sum((torch_mean - jax_mean) ** 2)
    print("\n====== Diffusion Policy Consistency ======")
    print(f"Torch intra-class variance : {torch_intra:.6e}")
    print(f"JAX   intra-class variance : {jax_intra:.6e}")
    print(f"Inter-class variance      : {inter:.6e}")
    print(f"Inter / Intra ratio       : {inter / (torch_intra + jax_intra + 1e-12):.3e}")

    np.savez("comparison_output.npz", torch_actions=torch_actions, jax_actions=jax_actions,
             torch_intra_var=torch_intra, jax_intra_var=jax_intra, inter_var=inter)

if __name__ == "__main__":
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    main()
