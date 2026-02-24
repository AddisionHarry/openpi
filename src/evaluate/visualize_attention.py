#!/usr/bin/env python3
"""
Attention Visualization Tool for OpenPI Policy

Visualizes VLM and action attention for OpenPI policy with three camera images:
- chest_rgb
- left_wrist_rgb
- right_wrist_rgb

Saves:
- Per-layer VLM attention overlays
- Per-diffusion-step action attention overlays
- Average attention maps
- Token ratio statistics (H5)
"""

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm

from openpi.training import config as _config
from openpi.policies import policy_config
import openpi.policies.policy as _policy


ACTION_IDX: Dict[str, Tuple[int, int]] = {
    "left_arm_tcp": (38, 45),
    "right_arm_tcp": (45, 52),
    "left_arm_joint": (0, 7),
    "right_arm_joint": (7, 14),
    "left_hand": (52, 58),
    "right_hand": (58, 64),
    "waist": (17, 19),
}


class EpisodeDataIterator:
    """Iterator over a single episode returning per-step observation dictionary."""

    def __init__(self, dataset_root: str, episode_index: int, cache_videos: bool = True):
        self.dataset_root = Path(dataset_root)
        self.episode_index = episode_index
        self.cache_videos = cache_videos
        self.episode_file = self._find_episode_file()
        self.episode_df = pq.ParquetFile(self.episode_file).read().to_pandas()
        self.episode_len = len(self.episode_df)
        self.video_paths = self._find_video_files()
        self.video_frames: Dict[str, List[np.ndarray]] = {}
        if self.cache_videos:
            self._cache_all_video_frames()
        self.current_step = 0

    def _find_episode_file(self) -> Path:
        data_dir = self.dataset_root / "data"
        pattern = f"episode_{self.episode_index:06d}.parquet"
        matched = list(data_dir.glob(f"chunk-*/*{pattern}"))
        if not matched:
            raise FileNotFoundError(f"Episode file not found: {pattern}")
        return matched[0]

    def _find_video_files(self) -> Dict[str, str]:
        video_root = self.dataset_root / "videos"
        video_paths: Dict[str, str] = {}
        pattern = f"episode_{self.episode_index:06d}.mp4"
        for chunk_dir in video_root.glob("chunk-*"):
            for cam_dir in chunk_dir.glob("observation.images.*"):
                cam_name = cam_dir.name.replace("observation.images.", "")
                video_file = cam_dir / pattern
                if video_file.exists():
                    video_paths[cam_name] = str(video_file)
        return video_paths

    def _cache_all_video_frames(self):
        for cam_name, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            self.video_frames[cam_name] = frames

    def _get_video_frame(self, cam_name: str, step: int) -> np.ndarray:
        return self.video_frames[cam_name][step]

    def _read_single_step(self, step: int) -> Dict:
        if step >= self.episode_len:
            raise StopIteration
        step_data = self.episode_df.iloc[step].to_dict()
        for cam_name in self.video_paths.keys():
            step_data[f"observation.images.{cam_name}"] = self._get_video_frame(cam_name, step)
        return step_data

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        data = self._read_single_step(self.current_step)
        self.current_step += 1
        return data

    def __len__(self):
        return self.episode_len


def unparse_observation(obs: dict, prompt: str) -> dict:
    images = {k.split('.')[-2] + '_' + k.split('.')[-1]: np.array(v).astype(np.uint8)
              for k, v in obs.items() if 'observation.images' in k}

    def slice_state(name):
        return np.array(obs['observation.state'][slice(*ACTION_IDX[name])], dtype=np.float32)

    return {
        "observation/images/chest_rgb": images["images_chest_rgb"],
        "observation/images/left_wrist_rgb": images["images_left_wrist_rgb"],
        "observation/images/right_wrist_rgb": images["images_right_wrist_rgb"],
        "observation/end_effector/left_tcp": slice_state("left_arm_tcp"),
        "observation/end_effector/right_tcp": slice_state("right_arm_tcp"),
        "observation/left_arm_joint_position": slice_state("left_arm_joint"),
        "observation/right_arm_joint_position": slice_state("right_arm_joint"),
        "observation/left_hand_joint_position": slice_state("left_hand"),
        "observation/right_hand_joint_position": slice_state("right_hand"),
        "observation/waist_joint_position": slice_state("waist"),
        "prompt": prompt
    }


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _infer_hw(token_count: int) -> Tuple[int, int]:
    for h in range(1, token_count + 1):
        if token_count % h == 0:
            return h, token_count // h
    raise ValueError(f"Cannot factor token count {token_count}")


def build_token_slices(token_annotations: Dict) -> Dict[str, object]:
    """
    Build token slices for three images (256 tokens each) using annotations.
    """
    image_slices = [
        slice(0, 256),    # chest_rgb
        slice(256, 512),  # left_wrist_rgb
        slice(512, 768),  # right_wrist_rgb
    ]

    vlm_ann = token_annotations["VLM_Inference"]
    language_slice = slice(int(vlm_ann["language_tokens"]["start"]),
                           int(vlm_ann["language_tokens"]["end"]))

    action_ann = token_annotations["Action_Generate"]
    state_slice = slice(int(action_ann["state_tokens"]["start"]),
                        int(action_ann["state_tokens"]["end"]))
    action_slice = slice(int(action_ann["action_tokens"]["start"]),
                         int(action_ann["action_tokens"]["end"]))

    return {
        "image_slices": image_slices,
        "language_slice": language_slice,
        "state_slice": state_slice,
        "action_slice": action_slice
    }


def save_attention_overlay(image: np.ndarray, attn_grid: np.ndarray, save_path: str,
                           highlight_color: Tuple[int, int, int], target_size: int = 224) -> None:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    attn_resized = cv2.resize(attn_grid.astype(np.float32), (target_size, target_size))
    attn_color = np.zeros_like(gray)
    for c in range(3):
        attn_color[:, :, c] = (attn_resized * highlight_color[c]).astype(np.uint8)
    overlay = cv2.addWeighted(gray, 0.4, attn_color, 0.6, 0)
    cv2.imwrite(save_path, overlay[:, :, ::-1])


def visualize_all_attention(res_attn_maps: Dict, step_idx: int, output_dir: str,
                            images_dict: Dict[str, np.ndarray], target_size: int = 224):

    step_dir = os.path.join(output_dir, f"step_{step_idx:06d}")
    os.makedirs(step_dir, exist_ok=True)
    slices = build_token_slices(res_attn_maps["token_annotations"])

    images = list(images_dict.values())
    n_images = len(images)

    # --- VLM Attention ---
    vlm_attn_all = _to_numpy(res_attn_maps["vlm"])
    vlm_dir = os.path.join(step_dir, "VLM_Image_Attention")
    os.makedirs(vlm_dir, exist_ok=True)

    for layer_idx in range(vlm_attn_all.shape[0]):
        layer_dir = os.path.join(vlm_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        layer_attn = vlm_attn_all[layer_idx, 0, 0].mean(axis=0)

        for i, img_slice in enumerate(slices["image_slices"]):
            h, w = _infer_hw(img_slice.stop - img_slice.start)
            self_grid = layer_attn[img_slice, img_slice].sum(axis=-1)
            self_grid = (self_grid - self_grid.min()) / (self_grid.max() - self_grid.min() + 1e-8)
            save_attention_overlay(images[i], self_grid.reshape(h, w),
                                   os.path.join(layer_dir, f"image{i+1}_self.png"), (255, 0, 0), target_size)

            cross_grid = layer_attn[img_slice, slices["language_slice"]].sum(axis=-1)
            cross_grid = (cross_grid - cross_grid.min()) / (cross_grid.max() - cross_grid.min() + 1e-8)
            save_attention_overlay(images[i], cross_grid.reshape(h, w),
                                   os.path.join(layer_dir, f"image{i+1}_cross_language.png"), (255, 0, 0), target_size)

    # --- Action Attention ---
    action_dir = os.path.join(step_dir, "Action_Generation_Attention")
    os.makedirs(action_dir, exist_ok=True)
    action_attn_all = _to_numpy(res_attn_maps["action_expert"])
    ratio_storage: Dict[str, List[float]] = {f"image{i+1}": [] for i in range(n_images)}
    ratio_storage.update({"language": [], "action": [], "state": []})

    avg_grids = []
    img_hw = []
    for s in slices["image_slices"]:
        h, w = _infer_hw(s.stop - s.start)
        avg_grids.append(np.zeros((h, w), dtype=np.float32))
        img_hw.append((h, w))

    for step in range(action_attn_all.shape[0]):
        layer_attn = action_attn_all[step][-1, 0, 0].mean(axis=0)
        query = layer_attn[1:, :].mean(axis=0)
        total_sum = float(query.sum()) + 1e-8

        for i, img_slice in enumerate(slices["image_slices"]):
            img_sum = float(query[img_slice].sum())
            ratio_storage[f"image{i+1}"].append(img_sum / total_sum)

            vec = query[img_slice]
            vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)
            h, w = img_hw[i]
            vec_2d = vec.reshape(h, w)
            avg_grids[i] += vec_2d

            save_attention_overlay(images[i], vec_2d, os.path.join(action_dir, f"diffstep_{step:02d}_image{i+1}.png"),
                                (0,0,255), target_size)

        ratio_storage["language"].append(float(query[slices["language_slice"]].sum()) / total_sum)
        ratio_storage["action"].append(float(query[slices["action_slice"]].sum()) / total_sum)
        ratio_storage["state"].append(float(query[slices["state_slice"]].sum()) / total_sum)

    for i in range(n_images):
        avg_grid = avg_grids[i] / action_attn_all.shape[0]
        save_attention_overlay(images[i], avg_grid, os.path.join(action_dir, f"average_image{i+1}.png"),
                               (0,255,0), target_size)

    # --- Save H5 ---
    h5_path = os.path.join(action_dir, "attention_ratios.h5")
    with h5py.File(h5_path, "w") as f:
        for k, v in ratio_storage.items():
            arr = np.array(v, dtype=np.float32)
            f.create_dataset(f"{k}_per_step", data=arr)
            f.create_dataset(f"{k}_mean", data=float(arr.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--episode-all", action="store_true")
    parser.add_argument("--prompt", type=str, default="do something")
    parser.add_argument("--num-steps", type=int)
    args = parser.parse_args()

    if not args.episode_all and args.episode_index is None:
        raise ValueError("Provide --episode-index or --episode-all")

    base_cfg = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(dataclasses.replace(base_cfg, batch_size=1), args.model_path)

    data_dir = Path(args.dataset_dir) / "data"
    episode_files = sorted(data_dir.glob("chunk-*/*episode_*.parquet"))
    episode_indices = [args.episode_index] if not args.episode_all else \
        [int(p.stem.split("_")[-1]) for p in episode_files]

    for ep_idx in episode_indices:
        iterator = EpisodeDataIterator(args.dataset_dir, ep_idx, cache_videos=True)
        ep_out_dir = os.path.join(args.output_dir, f"episode_{ep_idx:06d}")
        os.makedirs(ep_out_dir, exist_ok=True)
        max_steps = len(iterator) if args.num_steps is None else args.num_steps
        for step_idx, raw_data in enumerate(tqdm(iterator, desc=f"Episode {ep_idx}")):
            if step_idx >= max_steps:
                break
            obs = unparse_observation(raw_data, args.prompt)
            policy._sample_kwargs.update({"get_vlm_attn_map": True, "get_action_attn_map": True})
            res = policy.infer(obs)
            images_dict = {k: v for k,v in obs.items() if k.startswith("observation/images")}
            visualize_all_attention(res["attn_maps"], step_idx, ep_out_dir, images_dict)


if __name__ == "__main__":
    if os.environ.get("DEBUG_MODE") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
    main()
