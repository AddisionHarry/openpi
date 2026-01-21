# Pytorch model only
import argparse
import dataclasses
import os
from pathlib import Path
from typing import Tuple, Dict, Union, List

import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm

from openpi.training import config as _config
from openpi.policies import policy_config
import openpi.policies.policy as _policy

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


def visualize_vlm_attention_per_token(att_maps, step_idx: int, output_dir: str, images_dict: dict, highlight_color=[255, 0, 0], target_size=224):
    step_dir = os.path.join(output_dir, f"step_{step_idx:06d}")
    vlm_dir = os.path.join(step_dir, "VLM_Image_Attention")
    os.makedirs(vlm_dir, exist_ok=True)

    # 获取 VLM attention
    vlm_attn = att_maps[0]['VLM_Inference'][0]  # [1, heads, tokens, tokens]
    vlm_attn = vlm_attn[0]  # 去 batch 维度 -> [heads, tokens, tokens]
    n_heads, n_tokens, _ = vlm_attn.shape

    # 获取 image token slice
    image_start, image_stop = att_maps[1]['VLM_Inference']['image_tokens'].start, att_maps[1]['VLM_Inference']['image_tokens'].stop
    single_image_token_num = int((image_stop - image_start) / 3)
    image_tokens_slices = [slice(start, start + single_image_token_num) for start in range(image_start, image_stop, single_image_token_num)]

    # 获取 language token slice
    language_tokens_slice = att_maps[1]['VLM_Inference']['language_tokens']

    for layer_idx in range(n_heads):
        layer_dir = os.path.join(vlm_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for img_idx, img_slice in enumerate(image_tokens_slices):
            grid_size = int((img_slice.stop - img_slice.start) ** 0.5)

            def prepare_attention(att):
                att_vec = att.sum(dim=-1).to(torch.float32).cpu().numpy()  # [tokens_current]
                p_low, p_high = np.percentile(att_vec, [5, 95])
                att_vec = np.clip(att_vec, p_low, p_high)
                att_vec = (att_vec - att_vec.min()) / (att_vec.max() - att_vec.min() + 1e-8)
                return np.reshape(att_vec, (grid_size, grid_size))

            self_attn_grid = prepare_attention(vlm_attn[layer_idx, img_slice, img_slice])

            other_img_idx = 1 - img_idx
            other_slice = image_tokens_slices[other_img_idx]
            cross_attn_grid = prepare_attention(vlm_attn[layer_idx, img_slice, other_slice])

            lang_attn_grid = prepare_attention(vlm_attn[layer_idx, img_slice, language_tokens_slice])

            img_names = list(images_dict.keys())
            img_orig = images_dict[img_names[img_idx]]
            h, w = img_orig.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            pad_y, pad_x = (target_size - new_h) // 2, (target_size - new_w) // 2
            img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
            img_gray = cv2.cvtColor(img_padded, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            def overlay_attention(base_img, attn_grid, alpha=0.6):
                attn_resized = cv2.resize(attn_grid, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                attn_color = np.zeros_like(base_img)
                for c in range(3):
                    attn_color[:, :, c] = (attn_resized * highlight_color[c]).astype(np.uint8)
                overlay = cv2.addWeighted(base_img, 1-alpha, attn_color, alpha, 0)
                return overlay

            cv2.imwrite(os.path.join(layer_dir, f"image{img_idx+1}_self_attention.png"),
                        overlay_attention(img_gray, self_attn_grid)[:, :, ::-1])
            cv2.imwrite(os.path.join(layer_dir, f"image{img_idx+1}_cross_attention_from_image{other_img_idx+1}.png"),
                        overlay_attention(img_gray, cross_attn_grid)[:, :, ::-1])
            cv2.imwrite(os.path.join(layer_dir, f"image{img_idx+1}_cross_attention_from_language.png"),
                        overlay_attention(img_gray, lang_attn_grid)[:, :, ::-1])

def save_attention_maps_with_token_labels_rgb(
    att_maps: Tuple[Dict[str, Union[Tuple[torch.Tensor], List[Tuple[torch.Tensor]]]], Dict[str, Dict]],
    step_idx: int, output_dir: str, highlight_color: list = [255, 0, 0],
    border_px: int = 20, title_font_scale: float = 0.8
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, f"step_{step_idx:06d}")
    os.makedirs(step_dir, exist_ok=True)

    mix_order = [
        ("vlm_tokens_first", "language_tokens"), ("vlm_tokens_first", "image_tokens"),
        ("action_tokens", "state_tokens"), ("action_tokens", "action_tokens")
    ]

    for mode, maps in att_maps[0].items():
        mode_dir = os.path.join(step_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        token_slices = att_maps[1].get(mode, {})
        vlm_slices = att_maps[1].get("VLM_Inference", {})

        if isinstance(maps, tuple):
            maps_iter = [(0, maps)]
        elif isinstance(maps, list):
            maps_iter = [(i, step_maps) for i, step_maps in enumerate(maps)]
        else:
            print(f"Unknown type for mode {mode}: {type(maps)}")
            continue

        for step_idx_local, step_maps in maps_iter:
            save_dir = mode_dir if mode != "Action_Generate" else os.path.join(mode_dir, f"time_step_{step_idx_local}")
            os.makedirs(save_dir, exist_ok=True)

            for layer_idx, layer_attn in enumerate(step_maps):
                layer_attn = layer_attn[0]
                att = layer_attn.sum(dim=0).to(torch.float32).cpu().numpy()
                att_min, att_max = att.min(), att.max()
                att_norm = (att - att_min) / (att_max - att_min + 1e-8)

                h, w = att_norm.shape

                img = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(3):
                    img[:, :, c] = (att_norm * highlight_color[c]).astype(np.uint8)

                boundary_color = (255, 255, 255)
                label_list = []

                if mode == "Action_Generate":
                    max_vlm_stop = 0
                    for k in ["language_tokens", "image_tokens"]:
                        s = vlm_slices.get(k)
                        if isinstance(s, slice) and s.stop is not None:
                            max_vlm_stop = max(max_vlm_stop, s.stop)

                    current_pos = 0
                    for source, name in mix_order:
                        source_dict = vlm_slices if source == "vlm_tokens_first" else token_slices
                        s = source_dict.get(name)
                        if not isinstance(s, slice) or s.start is None:
                            continue
                        token_len = s.stop - s.start
                        if token_len <= 0 or (name == "state_tokens" and token_len <= 1):
                            continue
                        if source == "action_tokens":
                            display_start = max_vlm_stop + s.start
                            display_stop = max_vlm_stop + s.stop
                        else:
                            display_start = s.start
                            display_stop = s.stop
                        cv2.line(img, (0, display_stop), (w, display_stop), boundary_color, 1)
                        cv2.line(img, (display_stop, 0), (display_stop, h), boundary_color, 1)
                        label_list.append(f"{name}: [{display_start}, {display_stop})")
                else:
                    for token_name, s in token_slices.items():
                        if isinstance(s, slice) and s.start is not None and s.stop is not None:
                            start, stop = s.start, s.stop
                            cv2.line(img, (0, stop), (w, stop), boundary_color, 1)
                            cv2.line(img, (stop, 0), (stop, h), boundary_color, 1)
                            label_list.append(f"{token_name}: [{start}, {stop})")

                label_line_height = 25
                label_height = label_line_height * (len(label_list) + 1)
                label_img = np.ones((label_height, w, 3), dtype=np.uint8) * 255

                for i, text in enumerate(label_list):
                    cv2.putText(label_img, text, (10, 20 + i * label_line_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                title_height = 50
                title_img = np.ones((title_height, w, 3), dtype=np.uint8) * 255
                title_text = f"{mode} Step{step_idx_local} Layer{layer_idx}"
                cv2.putText(title_img, title_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            title_font_scale, (0, 0, 0), 2, cv2.LINE_AA)

                combined = np.vstack([title_img, img, label_img])

                final_img = cv2.copyMakeBorder(combined, border_px, border_px, border_px, border_px,
                                               cv2.BORDER_CONSTANT, value=(255, 255, 255))

                out_path = os.path.join(save_dir, f"layer{layer_idx:02d}_heads_summed.png")
                cv2.imwrite(out_path, final_img[:, :, ::-1])


def extract_and_save_attn(policy: _policy.Policy, raw_obs_dict: Dict, step_idx: int, output_dir: str, prompt: str):
    obs = unparse_observation(raw_obs_dict, prompt)
    policy._sample_kwargs.update({"output_attentions": True})
    policy.infer(obs)
    assert hasattr(policy._model, "get_atten_maps"), "Must be a PyTorch Model."
    attn_maps = policy._model.get_atten_maps()
    # save_attention_maps_with_token_labels_rgb(attn_maps, step_idx, output_dir)
    visualize_vlm_attention_per_token(attn_maps, step_idx, output_dir,
                                      {
                                          "chest": obs["observation/images/chest_rgb"],
                                          "left_wrist": obs["observation/images/left_wrist_rgb"],
                                          "right_wrist": obs["observation/images/right_wrist_rgb"]
                                        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="do something")
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    base_cfg = _config.get_config(args.config_name)
    final_cfg = dataclasses.replace(base_cfg, batch_size=1)

    print(f"Loading policy with attention enabled...")
    policy = policy_config.create_trained_policy(final_cfg, args.model_path)

    print(f"Loading Episode {args.episode_index} from {args.dataset_dir}...")
    ep_iterator = EpisodeDataIterator(args.dataset_dir, args.episode_index, batch_size=1, cache_videos=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for i, raw_data in enumerate(tqdm(ep_iterator, desc="Visualizing")):
        if i >= args.num_steps: break
        extract_and_save_attn(policy, raw_data, i, args.output_dir, args.prompt)


if __name__ == "__main__":
    if os.environ.get("DEBUG_MODE") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
    main()
