# LeRobot Dataset Tools

A collection of utilities for **validating, cleaning, and modifying LeRobot-style datasets**, with a focus on video-based robot manipulation datasets.

This repository provides:

* Dataset consistency checks (structure, metadata, indexing)
* Video / modality cleaning (e.g. removing depth streams)
* Dataset transformation tools (merge, prune, downsample, rebuild)

---

## 📂 Repository Structure

```
.
├── check_dataset/         # Dataset validation tools
│   ├── check_dataset.py
│   └── check_list/
│       ├── check_action_observation_consistency.py
│       ├── check_compute_episodes_stats.py
│       ├── check_dataset_info_consistency.py
│       ├── check_delete_depth_info.py
│       ├── check_delete_depth_jsondata.py
│       ├── check_episode_length.py
│       ├── check_lerobot_video_frames.py
│       ├── check_parquet_action_name_actions.py
│       ├── check_parquet_episode_index.py
│       ├── check_parquet_global_index_continuity.py
│       └── check_task_index_consistency.py
│
├── modify_dataset/        # Dataset modification tools
│   ├── dataset_video_deleter.py
│   ├── downsample_dataset.py
│   ├── expend_episodes_last_frames.py
│   ├── merge_datasets.py
│   ├── prune_episodes.py
│   ├── rebuild_episodes_jsonl.py
│   ├── reshape_video_images.py
│   └── update_task_prompt.py
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🚀 Installation

This project uses uv for dependency management.

### 1. Install uv (if not installed)

```
pip install uv
```

### 2. Install dependencies

```
uv sync
```

---

## 🧠 Dataset Format Assumption

The tools assume a dataset structure similar to:

```
dataset_root/
├── data/
│   └── chunk-xxx/
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
    └── chunk-xxx/
        └── observation.images.*
```

---

## 🔍 Dataset Validation

Run a full dataset check:

```
uv run check_dataset/check_dataset.py <dataset_root> --allow-depth
```

### Available Checks

| Check                          | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| action-observation consistency | Ensure alignment between actions and observations |
| episode length                 | Validate episode frame counts                     |
| dataset info consistency       | Validate `info.json` correctness                  |
| video frame check              | Ensure video frames match metadata                |
| parquet index checks           | Validate indexing consistency                     |
| task index consistency         | Ensure task mappings are valid                    |

---

## 🛠️ Dataset Modification Tools

### 1. Remove Specific Modalities (e.g. depth)

```
uv run modify_dataset/dataset_video_deleter.py <dataset_root> \
    --patterns depth --fix
```

Supports fuzzy matching:

* `depth` → remove all depth streams
* `chest_depth` → remove specific camera
* full name → exact match

Also:

* Removes corresponding JSON fields
* Deletes video directories
* Updates `info.json["total_videos"]`

---

### 2. Downsample Dataset

```
uv run modify_dataset/downsample_dataset.py --dataset-root <dataset_root> \
    --k <downsample_ratio> --output-root <output_root>
```

Used to reduce dataset frame rate.

---

### 3. Prune Episodes

```
uv run modify_dataset/prune_episodes.py --dataset-root <dataset_root> \
    --remove-indices 1,2,3
```

Remove invalid or unwanted episodes.

---

### 4. Merge Datasets

```
uv run modify_dataset/merge_datasets.py \
    --dataset-roots <dataset_root_1> <dataset_root_2> ... \
    --output-root <merged_dataset_root> \
    --chunk-size <merged_dataset_chunk_size>
```

Combine multiple datasets into one.

---

### 5. Rebuild Metadata

```
uv run modify_dataset/rebuild_episodes_jsonl.py \
    --dataset-root <dataset_root> --force
```

Reconstruct `episodes.jsonl` from existing data.

---

### 6. Update Task Prompts

```
uv run modify_dataset/update_task_prompt.py \
    --dataset-root <dataset_root> \
    --old-task "<old>" --new-task "<new>"
```

Modify task descriptions in `tasks.jsonl` and `episodes.jsonl`.

---

### 7. Reshape Videos

```
uv run modify_dataset/reshape_video_images.py \
    --folder <dataset_root> \
    --resolution 128 128 \
    [--force-resize] \
    [--zero-pad]
```

Resize all videos in the dataset to a target resolution. Supports multiple modes:

default (recommended): keep aspect ratio with minimal information loss (may exceed target size)
--zero-pad: keep aspect ratio and pad to exact resolution
--force-resize: directly resize to target resolution (may distort)

Also updates `meta/info.json` to reflect the new video dimensions.

---

### 8. Expend Episodes' Last Frames

```
uv run modify_dataset/expend_episodes_last_frames.py \
    --src-root <src_dataset_root> \
    --dst-root <dst_dataset_root> \
    --n <repeat_times> [--force-stats]
```

Pad each episode by duplicating its last frame N times, then recompute `episodes_stats.jsonl`.

---

## ⚠️ Important Notes

### 1. In-place Modification

Most tools use **in-place updates**:

* JSON files are overwritten
* Video directories are permanently deleted


👉 Always run without `--fix` first to preview changes if given `--fix` parameter.

