
## create env
```
conda create -n openpi_env python=3.11 -y
conda activate openpi_env
```
---
## download & config reposity
```
git clone https://ghfast.top/https://github.com/Physical-Intelligence/openpi.git
cd openpi

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install uv

export UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
git config --global url."https://bgithub.xyz/".insteadOf "https://github.com/"

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
---
## For first time running
**1. config huggingface default path** 

refer to https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables

```
ROOT=/mnt/pi0_datasets
mkdir -p $ROOT/.cache/huggingface/{datasets,hub} \
         $ROOT/.cache/{pip,uv} \
         $ROOT/.tmp \
         $ROOT/assets
```
```
export HF_HOME=$ROOT/.cache/huggingface
export HF_DATASETS_CACHE=$ROOT/.cache/huggingface/datasets
export HF_HUB_CACHE=$ROOT/.cache/huggingface/hub

export XDG_CACHE_HOME=$ROOT/.cache
export TMPDIR=$ROOT/.tmp
```

**2. config openpi cache**
```
export OPENPI_DATA_HOME=/mnt/pi0_datasets/openpi
```

**3.config openpi assets & checkpoints**
```
SRC=/root/workspace/openpi/checkpoints
DST=/mnt/pi0_datasets/openpi/checkpoints
mkdir -p "$DST"
ln -sfn "$DST" "$SRC"
```
Run comput norm
```
uv run scripts/compute_norm_stats.py --config-name name
```

Run the train process
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=my_experiment --overwrite
```

Download checkpoints
```
modelscope download --model masheng/pi0-fine-Tuned-Models pi0_fast_base.zip --local_dir /mnt/pi0_datasets/openpi
```

## When create a new termial
**config environment**
```
ROOT=/mnt/pi0_datasets
export HF_HOME=$ROOT/.cache/huggingface
export OPENPI_DATA_HOME=$ROOT/openpi

export XDG_CACHE_HOME=$ROOT/.cache
export TMPDIR=$ROOT/.tmp
```
```
export http_proxy=http://192.168.32.11:18000 && export https_proxy=http://192.168.32.11:18000
[option] export WANDB_MODE=offline

# to unset proxy
unset http_proxy
unset https_proxy
```


**Add new datasets**
```
ls -l
ln -sfn localdatasets_path hugging_face_path
```

## Modify the Datasets
0. check data by `lerobot_dataset_vertifier` or `check_data.py`

1. check task_index by `check_task_index.py`

- step 1: add this 'task index' below the info.json
    ```
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        }
    ```
- step 2: delet below in info.json
    ```
        "observation.images.chest_depth": {
            "dtype": "video",
            "shape": [
                720,
                1280,
                1
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "ffv1",
                "video.pix_fmt": "gray16le",
                "video.is_depth_map": true,
                "video.fps": 30,
                "video.channels": 1,
                "has_audio": false
            }
        },
    ```
- step 3: build and fix
    - `build_task_index.py` (you can use `check_task_index.py` firstly)
    - `build_stats.py`: Add `episoids_stats.jsonl`
    - `mask_action_observation.py`: Mask useless obs and actions
    - copy videos
    - [Option]`fix_ts_from_0.py`: Fix timestamps of pq start from 0
    - [Option]`fix_video_ts.py`: Fix timestamps of video start from 0
    - using `check_data.py` or `check.ipynb`

- step 4: downsample to 10HZ 
    - downsample the pq
    ```
    python /tools/downsample/downsample_pq.py
    ```
    - downsample the video
    ```
    python /tools/downsample/downsample_video.py --dataset-root path_to_data
    ```

- step5: update meta files (mainly for `info.json1` and `episodes.json`)
    - In `info.josn`
        - updata **fps** & **video.fps**
        - updata **total_frames** refer to the result gain from running `check.ipynb` 
        - [option: for delet some useless data] updata **total_episodes** & **total_videos**
    - In `episodes.json`
        - check by `check_real_length.py`
        - updata `length` by run `fix_length_in_episodes.py`

    - then build stats again, by `build_stats.py`


## Using hdf5 datasets

- Step 1: Using `tools/hdf5/check_h5.py` to make sure each group's name, for example: 
    ```
    $ python tools/hdf5/check_h5.py path_to_h5
    ```

    get ouput like
    ```
    [Group] / (root)
        [Dataset] /action
            shape=(300, 14)  dtype=float32  nbytes≈16800
            chunks=None  compression=None
        [Group] /observations
            ...
            ...
    ```

- Step 2: config following things in `tools/hdf5/convert_hdf5_data_to_lerobot.py`
    - REPO_NAME
    - RAW_DATASET_NAMES
    - desired datasets structure:
        ```
        dataset = LeRobotDataset.create(
            ...
            features={
                "image": {
                    "dtype": "image",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],},
            ...)
        ```
    - alternative name of group:
        ```
        dataset.add_frame(
            {
                "image": f['observations/images/cam_high'][step],
                ...
        ```
                           
- Step 3: Config huggingface home mentioned before and convert
    
    Use:

    ```
    ROOT=/mnt/pi0_datasets
    export HF_HOME=$ROOT/.cache/huggingface
    export OPENPI_DATA_HOME=$ROOT/openpi

    export XDG_CACHE_HOME=$ROOT/.cache
    export TMPDIR=$ROOT/.tmp
    ```

    Convert: 

    ```
    $ python tools/convert_hdf5_data_to_lerobot.py --data-dir /mnt/pi0_datasets/aloha/version_2/muti_obj_single_plate
    ```

- Step 4: Downsample pq, refer to step 4&5 in `Modify the Datasets`