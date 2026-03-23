#!/usr/bin/env python3
"""
Convert TrainConfig(...) in a Python file to separate YAML files.
- Short lists (length <=3) stay on single line.
- weight_loader and lr_schedule converted to YAML-safe dicts.
- model.pi05 updated properly.
"""

import ast
import yaml
import argparse
from pathlib import Path
import copy

# Default YAML template
DEFAULT_YAML = {
    "name": "test",
    "project_name": "openpi",
    "assets_base_dir": "/root/workspace/openpi/assets/default",
    "model": {
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": None,
        "pi05": True,
        "discrete_state_input": None,
    },
    "data": {
        "repo_id": "",
        "assets": {"assets_dir": "", "asset_id": ""},
        "base_config": {"prompt_from_task": True},
        "extra_delta_transform": False,
        "force_offline_dataset": True,
        "use_arms": [False, True],
        "obs_use_waist_angles": True,
        "action_use_waist_angles": False,
        "use_wrist_cameras": [False, True],
        "use_tcp_pose": True,
        "tcp_pose_in_wrist": False,
        "flip_wrist_images": False,
        "use_hand_align_state": False,
        "hand_align_state_chest_image_mask_prob": 0.0,
        "hand_align_state_idx": 0,
    },
    "weight_loader": {"params_path": ""},
    "lr_schedule": {"warmup_steps": 1500, "peak_lr": 1e-5, "decay_steps": 25000, "decay_lr": 1e-6},
    "pytorch_weight_path": None,
    "pytorch_training_precision": "bfloat16",
    "num_train_steps": 30000,
    "log_interval": 25,
    "save_interval": 500,
    "keep_period": 20000,
    "batch_size": 16,
    "num_workers": 15,
    "ema_decay": 0.99,
    "seed": 42,
    "wandb_enabled": True,
    "force_offline_dataset": True,
    "val_fraction": 0.0,
}


def parse_value(node):
    """Convert AST node to Python value, handling specific classes."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [parse_value(e) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(parse_value(e) for e in node.elts)
    elif isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
        # DataConfig / AssetsConfig / LeRobotZJHumanoidDataConfig
        if func_name == "LeRobotZJHumanoidDataConfig":
            cfg = {}
            for kw in node.keywords:
                if kw.arg == "repo_id":
                    cfg["repo_id"] = parse_value(kw.value)
                elif kw.arg == "assets":
                    cfg["assets"] = parse_value(kw.value)
                elif kw.arg == "base_config":
                    cfg["base_config"] = parse_value(kw.value)
                else:
                    cfg[kw.arg] = parse_value(kw.value)
            return cfg
        elif func_name == "AssetsConfig":
            cfg = {}
            for kw in node.keywords:
                cfg[kw.arg] = parse_value(kw.value)
            return cfg
        elif func_name == "DataConfig":
            cfg = {}
            for kw in node.keywords:
                cfg[kw.arg] = parse_value(kw.value)
            return cfg
        elif func_name == "CheckpointWeightLoader":
            if node.args:
                return parse_value(node.args[0])
            return ""
        elif func_name == "Pi0Config" or func_name == "pi0_config.Pi0Config":
            cfg = {}
            for kw in node.keywords:
                cfg[kw.arg] = parse_value(kw.value)
            return cfg
        elif func_name == "_optimizer.CosineDecaySchedule":
            # Convert to dict
            cfg = {}
            for kw in node.keywords:
                cfg[kw.arg] = parse_value(kw.value)
            return cfg
        else:
            return None
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Dict):
        return {parse_value(k): parse_value(v) for k, v in zip(node.keys, node.values)}
    return None


def parse_trainconfigs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    configs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "TrainConfig":
            cfg = {}
            for kw in node.keywords:
                cfg[kw.arg] = parse_value(kw.value)
            configs.append(cfg)
    return configs


def merge_with_defaults(cfg):
    yaml_cfg = copy.deepcopy(DEFAULT_YAML)

    yaml_cfg["name"] = cfg.get("name")
    yaml_cfg["project_name"] = cfg.get("project_name", yaml_cfg["project_name"])
    yaml_cfg["assets_base_dir"] = cfg.get("assets_base_dir", yaml_cfg["assets_base_dir"])
    yaml_cfg["num_train_steps"] = cfg.get("num_train_steps", yaml_cfg["num_train_steps"])
    yaml_cfg["log_interval"] = cfg.get("log_interval", yaml_cfg["log_interval"])
    yaml_cfg["save_interval"] = cfg.get("save_interval", yaml_cfg["save_interval"])
    yaml_cfg["keep_period"] = cfg.get("keep_period", yaml_cfg["keep_period"])
    yaml_cfg["batch_size"] = cfg.get("batch_size", yaml_cfg["batch_size"])
    yaml_cfg["num_workers"] = cfg.get("num_workers", yaml_cfg["num_workers"])
    yaml_cfg["force_offline_dataset"] = cfg.get("force_offline_dataset", yaml_cfg["force_offline_dataset"])
    yaml_cfg["val_fraction"] = cfg.get("val_fraction", yaml_cfg["val_fraction"])
    yaml_cfg["wandb_enabled"] = cfg.get("wandb_enabled", yaml_cfg["wandb_enabled"])

    # model
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        yaml_cfg["model"].update(model_cfg)

    # data
    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        yaml_cfg["data"]["repo_id"] = data_cfg.get("repo_id", yaml_cfg["data"]["repo_id"])
        assets = data_cfg.get("assets", {})
        if isinstance(assets, dict):
            yaml_cfg["data"]["assets"]["assets_dir"] = assets.get("assets_dir", yaml_cfg["data"]["assets"]["assets_dir"])
            yaml_cfg["data"]["assets"]["asset_id"] = assets.get("asset_id", yaml_cfg["data"]["assets"]["asset_id"])
        base_config = data_cfg.get("base_config", {})
        if isinstance(base_config, dict):
            yaml_cfg["data"]["base_config"].update(base_config)
        for key in ["extra_delta_transform", "force_offline_dataset", "use_arms", "obs_use_waist_angles",
                    "action_use_waist_angles", "use_wrist_cameras", "use_tcp_pose", "tcp_pose_in_wrist"]:
            if key in data_cfg:
                yaml_cfg["data"][key] = data_cfg[key]

    # weight_loader
    wl = cfg.get("weight_loader")
    if isinstance(wl, str):
        yaml_cfg["weight_loader"]["params_path"] = wl

    # lr_schedule
    lr = cfg.get("lr_schedule")
    if isinstance(lr, dict):
        yaml_cfg["lr_schedule"].update(lr)

    return yaml_cfg


class ShortListDumper(yaml.SafeDumper):
    """Custom dumper to keep short lists (≤3) in single line."""
    pass


def short_list_representer(dumper, data):
    if len(data) <= 3:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)


yaml.add_representer(list, short_list_representer, Dumper=ShortListDumper)


def write_yaml_files(input_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainconfigs = parse_trainconfigs(input_file)
    print(f"Found {len(trainconfigs)} TrainConfig entries.")

    for cfg in trainconfigs:
        yaml_cfg = merge_with_defaults(cfg)
        file_name = f"{yaml_cfg['name']}.yaml"
        with open(output_dir / file_name, "w") as f:
            yaml.dump(yaml_cfg, f, sort_keys=False, Dumper=ShortListDumper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Python file containing TrainConfig(...) definitions")
    parser.add_argument("--output_dir", required=True, help="Directory to write YAML files")
    args = parser.parse_args()
    write_yaml_files(args.input, args.output_dir)
