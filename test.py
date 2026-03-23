TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_waist",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_waist",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
            asset_id="pi0_industrial_sorting_joint_waist",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=1e-4,
        decay_steps=15_000,
        decay_lr=1e-5,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=40_000,
    log_interval=25,
    save_interval=500,
    keep_period=20_000,
    batch_size=128,
    num_workers=16,
    force_offline_dataset=True,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
            asset_id="pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=5e-5,
        decay_steps=15_000,
        decay_lr=5e-7,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=25_000,
    log_interval=25,
    save_interval=500,
    keep_period=20_000,
    batch_size=64,
    num_workers=16,
    force_offline_dataset=True,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_waist_manually_cleaned20251227",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
            asset_id="pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=1e-4,
        decay_steps=20_000,
        decay_lr=1e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=25_000,
    log_interval=25,
    save_interval=500,
    keep_period=20_000,
    batch_size=64,
    num_workers=16,
    force_offline_dataset=True,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_waist_manually_cleaned20251229",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
            asset_id="pi0_industrial_sorting_joint_waist_manually_cleaned20251224",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=3e-4,
        decay_steps=25_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=30_000,
    log_interval=25,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=16,
    force_offline_dataset=True,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_waist_manually_cleaned20251230",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_waist_manually_cleaned20251224_video_downsample",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214",
            asset_id="pi0_industrial_sorting_joint_waist_manually_cleaned20251224_video_downsample",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=3e-4,
        decay_steps=25_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=30_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.04,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_breaker_placement_joint_20260108",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_breaker_placement/zj-humanoid/breaker_placement_20260108",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_breaker_placement_joint_20260108",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_breaker_placement/zj-humanoid/breaker_placement_20260108",
            asset_id="pi0_breaker_placement_joint_20260108",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=False,
        action_use_waist_angles=False
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=3e-4,
        decay_steps=25_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=30_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.1,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_20260109",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260107",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_20260109",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260107",
            asset_id="pi0_industrial_sorting_joint_20260109",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=3e-4,
        decay_steps=25_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=30_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.04,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_20260112",
    project_name="industrial_sorting",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260112",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_20260112",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260112",
            asset_id="pi0_industrial_sorting_joint_20260112",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=5e-4,
        decay_steps=36_000,
        decay_lr=5e-5,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=40_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.0,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_20260130_last_frames_still",
    project_name="industrial_sorting",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260125",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_20260130_last_frames_still",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260125",
            asset_id="pi0_industrial_sorting_joint_20260130_last_frames_still",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=4e-4,
        decay_steps=22_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=25_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.0,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images",
    project_name="industrial_sorting",
    assets_base_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260125",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=False),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260125",
            asset_id="pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True,
        use_hand_align_state=True,
        hand_align_state_chest_image_mask_prob=0.2,
        hand_align_state_idx=64,
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=4e-4,
        decay_steps=22_000,
        decay_lr=2e-6,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=25_000,
    log_interval=50,
    save_interval=1000,
    keep_period=20_000,
    batch_size=64,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.0,
    # wandb_enabled=False,
),
TrainConfig(
    # Change the name to reflect your model and dataset.
    name="pi05_industrial_sorting_joint_20260112",
    project_name="industrial_sorting",
    assets_base_dir="/root/workspace/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260112",
    # Here you define the model config -- In this example we use pi0 as the model
    # architecture and perform *full* finetuning. in the examples below we show how to modify
    # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
    model=pi0_config.Pi0Config(pi05=True),
    # Here you define the dataset you are training on. In this example we use the Libero
    # dataset. For your own dataset, you can change the repo_id to point to your dataset.
    # Also modify the DataConfig to use the new config you made for your dataset above.
    data=LeRobotZJHumanoidDataConfig(
        repo_id="zj-humanoid/pi05_industrial_sorting_joint_20260112",
        assets=AssetsConfig(
            assets_dir="/root/workspace/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260112",
            asset_id="pi05_industrial_sorting_joint_20260112",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
            # a field called ``prompt`` in the input dict. The recommended setting is True.
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        tcp_pose_in_wrist=False,
        use_tcp_pose=False,
        use_arms=[False, True],
        use_wrist_cameras=[False, True],
        obs_use_waist_angles=True,
        action_use_waist_angles=True
    ),
    # Here you define which pre-trained checkpoint you want to load to initialize the model.
    # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1500,
        peak_lr=3e-4,
        decay_steps=45_000,
        decay_lr=5e-5,
    ),
    # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
    # Check the base TrainConfig class for a full list of available hyperparameters.
    num_train_steps=50_000,
    log_interval=100,
    save_interval=1000,
    keep_period=20_000,
    batch_size=32,
    num_workers=8,
    force_offline_dataset=True,
    val_fraction=0.0,
    # wandb_enabled=False,
),
