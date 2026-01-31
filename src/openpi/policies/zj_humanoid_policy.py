
import dataclasses

import einops
import torch
import numpy as np

from typing import List

from openpi import transforms
from openpi.models import model as _model

def make_zj_humanoid_example(
    use_arms: List[bool] = [True, True],
    use_tcp_pose: bool = False,
    use_wrist_cameras: List[bool] = [True, True],
    use_waist_angles: bool = False
) -> dict:
    """Creates a random humanoid observation example for testing.

    Args:
        use_arms: [left, right] — whether to include each arm's joint and hand position.
        use_wrist_cameras: [left, right] — whether to include each wrist camera image.
    """
    obs = {
        "observation/images/chest_rgb": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

    # Wrist cameras
    if use_wrist_cameras[0]:
        obs["observation/images/left_wrist_rgb"] = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    if use_wrist_cameras[1]:
        obs["observation/images/right_wrist_rgb"] = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # Arm and hand joints
    if use_arms[0]:
        if use_tcp_pose:
            obs["observation/end_effector/left_tcp"] = np.random.rand(7)
        else:
            obs["observation/left_arm_joint_position"] = np.random.rand(7)
        obs["observation/left_hand_joint_position"] = np.random.rand(6)
    if use_arms[1]:
        if use_tcp_pose:
            obs["observation/end_effector/right_tcp"] = np.random.rand(7)
        else:
            obs["observation/right_arm_joint_position"] = np.random.rand(7)
        obs["observation/right_hand_joint_position"] = np.random.rand(6)

    if use_waist_angles:
        obs["observation/waist_joint_position"] = np.random.rand(2)

    return obs


def _parse_image(image, flip: bool = False) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.cpu()
        image = image.numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if flip:
        image = np.flip(image, axis=(0, 1))
    return image


@dataclasses.dataclass(frozen=True)
class ZJHumanoidInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    use_arms: List[bool] = dataclasses.field(default_factory=lambda: [False, True])
    use_waist_angles: bool = False
    use_wrist_cameras: List[bool] = dataclasses.field(default_factory=lambda: [False, True])
    use_tcp_pose: bool = False
    flip_wrist_images: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        chest_image = _parse_image(data["observation/images/chest_rgb"], False)
        left_wrist = _parse_image(data["observation/images/left_wrist_rgb"], self.flip_wrist_images) if self.use_wrist_cameras[0] else np.zeros_like(chest_image)
        right_wrist = _parse_image(data["observation/images/right_wrist_rgb"], self.flip_wrist_images) if self.use_wrist_cameras[1] else np.zeros_like(chest_image)

        if self.use_tcp_pose:
            left_arm = np.asarray(data["observation/end_effector/left_tcp"], dtype=np.float32) if self.use_arms[0] else np.zeros(7, dtype=np.float32)
            right_arm = np.asarray(data["observation/end_effector/right_tcp"], dtype=np.float32) if self.use_arms[1] else np.zeros(7, dtype=np.float32)
        else:
            left_arm = np.asarray(data["observation/left_arm_joint_position"], dtype=np.float32) if self.use_arms[0] else np.zeros(7, dtype=np.float32)
            right_arm = np.asarray(data["observation/right_arm_joint_position"], dtype=np.float32) if self.use_arms[1] else np.zeros(7, dtype=np.float32)
        left_hand = np.asarray(data["observation/left_hand_joint_position"], dtype=np.float32) if self.use_arms[0] else np.zeros(6, dtype=np.float32)
        right_hand = np.asarray(data["observation/right_hand_joint_position"], dtype=np.float32) if self.use_arms[1] else np.zeros(6, dtype=np.float32)
        if self.use_arms[0] and self.use_arms[1]:
            state = np.concatenate([left_arm, left_hand, right_arm, right_hand])
        elif self.use_arms[0] and not self.use_arms[1]:
            state = np.concatenate([left_arm, left_hand])
        elif not self.use_arms[0] and self.use_arms[1]:
            state = np.concatenate([right_arm, right_hand])
        else:
            raise ValueError("At least one arm must be used.")
        if self.use_waist_angles:
            waist_angles = np.asarray(data["observation/waist_joint_position"], dtype=np.float32)
            state = np.concatenate([state, waist_angles])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": chest_image,
                "left_wrist_0_rgb": left_wrist,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.bool_(self.use_wrist_cameras[0]),
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.bool_(self.use_wrist_cameras[1]),
            },
        }

        if "hand_align_state" in data:
            inputs["hand_align_state"] = 1
        if "hand_align_state_chest_image_mask_prob" in data:
            inputs["chest_image_mask_prob"] = data["hand_align_state_chest_image_mask_prob"]

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ZJHumnanoidOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    use_arms: List[bool] = dataclasses.field(default_factory=lambda: [False, True])
    use_waist_angles: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        waist_angles = 2 if self.use_waist_angles else 0
        if self.use_arms[0] and self.use_arms[1]:
            return {"actions": np.asarray(data["actions"][:, :(26 + waist_angles)])}
        elif self.use_arms[0] or self.use_arms[1]:
            return {"actions": np.asarray(data["actions"][:, :(13 + waist_angles)])}
        else:
            raise ValueError("At least one arm must be used.")
