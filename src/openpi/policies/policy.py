from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batch(self, obs_batch: list[dict]) -> dict:
        if self._is_pytorch_model:
            return self._infer_batch_pytorch(obs_batch)

        assert isinstance(obs_batch, (list, tuple))
        assert len(obs_batch) > 0
        for obs in obs_batch:
            assert isinstance(obs, dict), "Each observation must be a dict"

        # ------------------------------------------------------------
        # 1. per-sample input transform
        # ------------------------------------------------------------
        processed = []
        for obs in obs_batch:
            obs = jax.tree.map(lambda x: x, obs)
            obs = self._input_transform(obs)
            processed.append(obs)

        # ------------------------------------------------------------
        # 2. stack into batched PyTree
        #    list[dict] -> dict[*b, ...]
        # ------------------------------------------------------------
        batched_inputs = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0),
            *processed,
        )

        # ------------------------------------------------------------
        # 3. build Observation
        # ------------------------------------------------------------
        self._rng, rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(batched_inputs)

        # ------------------------------------------------------------
        # 4. inference
        # ------------------------------------------------------------
        start_time = time.monotonic()
        actions = self._sample_actions(rng, observation, **self._sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs_list = []
        B = len(obs_batch)
        for i in range(B):
            single_outputs = {
                "state": batched_inputs["state"][i],
                "actions": actions[i],
            }
            single_outputs = jax.tree.map(lambda x: np.asarray(x), single_outputs)
            single_outputs = self._output_transform(single_outputs)
            outputs_list.append(single_outputs)

        # ---------- stack ----------
        batched_outputs = {
            k: np.stack([o[k] for o in outputs_list], axis=0)
            for k in outputs_list[0]
            if k != "policy_timing"
        }

        batched_outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        return batched_outputs


    def _infer_batch_pytorch(self, obs_batch: list[dict]) -> dict:
        """
        PyTorch batch inference.
        obs_batch: list of observation dicts, len = B
        """
        assert self._is_pytorch_model, "_infer_batch_pytorch called on non-pytorch policy"
        assert len(obs_batch) > 0, "Empty obs_batch"

        # ----------------------------
        # Stack inputs into batch
        # ----------------------------
        # obs_batch: List[Dict[str, np.ndarray]]
        # -> Dict[str, torch.Tensor(B, ...)]
        inputs = {}
        keys = obs_batch[0].keys()

        for k in keys:
            vals = [obs[k] for obs in obs_batch]
            # numpy -> torch, stack on batch dim
            inputs[k] = torch.as_tensor(
                np.stack(vals, axis=0),
                device=self._pytorch_device,
            )

        # Apply input transforms (batch-safe)
        inputs = self._input_transform(inputs)

        # ----------------------------
        # Prepare kwargs
        # ----------------------------
        sample_kwargs = dict(self._sample_kwargs)

        if "noise" in sample_kwargs:
            noise = sample_kwargs["noise"]
            if isinstance(noise, np.ndarray):
                noise = torch.from_numpy(noise)

            if noise.ndim == 2:
                # (T, D) -> (1, T, D) -> broadcastable
                noise = noise.unsqueeze(0)

            # Ensure batch dimension
            if noise.shape[0] == 1 and len(obs_batch) > 1:
                noise = noise.expand(len(obs_batch), *noise.shape[1:])

            sample_kwargs["noise"] = noise.to(self._pytorch_device)

        # ----------------------------
        # Model inference
        # ----------------------------
        observation = _model.Observation.from_dict(inputs)

        torch.cuda.synchronize() if self._pytorch_device.startswith("cuda") else None
        start_time = time.monotonic()

        with torch.inference_mode():
            actions = self._sample_actions(
                self._pytorch_device,
                observation,
                **sample_kwargs,
            )

        torch.cuda.synchronize() if self._pytorch_device.startswith("cuda") else None
        model_time = time.monotonic() - start_time

        # ----------------------------
        # Post-process outputs
        # ----------------------------
        # actions: torch.Tensor(B, T, D)
        actions_np = actions.detach().cpu().numpy()

        outputs = {
            "actions": actions_np,
            "policy_timing": {
                "infer_ms": model_time * 1000,
                "batch_size": len(obs_batch),
            },
        }
        return outputs



    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
