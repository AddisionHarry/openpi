import time
import dataclasses
import functools
import logging
import platform
from typing import Any, Iterator, Callable, Tuple

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
from functools import partial

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        assert wandb.run is not None
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        assert wandb.run is not None
        wandb.run.log_code(str(epath.Path(__file__).parent.parent))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(jnp.array(chunked_loss))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def eval_step(state: training_utils.TrainState, rng: at.KeyArrayLike, batch: tuple[_model.Observation, _model.Actions]) -> Array:
    """
    Run validation on the entire validation dataset and return average metrics.

    Args:
        state: train state containing model params
        rng: random key
        val_data_iter: iterable yielding (observation, actions)

    Returns:
        {"val_loss": avg_loss}
    """
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions = batch
    observation = jax.tree.map(lambda x: jnp.array(x), observation)
    actions = jax.tree.map(lambda x: jnp.array(x), actions)

    pred = model.sample_actions(rng, observation=observation)
    return jnp.mean(jnp.square(pred - actions))


def evaluate_model(
    train_state: training_utils.TrainState,
    eval_rng: at.KeyArrayLike,
    val_data_iter: Iterator[tuple[_model.Observation, _model.Actions]],
    peval_step: Callable[[training_utils.TrainState, at.KeyArrayLike, tuple[_model.Observation, _model.Actions]], Array],
    num_eval_batches: int,
) -> Tuple[dict[str, float], at.KeyArrayLike]:
    """
    Run evaluation on a validation set.

    Args:
        train_state: current TrainState with model params
        eval_rng: JAX random key for evaluation
        val_data_iter: iterator over validation batches
        peval_step: jit-compiled eval_step function
        num_eval_batches: number of batches to evaluate
    Returns:
        dict: {"val_action_mse": avg_loss}
    """
    eval_losses = []

    for _ in tqdm.tqdm(range(num_eval_batches), desc="Evaluating", leave=False):
        eval_rng, step_rng = jax.random.split(eval_rng)
        eval_batch = next(val_data_iter)
        eval_losses.append(float(peval_step(train_state, step_rng, eval_batch)))

    avg_loss = float(np.mean(eval_losses))
    return {"val_action_mse": avg_loss}, eval_rng


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    lr_fn = config.lr_schedule.create()

    rng = jax.random.key(config.seed)
    train_rng, init_rng, eval_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_data_loader, val_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        val_fraction=0.04,
        shuffle=True,
    )
    train_data_iter, val_data_iter = iter(train_data_loader), iter(val_data_loader)
    train_batch = next(train_data_iter)
    num_eval_batches = 20
    try:
        val_dataset_len = len(val_data_loader._data_loader._data_loader.dataset)
        batch_size = val_data_loader._data_loader._data_loader.batch_size
        num_eval_batches = int(np.ceil(val_dataset_len // batch_size))
    except:
        pass
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(train_batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in train_batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(train_batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        eval_step,
        in_shardings=(train_state_sharding, replicated_sharding, data_sharding),
        out_shardings=None,
        donate_argnums=(),
    )

    start_step = int(train_state.step)
    last_log_step = start_step
    last_log_time = time.perf_counter()
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        desc="Training",
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, train_batch)
        infos.append(info)
        pbar.set_postfix(loss=f"{info['loss']:.4f}")
        if step % config.log_interval == 0:
            eval_info, eval_rng = evaluate_model(
                train_state, eval_rng, val_data_iter, peval_step, num_eval_batches
            )
            jax.block_until_ready(train_state)

            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            now = time.perf_counter()
            log_dict = dict(reduced_info)
            log_dict.update(eval_info)
            if step != start_step:
                steps_since_last = step - last_log_step
                time_per_step = (now - last_log_time) / max(1, steps_since_last)
                log_dict["time_per_step"] = time_per_step
            else:
                time_per_step = None
            last_log_time = now
            last_log_step = step

            lr = float(lr_fn(step))
            log_dict["learning_rate"] = lr
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            if time_per_step is not None:
                pbar.write(f"Step {step}: {info_str}, lr={lr:.3e}, time/step={time_per_step:.3f} s")
            else:
                pbar.write(f"Step {step}: {info_str}, lr={lr:.3e}")

            wandb.log(log_dict, step=step)
            infos = []
        train_batch = next(train_data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, train_data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    elif os.environ.get("DEBUG_MODE", "0") == "2":
        import debugpy
        debugpy.listen(("0.0.0.0", 5679))
        print("Waiting for VS Code debugger to attach on port 5679...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    main(_config.cli())
