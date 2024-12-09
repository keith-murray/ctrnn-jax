"""Example training functions for JAX/Flax/Optax models."""

import jax
import jax.numpy as jnp
from flax import struct
from flax import serialization
from flax.training import train_state
import optax
from clu import metrics


@struct.dataclass
class Metrics(metrics.Collection):
    """Base `Metrics` class."""

    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")


# pylint: disable=abstract-method
class TrainState(train_state.TrainState):
    """Base `TrainState` class."""

    metrics: Metrics


def create_train_state(
    key,
    module,
    learning_rate,
    init_array,
):
    """Initializes a `TrainState` object."""
    params = module.init(key, init_array)["params"]
    tx = optax.adamw(
        learning_rate,
    )
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(
    key,
    state,
    batch,
):
    """Train for a single step."""

    def loss_fn(params):
        output, _ = state.apply_fn(
            {"params": params}, batch[0], rngs={"noise_stream": key}
        )
        loss = optax.squared_error(output, batch[1]).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_custom_accuracy(
    output,
    label,
):
    """Sample accuracy computation."""
    mse = jnp.mean((output - label) ** 2)
    variance = jnp.var(label)
    score = 1 - (mse / variance)
    return jnp.clip(score, 0, 1)


@jax.jit
def compute_metrics(
    key,
    state,
    batch,
):
    """Compute metrics after training step."""
    output, _ = state.apply_fn(
        {"params": state.params}, batch[0], rngs={"noise_stream": key}
    )
    loss = optax.squared_error(output, batch[1]).mean()
    accuracy = compute_custom_accuracy(output, batch[1])
    metric_updates = state.metrics.single_from_model_output(
        loss=loss, accuracy=accuracy
    )
    metrics_updated = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics_updated)
    return state


class ModelParameters:
    """
    Class to save and load model parameters.

    Note the class must be initialized with a train_state.
    """

    def __init__(self, state):
        self.params = {"params": state.params}

    def serialize(self, save_loc):
        """Serialize parameters to `.bin` file."""
        bytes_output = serialization.to_bytes(self.params)
        with open(save_loc, "wb") as f:
            f.write(bytes_output)

    def deserialize(self, save_loc):
        """Load parameters from `.bin` file."""
        with open(save_loc, "rb") as f:
            bytes_output = f.read()
        self.params = serialization.from_bytes(self.params, bytes_output)
