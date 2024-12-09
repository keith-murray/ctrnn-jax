"""Example of how to train `CTRNNCell`."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from ctrnn_jax.model import CTRNNCell
from ctrnn_jax.tasks import SineWaveGenerator
from ctrnn_jax.training import create_train_state, train_step, compute_metrics


# Initialize a key
key = random.PRNGKey(0)

# Configure model parameters
HIDDEN_NEURONS = 100
OUTPUT_NEURONS = 1
ALPHA = jnp.float32(0.1)
NOISE_SCALAR = jnp.float32(0.1)

# Initialize model
ctrnn = nn.RNN(
    CTRNNCell(
        hidden_features=HIDDEN_NEURONS,
        output_features=OUTPUT_NEURONS,
        alpha=ALPHA,
        noise_const=NOISE_SCALAR,
    ),
    split_rngs={"params": False, "noise_stream": True},
)

# Configure task parameters
frequencies = jnp.arange(0.1, 0.5, 0.001)
TASK_TIME = 50
BATCH_SIZE = 8

# Initalize task
key, task_key = random.split(key, num=2)
dataset = SineWaveGenerator(task_key, frequencies, TASK_TIME)
tf_dataset_train = dataset.generate_tf_dataset(BATCH_SIZE)


# Configure training parameters
EPOCHS = 1000
metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
}

LEARNING_RATE = 10e-4
init_array = jnp.ones([1, 10, 1])

# Create train_state
key, train_state_key = random.split(key, num=2)
train_state = create_train_state(
    train_state_key,
    ctrnn,
    LEARNING_RATE,
    init_array,
)

# Train the model
for epoch in tqdm(range(EPOCHS)):

    for _, batch in enumerate(tf_dataset_train.as_numpy_iterator()):
        key, train_key, metrics_key = random.split(key, num=3)
        train_state = train_step(
            train_key,
            train_state,
            batch,
        )
        train_state = compute_metrics(
            metrics_key,
            train_state,
            batch,
        )

    for metric, value in train_state.metrics.compute().items():
        metrics_history[f"train_{metric}"].append(value.item())

    train_state = train_state.replace(metrics=train_state.metrics.empty())

# Visualize the loss curve
plt.plot(jnp.arange(EPOCHS), metrics_history["train_loss"])
plt.title("Loss over training epochs")
plt.xlabel("Training epoch")
plt.ylabel("Mean squared error")
plt.savefig("./results/loss_curve_example.png")
