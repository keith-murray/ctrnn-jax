# pylint: disable=duplicate-code
"""Example of how to use the `compute_pca` function."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

import matplotlib as mpl
import matplotlib.pyplot as plt

from ctrnn_jax.model import CTRNNCell
from ctrnn_jax.tasks import SineWaveGenerator
from ctrnn_jax.training import create_train_state, ModelParameters
from ctrnn_jax.pca import compute_pca, cumulative_variance


# Initialize a key
key = random.PRNGKey(0)

# Configure model parameters
HIDDEN_NEURONS = 100
OUTPUT_NEURONS = 1
ALPHA = jnp.float32(1.0)
NOISE_SCALAR = jnp.float32(0.00)

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

# Configure train_state
LEARNING_RATE = 10e-4
init_array = jnp.ones([1, 10, 1])

# Create train_state
key, state_key = random.split(key, num=2)
analysis_state = create_train_state(
    state_key,
    ctrnn,
    LEARNING_RATE,
    init_array,
)

# Load model parameters from `train_ctrnn.py`
params = ModelParameters(analysis_state)
params.deserialize("./data/params_example.bin")

# Configure task parameters
frequencies = jnp.arange(0.1, 0.6, 0.01)
TASK_TIME = 50
BATCH_SIZE = 10

# Initalize task
key, task_key = random.split(key, num=2)
dataset = SineWaveGenerator(task_key, frequencies, TASK_TIME)
tf_dataset_train = dataset.generate_tf_dataset(BATCH_SIZE, shuffle=False)

# Configure PCA parameters
N_COMPONENTS = 3

# Compute PCA
key, pca_key = random.split(key, num=2)
model_behavior, _ = compute_pca(
    pca_key,
    analysis_state,
    params.params,
    tf_dataset_train,
    N_COMPONENTS,
)

# Compute cumulative variance of first 3 principal components
key, var_key = random.split(key, num=2)
variance_array = cumulative_variance(
    var_key,
    analysis_state,
    params.params,
    tf_dataset_train,
)
pc3_variance = round(variance_array[2].item(), 2)

# Visualize PCA
# pylint: disable=invalid-sequence-index
cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=min(frequencies), vmax=max(frequencies))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i, freq in enumerate(frequencies):
    ax.plot(
        model_behavior["rates_pc"][i, :, 0],
        model_behavior["rates_pc"][i, :, 1],
        model_behavior["rates_pc"][i, :, 2],
        color=cmap(norm(freq)),
    )
ax.set_xlabel("PCA dimension 1")
ax.set_ylabel("PCA dimension 2")
ax.set_zlabel("PCA dimension 3")
ax.set_title(
    f"PCA of hidden neuron firing rates - Cumulative variance of {pc3_variance}"
)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.1)
cbar.set_label("Frequency")
plt.savefig("./results/pca_plot.png")
