# pylint: disable=duplicate-code
"""Example of how to use fixed-point analysis function."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.linen.activation import tanh
import optax

import matplotlib.pyplot as plt

from ctrnn_jax.model import CTRNNCell
from ctrnn_jax.tasks import SineWaveGenerator
from ctrnn_jax.training import create_train_state, ModelParameters
from ctrnn_jax.pca import compute_pca
from ctrnn_jax.fixed_points import (
    find_fixed_points,
    analyze_fixed_points,
    find_unique_fixed_points,
)


# Initialize a key
key = random.PRNGKey(42069)

# Configure model parameters
HIDDEN_NEURONS = 250
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
LEARNING_RATE = 0.00001
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
params_dict = params.params.copy()
params_dict["params"] = params_dict["params"]["cell"]

# Configure task parameters
TASK_TIME = 100
BATCH_SIZE = 100
FREQUENCY = 0.3
FREQUENCIES = FREQUENCY * jnp.ones(BATCH_SIZE)

# Initalize task and generate inputs
key, task_key = random.split(key, num=2)
dataset = SineWaveGenerator(task_key, FREQUENCIES, TASK_TIME)
tf_dataset_train = dataset.generate_tf_dataset(BATCH_SIZE)
task_arrays = list(tf_dataset_train.as_numpy_iterator())
inputs = task_arrays[0][0][:, 0:1, 0]
outputs = task_arrays[0][1][0, :, 0]

# Compute PCA of CT-RNN performing task
key, pca_key = random.split(key, num=2)
model_arrays, pca = compute_pca(
    pca_key,
    analysis_state,
    params.params,
    tf_dataset_train,
    2,
)

# Fixed point parameters
TOLERANCE = 1e-7
MAX_STEPS = 20000
LEARNING_RATE = 1e-2

# Initialize utils
key, initialize_key = random.split(key, num=2)
initial_states = 2.0 * random.normal(initialize_key, shape=(BATCH_SIZE, HIDDEN_NEURONS))
optimizer = optax.adam(learning_rate=LEARNING_RATE)

# Initialize model for fixed-points
ctrnn_cell = CTRNNCell(
    hidden_features=HIDDEN_NEURONS,
    output_features=OUTPUT_NEURONS,
    alpha=ALPHA,
    noise_const=NOISE_SCALAR,
)

# Find fixed points
fixed_points, loss_history, squared_speeds = find_fixed_points(
    ctrnn_cell,
    params_dict,
    initial_states,
    inputs,
    optimizer,
    TOLERANCE,
    MAX_STEPS,
)

# Select for unique fixed points
unique_fixed_points = find_unique_fixed_points(
    fixed_points,
)

# Ensure that exactly two fixed points were found
assert (
    unique_fixed_points.shape[0] == 2
), f"Expected exactly two fixed points but found {unique_fixed_points.shape[0]}."

# Project fixed-points onto principal components
fixed_points_pca = pca.transform(tanh(unique_fixed_points))

# Compute eigenspectrum
eigenspectrum_dict = analyze_fixed_points(
    ctrnn_cell, params_dict, unique_fixed_points, inputs[fixed_points_pca.shape[0], :]
)

# Extract model output and task output
time_axis = jnp.arange(TASK_TIME)
model_output = model_arrays["outputs"][0, :, 0]
task_output = outputs  # First sine wave output

# Extract PCA trajectory for the first batch sample
pca_trajectory = model_arrays["rates_pc"][0, :, :]

# Extract eigenvalues for both fixed points
eigval_1 = eigenspectrum_dict["eigenvalues"][0]
eigval_2 = eigenspectrum_dict["eigenvalues"][1]

# Generate points for unit circle
theta = jnp.linspace(0, 2 * jnp.pi, 300)
circle_x = jnp.cos(theta)
circle_y = jnp.sin(theta)

# Create figure with 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (0,0): Model output vs Task output
axes[0, 0].plot(time_axis, model_output, label="Model Output", color="tab:blue")
axes[0, 0].plot(
    time_axis,
    task_output,
    label="Target sine wave",
    linestyle="dashed",
    color="tab:orange",
)
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Output")
axes[0, 0].legend()
axes[0, 0].set_title("Model Output vs Task Output")

# Define colors for the fixed points
fixed_point_colors = ["tab:green", "tab:purple"]

# Panel (0,1): PCA Trajectory with Fixed Points
axes[0, 1].plot(
    pca_trajectory[:, 0], pca_trajectory[:, 1], label="PCA Trajectory", color="tab:blue"
)

for i, color in enumerate(fixed_point_colors):
    axes[0, 1].scatter(
        fixed_points_pca[i, 0],
        fixed_points_pca[i, 1],
        color=color,
        label=f"Fixed Point {i+1}",
        edgecolor="black",
        s=80,  # Slightly larger for visibility
    )

axes[0, 1].set_xlabel("PC1")
axes[0, 1].set_ylabel("PC2")
axes[0, 1].legend(loc="upper right")
axes[0, 1].set_title("PCA of Hidden States with Fixed Points")

# Panel (1,0): Eigenvalue Spectrum for Fixed Point 1
axes[1, 0].axhline(
    FREQUENCY, color="tab:red", linestyle="--", linewidth=0.75, label="Target Frequency"
)
axes[1, 0].axhline(0, color="tab:gray", linestyle="--", linewidth=0.5)
axes[1, 0].axvline(0, color="tab:gray", linestyle="--", linewidth=0.5)
axes[1, 0].plot(
    circle_x, circle_y, color="black", linestyle="dashed", label="Unit Circle"
)
axes[1, 0].scatter(
    jnp.real(eigval_1),
    jnp.imag(eigval_1),
    color=fixed_point_colors[0],
    label="Eigenvalues",
)
axes[1, 0].set_xlabel("Real component")
axes[1, 0].set_ylabel("Imaginary component")
axes[1, 0].legend()
axes[1, 0].set_title("Eigenvalue Spectrum for Fixed Point 1")

# Panel (1,1): Eigenvalue Spectrum for Fixed Point 2
axes[1, 1].axhline(0, color="tab:gray", linestyle="--", linewidth=0.5)
axes[1, 1].axvline(0, color="tab:gray", linestyle="--", linewidth=0.5)
axes[1, 1].plot(
    circle_x, circle_y, color="black", linestyle="dashed", label="Unit Circle"
)
axes[1, 1].scatter(
    jnp.real(eigval_2),
    jnp.imag(eigval_2),
    color=fixed_point_colors[1],
    label="Eigenvalues",
)
axes[1, 1].set_xlabel("Real component")
axes[1, 1].set_ylabel("Imaginary component")
axes[1, 1].legend()
axes[1, 1].set_title("Eigenvalue Spectrum for Fixed Point 2")

plt.tight_layout()
plt.savefig("./results/fixed_points.png")
