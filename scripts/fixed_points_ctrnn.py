# pylint: disable=duplicate-code
"""Example of how to use fixed-point analysis function."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax

import matplotlib.pyplot as plt

from ctrnn_jax.model import CTRNNCell
from ctrnn_jax.tasks import SineWaveGenerator
from ctrnn_jax.training import create_train_state, ModelParameters
from ctrnn_jax.fixed_points import (
    find_fixed_points,
    analyze_fixed_points,
    find_unique_fixed_points,
)


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
params_dict = params.params.copy()
params_dict["params"] = params_dict["params"]["cell"]

# Configure task parameters
TASK_TIME = 100
BATCH_SIZE = 50
FREQUENCY = 0.3
FREQUENCIES = FREQUENCY * jnp.ones(BATCH_SIZE)

# Initalize task and generate inputs
key, task_key = random.split(key, num=2)
dataset = SineWaveGenerator(task_key, FREQUENCIES, TASK_TIME)
tf_dataset_train = dataset.generate_tf_dataset(BATCH_SIZE)
inputs = list(tf_dataset_train.as_numpy_iterator())[0][0][:, 0:1, 0]

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
unique_fixed_points = find_unique_fixed_points(fixed_points)
print(unique_fixed_points.shape)

# Compute eigenspectrum
eigenspectrum_dict = analyze_fixed_points(
    ctrnn_cell, params_dict, unique_fixed_points, inputs[2, :]
)
eigval = eigenspectrum_dict["eigenvalues"][1]

# Find dominant eigenvalue
max_eigval_ind = jnp.argmax(jnp.abs(eigval))
max_eigenvalue = eigval[max_eigval_ind]

# Compute frequency from eigenvalue
eigenvalue_freq = abs(jnp.imag(max_eigenvalue).item())
eigenvalue_freq = round(eigenvalue_freq, 2)

# Generate points along the unit circle
theta = jnp.linspace(0, 2 * jnp.pi, 300)  # 300 points for smoothness
circle_x = jnp.cos(theta)
circle_y = jnp.sin(theta)

# Plot eigenvalues
plt.scatter(jnp.real(eigval), jnp.imag(eigval), color="tab:blue", label="Eigenvalues")
plt.scatter(
    jnp.real(eigval[max_eigval_ind]),
    jnp.imag(eigval[max_eigval_ind]),
    color="tab:orange",
)

# Plot unit circle
plt.plot(
    circle_x,
    circle_y,
    color="black",
    linestyle="dashed",
    linewidth=1,
    label="Unit circle",
)

# Set axis properties
plt.xticks([-1, 0, 1])
plt.xlabel("Real component")
plt.ylabel("Imaginary component")
plt.yticks([-1, 0, 1])
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.legend()
plt.axis("equal")
plt.title(
    f"Eigenvalue spectrum for frequency {FREQUENCY} - Dominant eigenvalue of {eigenvalue_freq}"
)

plt.show()
