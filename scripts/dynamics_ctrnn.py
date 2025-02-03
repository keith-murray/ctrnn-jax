"""Example of dynamical systems phenomena in CT-RNNs."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

import matplotlib.pyplot as plt

from ctrnn_jax.model import SimpleCTRNNCell
from ctrnn_jax.initializers import set_weights_and_baises, index_init


# General function to run a model
# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_model(hidden_size, init_vals, inputs, weights, bias, dt=0.1, tau_init=None):
    """Run a CTRNN model with given parameters."""
    if tau_init:
        ctrnn = nn.RNN(
            SimpleCTRNNCell(
                hidden_features=hidden_size,
                dt=dt,
                carry_init=nn.initializers.constant(init_vals),
                tau_init=tau_init,
            ),
        )

    else:
        ctrnn = nn.RNN(
            SimpleCTRNNCell(
                hidden_features=hidden_size,
                dt=dt,
                carry_init=nn.initializers.constant(init_vals),
            ),
        )

    params = ctrnn.init(random.PRNGKey(0), inputs)
    params = set_weights_and_baises(params, weights, bias)
    return jnp.squeeze(ctrnn.apply(params, inputs))


# Initialize figure with 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)


# Panel 1: Fixed-point attractors (Switch Model)
TIME = 500
input_array = jnp.zeros((1, TIME, 1))
W = jnp.float32(8) * jnp.ones((1, 1))
b = jnp.float32(-4) * jnp.ones(1)
init_cond = [0.00, 0.25, 0.50, 0.75, 1.00]

switch_states = jnp.zeros((5, TIME))
for idx, init_const in enumerate(init_cond):
    switch_states = switch_states.at[idx].set(
        run_model(1, init_const, input_array, W, b, dt=0.01)
    )

for i in range(5):
    axes[0, 0].plot(switch_states[i, :])

axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Neuron firing rate")
axes[0, 0].set_title("Switch Model Trajectories")


# Panel 2: Limit Cycle
TIME = 500
input_array = jnp.zeros((1, TIME, 2))
W = jnp.array([[4.5, -1], [1, 4.5]])
b = jnp.array([-2.75, -1.75])
init_cond = [0.0, 0.4, 0.5, 0.6, 1.0]

limit_cycle_state = jnp.zeros((5, TIME, 2))
for idx, init_const in enumerate(init_cond):
    limit_cycle_state = limit_cycle_state.at[idx].set(
        run_model(2, init_const, input_array, W, b)
    )

axes[0, 1].plot(limit_cycle_state[0, :, 0], limit_cycle_state[0, :, 1])
axes[0, 1].plot(limit_cycle_state[1, :, 0], limit_cycle_state[1, :, 1])
axes[0, 1].scatter(
    limit_cycle_state[2, :, 0], limit_cycle_state[2, :, 1], c="tab:purple"
)
axes[0, 1].plot(limit_cycle_state[3, :, 0], limit_cycle_state[3, :, 1])
axes[0, 1].plot(limit_cycle_state[4, :, 0], limit_cycle_state[4, :, 1])

axes[0, 1].set_xlabel("Neuron 1 firing rate")
axes[0, 1].set_ylabel("Neuron 2 firing rate")
axes[0, 1].set_title("Limit Cycle Trajectories")


# Panel 3: Strange Attractor Over Time
TIME = 5000
input_array = jnp.zeros((1, TIME, 3))
W = jnp.array([[5.422, -0.24, 0.535], [-0.018, 4.59, -2.25], [2.750, 1.210, 3.885]])
b = jnp.array([-4.108, -2.787, -1.114])
init_cond = [0.45, 0.50, 0.55]

strange_attractor_state = jnp.zeros((3, TIME, 3))
for idx, init_const in enumerate(init_cond):
    strange_attractor_state = strange_attractor_state.at[idx].set(
        run_model(
            3,
            init_const,
            input_array,
            W,
            b,
            tau_init=index_init((0, 1), jnp.float32(2.5)),
        )
    )

axes[1, 0].plot(
    0.1 * jnp.arange(TIME), strange_attractor_state[1, :, 0], label="Neuron 1"
)
axes[1, 0].plot(
    0.1 * jnp.arange(TIME), strange_attractor_state[1, :, 1], label="Neuron 2"
)
axes[1, 0].plot(
    0.1 * jnp.arange(TIME), strange_attractor_state[1, :, 2], label="Neuron 3"
)

axes[1, 0].set_title("Strange Attractor Over Time")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Firing Rate")
axes[1, 0].legend()


# Panel 4: 3D Strange Attractor
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].spines["top"].set_visible(False)
axes[1, 1].spines["right"].set_visible(False)
axes[1, 1].spines["left"].set_visible(False)
axes[1, 1].spines["bottom"].set_visible(False)

ax = fig.add_subplot(2, 2, 4, projection="3d")

for i, color in zip(range(3), ["tab:red", "tab:purple", "tab:green"]):
    ax.plot(
        strange_attractor_state[i, :, 0],
        strange_attractor_state[i, :, 1],
        strange_attractor_state[i, :, 2],
        c=color,
    )

ax.set_xlabel("Neuron 1")
ax.set_ylabel("Neuron 2")
ax.set_zlabel("Neuron 3")
ax.set_title("Strange Attractor in 3D")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect([1, 1, 1])

plt.savefig("./results/ctrnn_dynamics.png")
plt.show()
