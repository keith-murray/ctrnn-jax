"""Tests for CTRNNCell cell."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

from ctrnn_jax.model import CTRNNCell


def test_ctrnn_outputs_1():
    """
    Testing deterministic outputs of CTRNN cell match expectations.
    """
    ctrnn = nn.RNN(
        CTRNNCell(
            hidden_features=2,
            output_features=1,
            alpha=jnp.float32(0.1),
            noise_const=jnp.float32(0.0),
        ),
        split_rngs={"params": False, "noise_stream": True},
    )
    params = ctrnn.init(
        random.PRNGKey(0),
        jnp.ones([1, 5, 1]),  # (batch, time, input_features)
    )

    output, rates = ctrnn.apply(
        params,
        jnp.ones([1, 5, 1]),
        rngs={"noise_stream": random.PRNGKey(0)},
    )

    correct_output = jnp.array(
        [[[0.09670737], [0.08286397], [0.06787399], [0.05202901], [0.03562872]]],
        dtype=jnp.float32,
    )
    correct_rates = jnp.array(
        [
            [
                [0.7410722, 0.70247406],
                [0.7206758, 0.63767034],
                [0.7004661, 0.56828207],
                [0.68048954, 0.49551427],
                [0.6607816, 0.42060047],
            ]
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(correct_output, output, rtol=1e-08)
    assert jnp.allclose(correct_rates, rates, rtol=1e-08)


# Create 2 more output tests
# Create noise tests (test deterministic noise, nondeterministic noise, and relative noise)
