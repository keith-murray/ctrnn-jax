"""Tests for CTRNNCell cell."""

import jax.numpy as jnp
from jax import random
from flax import linen as nn

from ctrnn_jax.model import CTRNNCell


def test_ctrnn_outputs_1():
    """
    Testing deterministic outputs of CTRNN cell
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

    expected_output = jnp.array(
        [[[0.09670737], [0.08286397], [0.06787399], [0.05202901], [0.03562872]]],
        dtype=jnp.float32,
    )
    expected_rates = jnp.array(
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

    assert jnp.allclose(expected_output, output, rtol=1e-08)
    assert jnp.allclose(expected_rates, rates, rtol=1e-08)


def test_ctrnn_outputs_2():
    """
    Testing nondeterministic outputs of CTRNN cell
    """
    key = random.PRNGKey(0)

    ctrnn = nn.RNN(
        CTRNNCell(
            hidden_features=2,
            output_features=1,
            alpha=jnp.float32(0.1),
            noise_const=jnp.float32(1.0),
        ),
        split_rngs={"params": False, "noise_stream": True},
    )

    key, params_key = random.split(key, num=2)
    params = ctrnn.init(
        params_key,
        jnp.ones([1, 3, 2]),
    )

    (
        key,
        noise_stream_key_1,
    ) = random.split(key, num=2)
    output, rates = ctrnn.apply(
        params,
        jnp.ones([1, 5, 2]),
        rngs={"noise_stream": noise_stream_key_1},
    )

    expected_output = jnp.array(
        [[[-0.7016234], [-0.6496402], [-0.4007677], [-0.37901345], [-0.40708703]]],
        dtype=jnp.float32,
    )
    expected_rates = jnp.array(
        [
            [
                [0.77777886, 0.56863827],
                [0.76741546, 0.5057674],
                [0.6655982, 0.22767769],
                [0.5822597, 0.23603618],
                [0.65366364, 0.24111083],
            ]
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(expected_output, output, rtol=1e-08)
    assert jnp.allclose(expected_rates, rates, rtol=1e-08)


def test_ctrnn_noise_1():
    """
    Testing nondeterministic outputs of CTRNN cells with same PRNG keys match
    """
    key = random.PRNGKey(0)
    key, params_key = random.split(key, num=2)
    (
        key,
        noise_stream_key_1,
    ) = random.split(key, num=2)

    ctrnn = nn.RNN(
        CTRNNCell(
            hidden_features=100,
            output_features=1,
            alpha=jnp.float32(0.9),
            noise_const=jnp.float32(1.0),
        ),
        split_rngs={"params": False, "noise_stream": True},
    )

    params_1 = ctrnn.init(
        params_key,
        jnp.ones([1, 3, 2]),
    )

    output_1, rates_1 = ctrnn.apply(
        params_1,
        jnp.ones([1, 50, 2]),
        rngs={"noise_stream": noise_stream_key_1},
    )
    params_2 = ctrnn.init(
        params_key,
        jnp.ones([1, 3, 2]),
    )

    output_2, rates_2 = ctrnn.apply(
        params_2,
        jnp.ones([1, 50, 2]),
        rngs={"noise_stream": noise_stream_key_1},
    )

    assert jnp.allclose(output_1, output_2, rtol=1e-08)
    assert jnp.allclose(rates_1, rates_2, rtol=1e-08)


def test_ctrnn_noise_2():
    """
    Testing nondeterministic outputs of CTRNN cells with different PRNG keys do not match
    """
    ctrnn = nn.RNN(
        CTRNNCell(
            hidden_features=100,
            output_features=1,
            alpha=jnp.float32(0.9),
            noise_const=jnp.float32(1.0),
        ),
        split_rngs={"params": False, "noise_stream": True},
    )

    key = random.PRNGKey(0)
    key, params_key_1 = random.split(key, num=2)
    (
        key,
        noise_stream_key_1,
    ) = random.split(key, num=2)

    params_1 = ctrnn.init(
        params_key_1,
        jnp.ones([1, 3, 2]),
    )
    output_1, rates_1 = ctrnn.apply(
        params_1,
        jnp.ones([1, 50, 2]),
        rngs={"noise_stream": noise_stream_key_1},
    )

    key = random.PRNGKey(1)
    key, params_key_2 = random.split(key, num=2)
    (
        key,
        noise_stream_key_2,
    ) = random.split(key, num=2)

    params_2 = ctrnn.init(
        params_key_2,
        jnp.ones([1, 3, 2]),
    )

    output_2, rates_2 = ctrnn.apply(
        params_2,
        jnp.ones([1, 50, 2]),
        rngs={"noise_stream": noise_stream_key_2},
    )

    assert not jnp.allclose(output_1, output_2, rtol=1e-04)
    assert not jnp.allclose(rates_1, rates_2, rtol=1e-04)
