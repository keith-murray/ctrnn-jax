"""Helper functions for `dynamics_of_CTRNN.ipynb."""

from functools import partial
from typing import Any

from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array
import jax.numpy as jnp

from flax.typing import Initializer

KeyArray = Array
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


def set_weights_and_baises(
    params,
    weights,
    bias,
):
    """Initailize preset wights (W) and biases (B)."""
    params["params"]["cell"]["recurrent_kernel"]["kernel"] = weights
    params["params"]["cell"]["recurrent_kernel"]["bias"] = bias
    return params


# pylint: disable=unused-argument
def index_ones(
    index: tuple,
    value: jnp.float32,
    key: KeyArray,
    shape: core.Shape,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Array:
    """An `ones` initialization modified at one index."""
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)).at[index].set(value)


def index_init(
    index: tuple,
    value: jnp.float32,
) -> Initializer:
    """Flax's `Initializer` wrapper for `index_ones` initialization."""
    return partial(
        index_ones,
        index,
        value,
    )
