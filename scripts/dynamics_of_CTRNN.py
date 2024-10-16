# pylint: skip-file
"""Helper functions for `dynamics_of_CTRNN.ipynb."""

from functools import partial
from typing import Any

from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp

from flax.typing import Initializer as Initializer

KeyArray = Array
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


def modulated_ones(
    modulation: jnp.float32,
    key: KeyArray,
    shape: core.Shape,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Array:
    """Modulated `ones` initialization."""
    return modulation * jnp.ones(shape, dtypes.canonicalize_dtype(dtype))


def constant_init(
    modulation,
) -> Initializer:
    """Wrapper for flax's `Initializer` class."""
    return partial(modulated_ones, modulation)


def set_weights_and_baises(
    params,
    W,
    b,
):
    """Initailize preset wights (W) and biases (B)."""
    params["params"]["cell"]["recurrent_kernel"]["kernel"] = W
    params["params"]["cell"]["recurrent_kernel"]["bias"] = b
    return params


def hack_array(
    key: KeyArray, shape: core.Shape, dtype: DTypeLikeInexact = jnp.float_
) -> Array:
    """A hacky method to make custom initialization vectors."""
    return jnp.array([[1, 2.5, 1]])


def chaos_init() -> Initializer:
    """Wrapper for flax's `Initializer` class."""
    return hack_array
