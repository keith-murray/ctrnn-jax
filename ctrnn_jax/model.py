"""Continuous-time recurrent neural network (CTRNN) modules."""

from functools import partial
from typing import Any
from collections.abc import Callable

import jax.numpy as jnp
from jax import random

from flax.linen import initializers
from flax.linen.linear import Dense
from flax.linen.recurrent import RNNCellBase
from flax.linen.activation import tanh
from flax.linen.module import compact, nowrap
from flax.typing import PRNGKey, Dtype, Initializer


# pylint: disable=too-many-function-args
class CTRNNCell(RNNCellBase):
    """
    A continuous-time recurrent neural network (CTRNN) cell
    discritized and numerically integrated with Euler's method.

    Number of input neurons is infered via `.init()`.

    Attributes:
        hidden_features (int): number of hidden neurons.
        output_features (int): number of output neurons.
        alpha (jnp.float32): discretization factor for Euler integeration.
        noise_const (jnp.float32): scaling factor for injected noise.
    """

    hidden_features: int
    output_features: int
    alpha: jnp.float32
    noise_const: jnp.float32
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Initializer = initializers.glorot_normal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.ones_init()

    @compact
    def __call__(
        self,
        carry,
        task_input,
    ):
        """
        Compute hidden activations at current time step (t) given previous hidden
        activation (t-1) and current inputs (t). Return current outputs (t).

        Args:
            carry (jnp.ndarray): hidden activations at previous time step (t-1).
            task_input (jnp.ndarray): inputs at current time step (t).

        Returns:
            A tuple containing two elements:
                hidden_activations (jnp.ndarray): hidden activations at current time step (t).
                A tuple containing two elements:
                    z (jnp.ndarray): output activations at current time step (t).
                    rates_output (jnp.ndarray): hidden firing rates at current time step (t).
                        - Used for L2 regularization.
        """
        dense_h = partial(
            Dense,
            features=self.hidden_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_i = partial(
            Dense,
            features=self.hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_o = partial(
            Dense,
            features=self.output_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        noise_shape = carry.shape
        noise_key = self.make_rng("noise_stream")
        noise = self.noise_const * random.normal(noise_key, noise_shape)

        rates_input = self.activation_fn(carry)
        hidden_input = (
            dense_h(name="recurrent_kernel")(rates_input)
            + dense_i(name="input_kernel")(task_input)
            + noise
        )
        hidden_activations = (
            jnp.float32(1.0) - self.alpha
        ) * carry + self.alpha * hidden_input

        rates_output = self.activation_fn(hidden_activations)
        z = dense_o(name="output_kernel")(rates_output)

        return hidden_activations, (z, rates_output)

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """
        Initialize CTRNN cell hidden activations carry.

        Args:
            rng (PRNGKey): random number generator passed to the init_fn.
            input_shape (tuple[int]): a tuple providing the shape of the input to the cell.

        Returns:
            h (jnp.ndarray): A JAX array corresponding to the initialization of the hidden state.
        """
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(
            rng,
            mem_shape,
            self.param_dtype,
        )

    @property
    def num_feature_axes(
        self,
    ) -> int:
        return 1
