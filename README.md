# CT-RNN Implementation in Python's JAX ecosystem
This repository is an implementation of continuous-time recurrent neural networks (CT-RNNs) in the Python programming language and [JAX ecosystem](https://jax.readthedocs.io/en/latest/). Specifically, the architecture and training of CT-RNNs are implemented with the [Flax Linen](https://flax-linen.readthedocs.io/en/latest/) and [Optax](https://optax.readthedocs.io/en/latest/) APIs.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/ctrnn_strange_attractor.png" alt="logo" width="400"></img>
</div>

## What is a CT-RNN?
CT-RNNs are a class of recurrent neural networks described by the following equations:
```math
x_{t+1}=(1-\alpha)x_t+\alpha(W_{\text{rec}}f(x_t)+W_{\text{in}}u_t + b_{\text{rec}} + \eta_i(t))
```
```math
y_t=W_{\text{out}}f(x_t) + b_{\text{out}}
```
where $x_t\in\mathbb{R}^{n}$ is the voltage vector of recurrent neurons, $u_t\in\mathbb{R}^{m}$ is the input vector to the CT-RNN, $y_t\in\mathbb{R}^{p}$ is the firing rate vector of output neurons, $f$ is an activation function mapping voltage to firing rate, $W$ is a weight matrix, $b$ is a bias vector, and $\eta_i(t)\in\mathbb{R}^{n}$ is a vector of random values sampled from $\mathcal{N}(0,\sigma^2)$.

Computational neuroscience studies typically train CT-RNNs via the backpropigation-through-time (BPTT) learning algorithm to hypothesize low-dimensional dynamical systems underyling cognitive tasks. CT-RNNs are usually costly to train due to the `for` loop inherent in their architecture. Using JAX to implement and train CT-RNNs allows for large speedups over existing deep learning frameworks, like [PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org), due to the [`scan` primative](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html). Through using `scan`, the CT-RNN is unrolled in time, creating a large feedforward network with shared parameters broadcasted to each layer.

One potential issue with using JAX is that [pseudorandomly generated numbers need to be called via a key](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers). For the $\eta_i(t)$ vector, this means broadcasting unique keys to all time steps of the CT-RNN in the `scan` function. Fortunately, this is not too difficult.

## Installation
```
pip install git+https://github.com/keith-murray/ctrnn-jax.git
```

## Usage
```python
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from ctrnn_jax.model import CTRNNCell

ctrnn = nn.RNN(
    CTRNNCell(
        hidden_features=2,
        output_features=1,
        alpha=jnp.float32(0.1),
        noise_const=jnp.float32(0.1),
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

```
Take note that the [Flax's `nn.RNN` module](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.RNN) is a wrapper for Jax's `scan` primative that unrolls the `CTRNNCell`. Also take note of the `rngs` argument in the `ctrnn.apply` method. This is necessary to properly seed the $\eta_i(t)$ vector.

For an example of how to train the `CTRNNCell` via BPTT, refer to `scripts/train_CTRNN.ipynb` notebook.

## Examples
Refer to the `scripts/dynamics_CTRNN.ipynb` notebook for examples of the various dynamical systems phenomena CT-RNNs can implement.

Checkout my [attract-or-oscillate repository](https://github.com/keith-murray/attract-or-oscillate) where I was able to train [16,128 RNNs](https://openreview.net/forum?id=ql3u5ITQ5C) on the [MIT SuperCloud HPC](https://doi.org/10.1109/HPEC.2018.8547629) in about 60 hours using a JAX implementation of CT-RNNs.
