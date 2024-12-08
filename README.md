# CT-RNN Implementation in Python's JAX ecosystem
This repository is an implementation of continuous-time recurrent neural networks (CT-RNNs) in the Python programming language and [JAX ecosystem](https://jax.readthedocs.io/en/latest/). Specifically, the architecture and training of CT-RNNs are implemented with the [Flax Linen](https://flax-linen.readthedocs.io/en/latest/) and [Optax](https://optax.readthedocs.io/en/latest/) APIs.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/ctrnn_strange_attractor.png" alt="logo" width="400"></img>
</div>

## What is a CT-RNN?
CT-RNNs are a class of recurrent neural networks described by the following equations:
```math
x_{t+1}=(1-\alpha)x_t+\alpha(W_{\text{rec}}f(x_t)+W_{\text{in}}u_t + b_{\text{rec}} + \eta_t)
```
```math
y_t=W_{\text{out}}f(x_t) + b_{\text{out}}
```
where $x_t\in\mathbb{R}^{n}$ is the voltage vector of recurrent neurons, $u_t\in\mathbb{R}^{m}$ is the input vector to the CT-RNN, $y_t\in\mathbb{R}^{p}$ is the firing rate vector of output neurons, $f$ is an activation function mapping voltage to firing rate, $W$ is a weight matrix, $b$ is a bias vector, and $\eta_t\in\mathbb{R}^{n}$ is a vector of randomly sampled values from $\mathcal{N}(0,\sigma^2)$ at each time step, $t$.

Computational neuroscience studies typically train CT-RNNs via the backpropigation-through-time (BPTT) learning algorithm to [hypothesize low-dimensional dynamical systems](https://doi.org/10.1162/NECO_a_00409) underyling cognitive tasks. CT-RNNs, though, can be time-consuming to train due to the `for` loop inherent in their architecture. JAX eleminates the `for` loop by unrolling CT-RNNs in time via the [`scan` primative](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html), creating a large feedforward network with shared parameters broadcasted to each layer. Using JAX's `scan` primative to implement and train CT-RNNs results in large speedups over existing deep learning frameworks, like [PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org).

One challenge in using JAX is that [pseudorandomly generated numbers need to be generated with a key](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers). For the $\eta_t$ vector, this requires broadcasting unique keys to all time steps of the CT-RNN through `scan`. Fortunately, Flax can gracefully manage [layer-dependent keys](https://flax-linen.readthedocs.io/en/latest/guides/flax_sharp_bits.html#flax-linen-dropout-layer-and-randomness).

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
Take note that [Flax's `nn.RNN` module](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.RNN) is a wrapper for Jax's `scan` primative. Also take note of the `rngs` argument in the `ctrnn.apply` method that is necessary to seed the $\eta_t$ vector.

For an example of how to train the `CTRNNCell` via BPTT, refer to `scripts/train_CTRNN.ipynb` notebook where a 100-neuron CT-RNN was trained for 1,000 epochs in less than 150 seconds.

## Examples
Refer to the `scripts/dynamics_CTRNN.ipynb` notebook for examples of the various dynamical systems phenomena CT-RNNs can implement.

Checkout [`keith-murray/grokking-recurrent-networks`](https://github.com/keith-murray/grokking-recurrent-networks) where I was able to train [16,128 CT-RNNs](https://openreview.net/forum?id=ql3u5ITQ5C) on the [MIT SuperCloud HPC](https://doi.org/10.1109/HPEC.2018.8547629) in about 60 hours via JAX.
