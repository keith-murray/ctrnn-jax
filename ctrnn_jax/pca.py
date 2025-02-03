"""PCA functions to analyze trained CT-RNNs."""

from jax import random
import jax.numpy as jnp
from sklearn.decomposition import PCA


def extract_task_data(
    key,
    state,
    params,
    task_tf,
):
    """
    Extracts inputs, rates, and outputs from the task dataset.

    Args:
        key (PRNGKey): Random number generator for seeding "noise_stream".
        state (flax.training.train_state.TrainState): Model state object.
        params (dict): Model parameters dictionary.
        task_tf (tf.data.Dataset): Training task dataset.

    Returns:
        tuple: (inputs, rates, outputs), each as a concatenated JAX array.
    """
    inputs, rates, outputs = [], [], []

    for _inputs, _ in task_tf.as_numpy_iterator():
        key, test_key = random.split(key, num=2)
        _outputs, _rates = state.apply_fn(
            params, _inputs, rngs={"noise_stream": test_key}
        )

        inputs.append(_inputs)
        rates.append(_rates)
        outputs.append(_outputs)

    return (
        jnp.concatenate(inputs, axis=0),
        jnp.concatenate(rates, axis=0),
        jnp.concatenate(outputs, axis=0),
    )


def compute_pca(key, state, params, task_tf, n_components=None):
    """
    Computes PCA on model rates.

    Args:
        key (PRNGKey): Random number generator for seeding "noise_stream".
        state (flax.training.train_state.TrainState): Model state object.
        params (dict): Model parameters dictionary.
        task_tf (tf.data.Dataset): Training task in TensorFlow Dataset format.
        n_components (int, optional): Number of principal components.

    Returns:
        tuple:
            model_arrays (dict): Dictionary containing:
                - "inputs" (jnp.ndarray): Task inputs.
                - "rates" (jnp.ndarray): CT-RNN firing rates.
                - "rates_pc" (jnp.ndarray): Principal components of `rates`.
                - "outputs" (jnp.ndarray): Model outputs.
            pca (sklearn.decomposition.PCA): Fitted PCA object.
    """
    inputs, rates, outputs = extract_task_data(key, state, params, task_tf)

    model_arrays = {
        "inputs": inputs,
        "rates": rates,
        "outputs": outputs,
    }

    rates_reshaped = rates.reshape(-1, rates.shape[-1])

    pca = PCA(n_components=n_components)
    rates_pc_reshaped = pca.fit_transform(rates_reshaped)

    rates_pc_shape = rates.shape
    if n_components is not None:
        rates_pc_shape = (rates_pc_shape[0], rates_pc_shape[1], n_components)
    model_arrays["rates_pc"] = rates_pc_reshaped.reshape(rates_pc_shape)

    return model_arrays, pca


def cumulative_variance(
    key,
    state,
    params,
    task_tf,
):
    """
    Computes the cumulative variance across all principal components.

    Args:
        key (PRNGKey): Random number generator for seeding "noise_stream".
        state (flax.training.train_state.TrainState): Model state object.
        params (dict): Model parameters dictionary.
        task_tf (tf.data.Dataset): Training task in TensorFlow Dataset format.

    Returns:
        cumulative_variance (jnp.ndarray): A JAX array containing cumulative variance.
    """
    _, pca = compute_pca(
        key,
        state,
        params,
        task_tf,
    )
    return jnp.cumsum(jnp.array(pca.explained_variance_ratio_))
