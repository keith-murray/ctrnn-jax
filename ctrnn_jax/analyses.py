"""Analysis functions for trained CT-RNNs."""

from jax import random
import jax.numpy as jnp
from sklearn.decomposition import PCA


def compute_pca(
    key,
    state,
    task_tf,
    n_components,
):
    """
    Function to compute principal component analysis (PCA) on model rates.

    Args:
        key key (PRNGKey): random number generator used for seeding "noise_stream".
        state (flax.training.train_state.TrainState): model state containing parameters.
        task_tf (tf.data.Dataset): training task in the form of TensorFlow Dataset.
        n_components (int): number of principal components to compute.

    Returns:
        A tuple containing two elements:
            model_behavior (dict): python dictionary containing:
                inputs (jnp.ndarray): array of task inputs.
                rates (jnp.ndarray): array of CT-RNN firing rates.
                rates_pc (jnp.ndarray): array of principal components of `rates`.
                outputs (jnp.ndarray): array of model outputs.
            pca (sklearn.decomposition.PCA): sklearn PCA object.
    """
    model_behavior = {
        "inputs": [],
        "rates": [],
        "rates_pc": [],
        "outputs": [],
    }

    for _inputs, _ in task_tf.as_numpy_iterator():
        key, test_key = random.split(key, num=2)
        _outputs, _rates = state.apply_fn(
            {"params": state.params}, _inputs, rngs={"noise_stream": test_key}
        )

        model_behavior["inputs"].append(_inputs)
        model_behavior["rates"].append(_rates)
        model_behavior["outputs"].append(_outputs)

    model_behavior["inputs_list"] = jnp.concatenate(model_behavior["inputs"], axis=0)
    rates_ = jnp.concatenate(model_behavior["rates"], axis=0)
    model_behavior["rates"] = rates_
    model_behavior["outputs"] = jnp.concatenate(model_behavior["outputs"], axis=0)

    rates_reshaped = rates_.reshape(-1, rates_.shape[-1])
    pca = PCA(n_components=n_components)
    pca.fit(rates_reshaped)

    for i in range(rates_.shape[0]):
        model_behavior["rates_pc"].append(pca.transform(rates_[i, :, :]))

    model_behavior["rates_pc"] = jnp.stack(model_behavior["rates_pc"], axis=0)
    return model_behavior, pca
