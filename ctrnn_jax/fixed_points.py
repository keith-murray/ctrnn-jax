"""Functions to compute and analyze fixed-points of CT-RNNs."""

import jax
from jax import random
import jax.numpy as jnp
import optax

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN


def compute_fixed_point_loss(
    module,
    params,
    hidden_states,
    inputs,
):
    """
    Compute fixed-point loss for a batch of hidden states.

    Args:
        module (flax.linen.recurrent.RNNCellBase): RNN cell.
        params (dict): model parameters dictionary.
        inputs (jnp.ndarray): input array with shape (batch_size, input_dim).
        hidden_states (jnp.ndarray): hidden states to evaluate with shape (batch_size, hidden_dim).

    Returns:
        loss (jnp.ndarray): array of fixed-point losses for each hidden state.
    """
    next_hidden_states, _ = module.apply(
        params,
        hidden_states,
        inputs,
        rngs={"noise_stream": random.PRNGKey(0)},
    )
    loss = 0.5 * jnp.sum((hidden_states - next_hidden_states) ** 2, axis=1)
    return loss


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
def find_fixed_points(
    module,
    params,
    initial_states,
    inputs,
    optimizer,
    tolerance=1e-6,
    max_steps=1000,
):
    """
    Find fixed points by minimizing the fixed-point loss.

    Args:
        module (flax.linen.recurrent.RNNCellBase): RNN cell.
        params (dict): model parameters dictionary.
        initial_states (jnp.ndarray): initial states for optimization.
        inputs (jnp.ndarray): constant input to evaluate fixed points.
        optimizer (optax.GradientTransformation): optimizer for gradient descent.
        tolerance (float): tolerance for stopping optimization.
        max_steps (int): maximum number of optimization steps.

    Returns:
        fixed_points (jnp.ndarray): array of fixed points found.
        loss_history (list): loss values during optimization.
        squared_speeds (jnp.ndarray): final speeds (loss values) of fixed points.
    """

    def loss_fn(h):
        return jnp.mean(compute_fixed_point_loss(module, params, h, inputs))

    opt_state = optimizer.init(initial_states)
    loss_history = []

    for _ in range(max_steps):
        loss, grads = jax.value_and_grad(loss_fn)(initial_states)
        updates, opt_state = optimizer.update(grads, opt_state)
        initial_states = optax.apply_updates(initial_states, updates)

        loss_history.append(loss)
        if loss < tolerance:
            break

    fixed_points = initial_states
    squared_speeds = compute_fixed_point_loss(module, params, fixed_points, inputs)
    return fixed_points, loss_history, squared_speeds


def analyze_fixed_points(module, params, fixed_points, inputs):
    """
    Compute Jacobians and analyze eigenvalues at fixed points.

    Args:
        module (flax.linen.recurrent.RNNCellBase): RNN cell.
        params (dict): model parameters dictionary.
        fixed_points (jnp.ndarray): fixed points to analyze.
        inputs (jnp.ndarray): constant input at which fixed points are computed.

    Returns:
        analysis_results (dict): dictionary containing:
            - 'eigenvalues': eigenvalues of Jacobians.
            - 'eigenvectors': eigenvectors of Jacobians.
    """
    rec_jacobian = jax.jacobian(module.apply, argnums=1)
    eigenvalues = []
    eigenvectors = []

    for fp in range(fixed_points.shape[0]):
        j_h, _ = rec_jacobian(
            params,
            fixed_points[fp, :],
            inputs,
            rngs={"noise_stream": random.PRNGKey(0)},
        )
        eigvals, eigvecs = jnp.linalg.eig(j_h)
        eigenvalues.append(eigvals)
        eigenvectors.append(eigvecs)

    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def find_unique_fixed_points(hidden_states, distance_threshold=0.1, min_samples=5):
    """
    Identify unique fixed points from optimized hidden states using a distance-based
    clustering approach.

    Args:
        hidden_states (jnp.ndarray): Array of shape (num_states, state_dim) of fixed points.
        distance_threshold (float): Distance threshold to consider points as being one fixed point.
        min_samples (int): Number of fixed points needed to be considered as unique.

    Returns:
        jnp.ndarray: Array of unique fixed points of shape (num_fixed_points, state_dim).
    """
    pairwise_distances = pdist(hidden_states, metric="euclidean")
    distance_matrix = squareform(pairwise_distances)

    clustering = DBSCAN(
        eps=distance_threshold, min_samples=min_samples, metric="precomputed"
    )
    labels = clustering.fit_predict(distance_matrix)

    unique_fixed_points = []
    for cluster_label in jnp.unique(labels):
        cluster_points = hidden_states[labels == cluster_label]
        cluster_center = jnp.mean(
            cluster_points, axis=0
        )  # Compute cluster center (mean)
        unique_fixed_points.append(cluster_center)

    return jnp.stack(unique_fixed_points, axis=0)
