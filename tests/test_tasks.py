"""Tests for sample tasks."""

import jax.numpy as jnp
from jax import random
import tensorflow as tf

from ctrnn_jax.tasks import SineWaveGenerator


def test_get_wave():
    """
    Test that SineWaveGenerator outputs are as expected.
    """
    key = random.PRNGKey(0)
    frequencies = jnp.array([0.1, 0.2])
    time = 10
    dataset = SineWaveGenerator(key, frequencies, time)

    x, y = dataset.get_wave(0)

    # Check shapes
    assert x.shape == (time, 1), "Feature shape is incorrect."
    assert y.shape == (time, 1), "Label shape is incorrect."

    # Check values for the first frequency (0.1)
    expected_x = 0.25 * jnp.ones(time) + (100 * 0.1 - 9) / 51
    expected_x = jnp.expand_dims(expected_x, axis=1)
    expected_y = jnp.expand_dims(jnp.sin(0.1 * jnp.arange(0, time)), axis=1)

    assert jnp.allclose(
        x, expected_x
    ), "Generated X values do not match the expected values."
    assert jnp.allclose(
        y, expected_y
    ), "Generated y values do not match the expected values."


def test_generate_jax_tensor():
    """
    Test that generate_jax_tensor produces correctly shaped concatenated tensors.
    """
    key = random.PRNGKey(0)
    frequencies = jnp.array([0.1, 0.2, 0.3])
    time = 10
    dataset = SineWaveGenerator(key, frequencies, time)

    features_tensor, labels_tensor = dataset.generate_jax_tensor()

    # Check shapes
    expected_feature_shape = (len(frequencies), time, 1)
    expected_label_shape = (len(frequencies), time, 1)
    assert (
        features_tensor.shape == expected_feature_shape
    ), f"Feature shape incorrect. Got {features_tensor.shape}, expected {expected_feature_shape}."
    assert (
        labels_tensor.shape == expected_label_shape
    ), f"Label shape incorrect. Got {labels_tensor.shape}, expected {expected_label_shape}."

    # Check values for first frequency (0.1)
    index = 0
    expected_x, expected_y = dataset.get_wave(index)
    assert jnp.allclose(
        features_tensor[index], expected_x
    ), "Generated feature values for the first wave do not match."
    assert jnp.allclose(
        labels_tensor[index], expected_y
    ), "Generated label values for the first wave do not match."

    # Check values for the second frequency (0.2)
    index = 1
    expected_x, expected_y = dataset.get_wave(index)
    assert jnp.allclose(
        features_tensor[index], expected_x
    ), "Generated feature values for the second wave do not match."
    assert jnp.allclose(
        labels_tensor[index], expected_y
    ), "Generated label values for the second wave do not match."


def test_generate_tf_dataset():
    """
    Test that generate_tf_dataset creates a TensorFlow Dataset with the correct structure.
    """
    key = random.PRNGKey(0)
    frequencies = jnp.array([0.1, 0.2, 0.3, 0.4])
    time = 10
    batch_size = 2
    dataset = SineWaveGenerator(key, frequencies, time)

    tf_dataset = dataset.generate_tf_dataset(batch_size)
    dataset_list = list(tf_dataset.as_numpy_iterator())

    # Check that the dataset is of the correct type
    assert isinstance(
        tf_dataset, tf.data.Dataset
    ), "The generated object is not a TensorFlow Dataset."

    # Check that the number of batches is correct
    expected_batches = len(frequencies) // batch_size
    assert (
        len(dataset_list) == expected_batches
    ), f"Number of batches is incorrect. Got {len(dataset_list)}, expected {expected_batches}."

    # Check the shapes of the first batch
    batch_x, batch_y = dataset_list[0]
    assert batch_x.shape == (
        batch_size,
        time,
        1,
    ), f"Batch feature shape is incorrect. Got {batch_x.shape}, expected {(batch_size, time, 1)}."
    assert batch_y.shape == (
        batch_size,
        time,
        1,
    ), f"Batch label shape is incorrect. Got {batch_y.shape}, expected {(batch_size, time, 1)}."

    # Check for randomness in batching by generating the dataset again
    new_dataset = dataset.generate_tf_dataset(batch_size)
    new_dataset_list = list(new_dataset.as_numpy_iterator())
    assert any(
        not jnp.array_equal(x1, x2)
        for (x1, y1), (x2, y2) in zip(dataset_list, new_dataset_list)
    ), "The datasets are not shuffled as expected."
