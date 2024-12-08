"""Implementation of the `SineWaveGenerator` task."""

from jax import random
import jax.numpy as jnp
import tensorflow as tf


class SineWaveGenerator:
    """A constructor for the sine wave generator task."""

    def __init__(self, key, frequencies, time):
        """
        Initialize the SineWaveGenerator class with given frequencies and time series length.

        Parameters:
        - key (PRNGKey): random number generator used for shuffeling.
        - frequencies (jnp.ndarray): A JAX array of frequencies for generating the time series.
        - time (int): The length of the time series.
        """
        self.key = key
        self.frequencies = frequencies
        self.time = time
        self.time_series = jnp.arange(0, self.time)  # Time points from 0 to `time - 1`

    def get_wave(self, index):
        """
        Generate a single sample of data given an index.

        Parameters:
        - index (int): The index of the desired frequency.

        Returns:
        - x (jnp.ndarray): Input features of shape (time, 1), with constant values.
        - y (jnp.ndarray): Output target of shape (time, 1), a sine wave with the given frequency.
        """
        omega = self.frequencies[index]
        new_index = 100 * omega - 9
        x = 0.25 * jnp.ones(self.time) + new_index / 51
        x = jnp.expand_dims(x, axis=1)
        y = jnp.expand_dims(jnp.sin(omega * self.time_series), axis=1)
        return x, y

    def generate_subkey(self):
        """
        Generate a new subkey using JAX's random module for seeding purposes.

        Returns:
        - subkey (jax.random.PRNGKey): A new PRNGKey for randomness control.
        """
        self.key, subkey = random.split(self.key)
        return subkey

    def generate_jax_tensor(
        self,
    ):
        """
        Generate JAX tensors for features and labels via iterating over `frequencies`.

        Returns:
        - features_tensor (jnp.ndarray): Concatenated features tensor from all sets.
        - labels_tensor (jnp.ndarray): Concatenated labels tensor from all sets.
        """
        features, labels = [], []

        for index in range(len(self.frequencies)):
            x, y = self.get_wave(index)
            features.append(x)
            labels.append(y)

        features_tensor = jnp.stack(features, axis=0)
        labels_tensor = jnp.stack(labels, axis=0)

        return features_tensor, labels_tensor

    def generate_tf_dataset(self, batch_size):
        """
        Create a TensorFlow Dataset object from the provided batch size.

        Returns:
            tf.data.Dataset: A shuffled and batched TensorFlow Dataset object ready for training.
        """
        features_tensor, labels_tensor = self.generate_jax_tensor()
        dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))

        subkey = self.generate_subkey()
        dataset = dataset.shuffle(
            buffer_size=len(features_tensor),
            reshuffle_each_iteration=True,
            seed=subkey[0].item(),
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(2)

        return dataset
