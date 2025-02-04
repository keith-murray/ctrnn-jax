# CT-RNN Examples
This directory contains a variety of examples demonstrating how to use the `ctrnn_jax` package.

## Dynamical systems
The `dynamics_ctrnn.py` script contains examples of CT-RNNs to producing a variety of nonlinear phenomena. The parameters of these models were taken from "[On the Dynamics of Small Continuous-Time Recurrent Neural Networks](https://doi.org/10.1177/105971239500300405)".

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/ctrnn_dynamics.png" alt="ctrnn_dynamics" width="600"></img>
</div>

## Training CT-RNNs
The `train_ctrnn.py` script demonstrates how to train CT-RNNs on a TensorFlow dataset.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/loss_curve.png" alt="loss_curve" width="400"></img>
</div>

## PCA of CT-RNNs
The `pca_ctrnn.py` script demonstrates how run PCA on a task-optimized CT-RNN to visualize its low-dimensional dynamics.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/pca_plot.png" alt="pca_plot" width="400"></img>
</div>

## Training CT-RNNs
To analyze the stability of fixed-points learned in task-optimized CT-RNNs, refer to functions in `ctrnn_jax/fixed_points.py`. The `fixed_points_ctrnn.py` script demonstrates how to use these functions to find and analyze fixed points.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-jax/blob/main/results/fixed_points.png" alt="fixed_points" width="600"></img>
</div>
