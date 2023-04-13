import numpy as np
from sklearn.datasets import load_iris

from som import SOM

# Load the Iris dataset
data = load_iris()


# Online learning test case
def test_online_learning():
    som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=1, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    som.train()
    assert som.weights is not None, "Weights should be updated after training."


# Batch learning test case
def test_batch_learning():
    som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=32, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    som.train()
    assert som.weights is not None, "Weights should be updated after training."


def test_online_vs_batch_learning():
    online_som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=1, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    online_som.train()

    batch_som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=32, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    batch_som.train()

    # Check if online and batch SOMs have different weights
    assert np.any(online_som.weights != batch_som.weights), "Online and batch learning should result in different weights."


def mean_squared_error(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)


def test_online_vs_batch_learning_numerical():
    online_som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=1, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    online_som.train()

    batch_som = SOM(data, x_size=10, y_size=10, input_dim=4, batch_size=32, epochs=100, learning_rate=0.5, initial_radius=5, final_radius=1)
    batch_som.train()

    mse = mean_squared_error(online_som.weights, batch_som.weights)
    print(f"Mean squared error between online and batch learning weights: {mse}")

    assert mse > 1e-2, "Online and batch learning should result in numerically different weights."
