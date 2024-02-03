import numpy as np
from sklearn.datasets import load_iris

from somkit import create_trainer

# Load the Iris dataset
data = load_iris()


# Online learning test case
def test_online_learning():
    som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)
    assert som.weights is not None, "Weights should be updated after training."


# Batch learning test case
def test_batch_learning():
    som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    som.train(n_epochs=100, batch_size=32, shuffle_each_epoch=True)
    assert som.weights is not None, "Weights should be updated after training."


def test_online_vs_batch_learning():
    online_som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    online_som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)

    batch_som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    batch_som.train(n_epochs=100, batch_size=32, shuffle_each_epoch=True)

    # Check if online and batch SOMs have different weights
    assert np.any(
        online_som.weights != batch_som.weights
    ), "Online and batch learning should result in different weights."


def mean_squared_error(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)


def test_online_vs_batch_learning_numerical():
    online_som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    online_som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)

    batch_som = create_trainer(
        data,
        size=(10, 10),
        learning_rate=0.5,
    )
    batch_som.train(n_epochs=100, batch_size=32, shuffle_each_epoch=True)

    mse = mean_squared_error(online_som.weights, batch_som.weights)
    print(f"Mean squared error between online and batch learning weights: {mse}")

    assert (
        mse > 1e-2
    ), "Online and batch learning should result in numerically different weights."
