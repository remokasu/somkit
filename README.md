# PySOM: A Python Implementation of Self-Organizing Maps (SOMs)

## Overview

PySOM is a simple implementation of Self-Organizing Maps (SOMs) in Python. This library provides an easy-to-use interface to train and visualize SOMs on various datasets. PySOM can be used for clustering, data visualization, and dimensionality reduction tasks.

## Usage

To get started, follow these steps:

1. Install the required dependencies:

~~~ bash
pip install numpy scikit-learn matplotlib
~~~

2. Clone this repository:

~~~ bash
git clone https://github.com/remokasu/pysom.git
cd pysom
~~~


3. Train and visualize a SOM with the sample datasets provided:

- For the Iris dataset:
~~~ bash
python sample_iris.py
~~~

- For the animal dataset:

~~~ bash
python sample_animal.py
~~~

## File Descriptions

- `animal.dat`: A sample dataset containing animal features in the SOM_PAK format.
- `sample_animal.py`: A sample program that trains a SOM using the `animal.dat` dataset and visualizes the results.
- `sample_iris.py`: A sample program that trains a SOM using the Iris dataset and visualizes the results.
- `som_pak_data_loader.py`: A module for loading data in the SOM_PAK format and converting it to a format suitable for training with PySOM.
- `som_visualizer.py`: A module for visualizing the trained SOM, including the grid and cluster/block labels.
- `som.py`: The main module containing the implementation of the Self-Organizing Map algorithm.

Feel free to customize the sample programs or create your own to suit your specific needs.

## Additional Notes

You can modify the parameters of the SOM, such as the size of the grid, number of epochs, and learning rate, to optimize the performance for your specific dataset. Experiment with different settings to find the best configuration for your use case.

## Dependencies and Licenses

This project uses the following external libraries:

- numpy: Licensed under the [NumPy License](https://numpy.org/doc/stable/license.html)
- scikit-learn: Licensed under the [New BSD License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
- matplotlib: Licensed under the [Matplotlib License](https://matplotlib.org/stable/users/license.html)

## Acknowledgments

This software includes some code and content provided by OpenAI's ChatGPT. We would like to thank the OpenAI team for their support and assistance in the development of this project. Please note that any code or content provided by ChatGPT is subject to OpenAI's copyright and usage terms.
