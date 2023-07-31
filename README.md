# PySOM: A Python Implementation of Self-Organizing Maps (SOMs)

## Overview

PySOM is a simple implementation of Self-Organizing Maps (SOMs) in Python. This library provides an easy-to-use interface to train and visualize SOMs on various datasets. PySOM can be used for clustering, data visualization, and dimensionality reduction tasks.
<div style="text-align: center;">
<img width="250" src="doc/fig_001.png">
</div>


## Usage

To get started, follow these steps:

1. Install

~~~ bash
git clone https://github.com/remokasu/pysom.git
cd pysom
python -m setup.py install
~~~


2. Train and visualize a SOM with the sample datasets provided:

- For the animal dataset:
~~~ bash
cd sample
python sample_animal.py
~~~

## File Descriptions

- `animal.dat`: A sample dataset containing animal features in the SOM_PAK format.
- `sample_animal.py`: A sample program that trains a SOM using the `animal.dat` dataset and visualizes the results.
- `som.py`: The main SOM implementation.
- `som_evaluator.py`: A module for evaluating the trained SOM using various metrics.git
- `som_pak_data_loader.py`: A module for loading data from the SOM_PAK data format.
- `som_topology.py`: A module for handling different SOM topologies.
- `som_visualizer.py`: A module for visualizing the trained SOM using U-Matrix plots.
- `son_neighborhood_functions.py`: A module for defining different neighborhood functions.

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
