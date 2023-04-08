# PySOM Documentation

PySOM is a Python implementation of the Self-Organizing Map (SOM), a type of unsupervised learning algorithm used for clustering and visualization of high-dimensional data.

## Table of Contents
1. [Module Overview](#module-overview)
2. [Algorithm Explanation](#algorithm-explanation)
3. [Usage Guide](#usage-guide)

## Module Overview

PySOM consists of the following modules:

1. som.py: The main SOM implementation.
2. som_evaluator.py: A module for evaluating the trained SOM using various metrics.
3. som_pak_data_loader.py: A module for loading data from the SOM_PAK data format.
4. som_topology.py: A module for handling different SOM topologies.
5. som_visualizer.py: A module for visualizing the trained SOM using U-Matrix plots.
6. son_neighborhood_functions.py: A module for defining different neighborhood functions.

## Algorithm Explanation

The Self-Organizing Map (SOM) is an unsupervised learning algorithm that projects high-dimensional data onto a low-dimensional grid, preserving the topological relationships between the data points. The algorithm iteratively updates the weight vectors of the nodes in the grid to better represent the input data.

### Initialization

The weight vectors for all nodes in the grid are initialized using either random values or Principal Component Analysis (PCA). In the case of random initialization, the weight vectors are chosen randomly from the input data. When using PCA, the first two principal components are used to initialize the weight vectors, which can help speed up the convergence of the algorithm.

### Training

The SOM algorithm consists of the following steps:

1. For each epoch:
   a. Select a data point `x` from the input dataset.
   b. Find the Best Matching Unit (BMU), the node with the closest weight vector to the selected data point.
   c. Update the weight vectors of the BMU and its neighbors.

The BMU for a given input data point `x` is the node with the smallest Euclidean distance between its weight vector `w` and the input data point:

$$
\text{BMU} = \text{argmin}_i || x - w_i ||
$$

The weight update formula is as follows:

$$
w_i(t+1) = w_i(t) + \eta(t) \cdot h_{ij}(t) \cdot (x - w_i(t))
$$

Where:
- $w_i(t)$ is the weight vector of node $i$ at time $t$.
- $\eta(t)$ is the learning rate at time $t$.
- $h_{ij}(t)$ is the neighborhood function value between node $i$ and the BMU $j$ at time $t$.
- $x$ is the input data point.
- $t$ is the current iteration.

The learning rate and neighborhood function values decrease over time to allow the algorithm to converge. The learning rate typically starts at a high value (e.g., 0.1) and decreases exponentially over time:

$$
\eta(t) = \eta_0 \cdot \exp(-t / \tau)
$$

Where $\eta_0$ is the initial learning rate and $\tau$ is a time constant.

The neighborhood function determines the extent to which the BMU's neighbors are updated. Common neighborhood functions include Gaussian, Bubble, Mexican Hat, and Cone. The neighborhood function value $h_{ij}(t)$ depends on the distance between node $i$ and the BMU $j$ and decreases over time. For example, the Gaussian neighborhood function is defined as:

$$
h_{ij}(t) = \exp \left(-\frac{||r_i - r_j||^2}{2 \cdot \sigma^2(t)} \right)
$$

Where $r_i$ and $r_j$ are the grid coordinates of nodes $i$ and $j$, respectively, and $\sigma(t)$ is the neighborhood width at time $t$.

The neighborhood width typically starts at a high value (e.g., half the grid size) and decreases exponentially over time:

$$
\sigma(t) = \sigma_0 \cdot \exp(-t / \tau)
$$

Where $\sigma_0$ is the initial neighborhood width.

### Example Code

Here's an example of how the weight update step is implemented in code:

~~~ python
def update_weights(self, x, bmu_idx, epoch):
    bmu_coords = self.get_node_coordinates(bmu_idx)
    for i, w in enumerate(self.weights):
        node_coords = self.get_node_coordinates(i)
        distance = self.distance_calculator(bmu_coords, node_coords)
        neighborhood_val = self.neighborhood_function(distance, epoch)
        learning_rate = self.calculate_learning_rate(epoch)
        self.weights[i] += learning_rate * neighborhood_val * (x - w)
~~~

### Evaluation Metrics

To evaluate the performance of the trained SOM, various metrics can be used, such as the Within-Cluster Sum of Squares (WCSS), Silhouette Score, and Topological Error.

- **WCSS**: The sum of the squared distances between each data point and its corresponding BMU. A lower WCSS indicates better clustering performance.
- **Silhouette Score**: A measure of the quality of the clustering, where higher values indicate better-defined clusters.
- **Topological Error**: The proportion of data points for which the first and second BMUs are not adjacent in the grid. A lower topological error indicates better preservation of the input data's topological relationships.


## Usage Guide

To use PySOM, follow these steps:

1. Import the required modules.
2. Set the SOM parameters (grid size, epochs, topology, neighborhood function, learning rate, initial radius, final radius).
3. Load the input dataset.
4. Get the input dimension (number of features) of the dataset.
5. Create an instance of SOM with the specified parameters.
6. Standardize the input data.
7. Initialize the weights using random values or PCA.
8. Train the SOM using the input data.
9. Evaluate the trained SOM using various metrics (WCSS, Silhouette Score, Topological Error).
10. Visualize the SOM using the U-Matrix plot.

For a complete example, refer to the provided [usage example](#usage-example) in this documentation.
