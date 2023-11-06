from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine)

from pysom import SOM, SOMEvaluator, SOMPakDataLoader, SOMVisualizer


# Set SOM parameters
x_size = 10
y_size = 10
batch_size = 1
epochs = 100
topology = 'hexagonal'
# topology = 'rectangular'
neighborhood_function = "gaussian"
# neighborhood_function = "bubble"
# neighborhood_function = "mexican_hat"
# neighborhood_function = "cone"
learning_rate = 0.01
initial_radius = 0.02
final_radius = 3
shuffle_each_epoch = True

# Load dataset
input_data = load_iris()
# input_data = load_digits()
# input_data = load_breast_cancer()
# input_data = load_wine()

# Get the input dimension (number of features) of the dataset
input_dim = input_data.data.shape[1]

# Create an instance of SOM with the specified parameters
som = SOM(
    data=input_data,
    x_size=x_size,
    y_size=y_size,
    input_dim=input_dim,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    initial_radius=initial_radius,
    final_radius=final_radius,
    topology=topology,
    neighborhood_function=neighborhood_function,
    shuffle_each_epoch=shuffle_each_epoch
)

som.shuffle_data()

# Standardize the input data
som.standardize_data()

# Initialize the weights using random values
som.initialize_weights_randomly()

# or
# Initialize the weights using PCA
# som.initialize_weights_with_pca()

# Train the SOM using the input data
som.train()

# Evaluate the trained SOM using various metrics
evaluator = SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using the U-Matrix plot
som_visualizer = SOMVisualizer(som)

# plot the U-Matrix with data points
som_visualizer.plot_umatrix(show_data_points=True, save_image=True)
