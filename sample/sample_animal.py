from pysom import SOM, SOMEvaluator, SOMPakDataLoader, SOMVisualizer

# Set SOM parameters
x_size = 10
y_size = 10
batch_size = 32
epochs = 1000
topology = 'hexagonal'
# topology = 'rectangular'
neighborhood_function = "bubble"
# neighborhood_function = "gaussian"
# neighborhood_function = "mexican_hat"
# neighborhood_function = "cone"
learning_rate = 0.01
initial_radius = 0.02
final_radius = 3
shuffle_each_epoch = True

# Load the 'animal.dat' dataset using the SOMPakDataLoader
loader = SOMPakDataLoader("animal.dat")
animal_data = loader.load_data()

# Get the input dimension (number of features) of the dataset
input_dim = animal_data.data.shape[1]

# Create an instance of SOM with the specified parameters
som = SOM(
    data=animal_data,
    x_size=x_size,
    y_size=y_size,
    input_dim=input_dim,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    initial_radius=initial_radius,
    final_radius=final_radius,
    topology=topology,
    shuffle_each_epoch=shuffle_each_epoch
)

# Standardize the input data
som.standardize_data()

# Initialize the weights using PCA
som.initialize_weights_with_pca()

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
som_visualizer.plot_umatrix(show_data_points=True)
