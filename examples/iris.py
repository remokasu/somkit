from sklearn.datasets import load_iris

import somkit

# Set SOM parameters
x_size = 10
y_size = 10
batch_size = 1
n_epochs = 100
learning_rate = 0.01
initial_radius = 1.0
dynamic_radius = True
shuffle_each_epoch = True
random_seed = 42

# Load dataset
input_data = load_iris()

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=learning_rate,
    n_func=somkit.functions.gaussian,
    initial_radius=initial_radius,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=10,
)

# Shuffle the input data
som.shuffle_data()

# Standardize the input data
som.standardize_data()

# Initialize the weights using random values
som.initialize_weights_randomly()
# or, initialize the weights using PCA
# som.initialize_weights_with_pca()

# Train the SOM using the input data
som.train(n_epochs=n_epochs, batch_size=batch_size, shuffle_each_epoch=True)

# Save the trained SOM model
som.save_model("iris_som_model")
print("radius: ", som.get_radius())

# Evaluate the trained SOM using various metrics
evaluator = somkit.SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using the U-Matrix plot
som_visualizer = somkit.SOMVisualizer(som)

# plot the U-Matrix with data points
som_visualizer.plot_umatrix(show_data_points=True, show_legend=False, file_name="umatrix_iris.png")
