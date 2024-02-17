from sklearn.datasets import load_breast_cancer

import somkit

# Set SOM parameters
x_size = 20
y_size = 20
batch_size = 1
n_epochs = 1000
learning_rate = 0.01
n_radius = 1.0
shuffle_each_epoch = True

# Load dataset
input_data = load_breast_cancer()

# Get the input dimension (number of features) of the dataset

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=learning_rate,
    n_func=somkit.functions.gaussian,
    n_radius=n_radius,
)

som.shuffle_data()

# Standardize the input data
som.standardize_data()

# Initialize the weights using random values
som.initialize_weights_randomly()
# or, initialize the weights using PCA
# som.initialize_weights_with_pca()

# Train the SOM using the input data
som.train(n_epochs=n_epochs, batch_size=batch_size, shuffle_each_epoch=True)

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
som_visualizer.plot_umatrix(
    show_data_points=True, file_name="umatrix_breast_cancer.png"
)
