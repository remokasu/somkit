from sklearn.datasets import load_wine

import somkit

# Set SOM parameters
x_size = 10
y_size = 10
batch_size = 1
n_epochs = 100
learning_rate = 0.01
initial_radius = 5.0
dynamic_radius = True
shuffle_each_epoch = True
random_seed = 42
tau = None  # Time constant for decay (None = defaults to n_epochs)

# Load dataset
input_data = load_wine()


# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=learning_rate,
    n_func=somkit.functions.gaussian,  # or somkit.functions.bubble, somkit.functions.mexican_hat
    initial_radius=initial_radius,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=10,
    tau=tau,
    topology="hexagonal",  # or "rectangular"
)

# Shuffle the input data (optional)
som.shuffle_data()

# Normalize the input data
som.standardize_data()
# or, use other normalization methods
# som.normalize_data(method='standard')  # Z-score normalization (mean=0, std=1)
# som.normalize_data(method='minmax')    # Min-Max normalization [0, 1]
# som.normalize_data(method='variance')  # Variance normalization (std=1, preserve mean)

# Initialize the weights using random values
som.initialize_weights_randomly()
# or, initialize the weights using PCA
# som.initialize_weights_with_pca()
# or, initialize the weights using linear mapping (recommended)
# som.initialize_weights_linearly()

# Train the SOM using sequential learning
som.train(n_epochs=n_epochs, batch_size=batch_size, shuffle_each_epoch=shuffle_each_epoch)
# or, train using batch learning
# som.train_batch(n_epochs=n_epochs)

# Evaluate the trained SOM using various metrics
evaluator = somkit.SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using various visualization methods
som_visualizer = somkit.SOMVisualizer(som)

# Plot the U-Matrix with data points
som_visualizer.plot_umatrix(
    show_data_points=True,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="umatrix_wine.png",
    show=False,
)

# Plot Component Planes showing distribution of each feature
som_visualizer.plot_component_planes(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="component_planes_wine.png",
    show=False,
)

# Plot Hit Map showing data density distribution
som_visualizer.plot_hit_map(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="hit_map_wine.png",
    show=False,
)

# Plot Class Distribution Map showing class boundaries
som_visualizer.plot_class_distribution(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="class_distribution_wine.png",
    show=False,
)

# Plot Sammon's Mapping Projection
som_visualizer.plot_sammon_projection(
    show_nodes=True,
    show_data_points=True,
    show_connections=True,
    show_legend=True,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="sammon_projection_wine.png",
    show=False,
)
