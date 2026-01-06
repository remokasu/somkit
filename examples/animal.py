import somkit

# Set SOM parameters
x_size = 10
y_size = 10
batch_size = 1
n_epochs = 500
learning_rate = 0.01
initial_radius = 5.0
shuffle_each_epoch = False
dynamic_radius = True
random_seed = 42

# Load the 'animal.dat' dataset using the SOMPakDataLoader
animal_data = somkit.load_som_pak_data("animal.dat")

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=animal_data,
    size=(x_size, y_size),
    learning_rate=learning_rate,
    n_func=somkit.functions.gaussian,
    # n_func=somkit.functions.bubble,
    # n_func=somkit.functions.mexican_hat,
    initial_radius=initial_radius,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=10,
    topology="hexagonal",
    # topology="rectangular",
)

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
som.train(
    n_epochs=n_epochs, batch_size=batch_size, shuffle_each_epoch=shuffle_each_epoch
)
# or, train using batch learning
# som.train_batch(n_epochs=n_epochs)

# Save the trained SOM model
som.save_model("animal_som_model")

# Evaluate the trained SOM using various metrics
evaluator = somkit.SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("radius: ", som.get_radius())
print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using various visualization methods
visualizer = somkit.SOMVisualizer(som)

# Plot the U-Matrix with data points
visualizer.plot_umatrix(
    show_data_points=True,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="umatrix_animal.png",
    show=False
)

# Plot Component Planes showing distribution of each feature
visualizer.plot_component_planes(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="component_planes_animal.png",
    show=False
)

# Plot Hit Map showing data density distribution
visualizer.plot_hit_map(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="hit_map_animal.png",
    show=False
)

# Plot Class Distribution Map showing class boundaries
visualizer.plot_class_distribution(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="class_distribution_animal.png",
    show=False
)

# Plot Sammon's Mapping Projection - showing both nodes and data
visualizer.plot_sammon_projection(
    show_nodes=True,
    show_data_points=True,
    show_connections=True,
    show_legend=True,
    show_labels=False,
    node_size=200,
    data_point_size=100,
    connection_style="spring",  # "spring" for distance-based thickness, "line" for uniform
    colormap="tab10",
    max_iter=500,
    learning_rate=0.2,
    random_state=random_seed,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="sammon_projection_animal.png",
    show=False,
)
