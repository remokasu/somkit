import somkit

# Set SOM parameters
x_size = 10
y_size = 10
dynamic_radius = True
random_seed = 123  # SOM_PAK randinit seed (command.sh RAND_SEED)

# SOM_PAK two-stage sequential learning (cf. SOM_PAK command.sh for animal)
# Stage 1 = coarse ordering: radius ~ map diameter, higher alpha, short run
LEARN1_RLEN = 1000
LEARN1_ALPHA = 0.05
LEARN1_RADIUS = 10
# Stage 2 = fine tuning: small radius, lower alpha, 10x longer run
LEARN2_RLEN = 10000
LEARN2_ALPHA = 0.02
LEARN2_RADIUS = 3

# Load the 'animal.dat' dataset using the SOMPakDataLoader
animal_data = somkit.load_som_pak_data("animal.dat")

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=animal_data,
    size=(x_size, y_size),
    learning_rate=LEARN1_ALPHA,
    n_func=somkit.functions.gaussian,
    # n_func=somkit.functions.bubble,
    # n_func=somkit.functions.mexican_hat,
    initial_radius=LEARN1_RADIUS,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=10,
    topology="hexagonal",
    # topology="rectangular",
)

# Normalize the input data
# SOM_PAK trains on raw data (command.sh does no normalization); disabled for conformance.
# som.standardize_data()
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

# Train the SOM the SOM_PAK way: two-stage sequential learning (train_pak x2).
# The same trainer is reused, so stage 2 continues from stage 1's weights,
# mirroring SOM_PAK's two consecutive `vsom` calls in command.sh.
# seed=1 matches `vsom -rand 1` in command.sh.
som.train_pak(
    rlen=LEARN1_RLEN, alpha=LEARN1_ALPHA, radius=LEARN1_RADIUS,
    neighborhood="bubble", seed=1,
)
som.train_pak(
    rlen=LEARN2_RLEN, alpha=LEARN2_ALPHA, radius=LEARN2_RADIUS,
    neighborhood="bubble", seed=1,
)

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
