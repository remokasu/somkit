from sklearn.datasets import load_digits

import somkit

# Set SOM parameters
x_size = 50
y_size = 50
dynamic_radius = False
random_seed = 42

# SOM_PAK two-stage sequential learning (cf. SOM_PAK command.sh).
# Stage 1 = coarse ordering: radius ~ map diameter, higher alpha, short run
LEARN1_RLEN = 2500
LEARN1_ALPHA = 0.05
LEARN1_RADIUS = 50
# Stage 2 = fine tuning: small radius, lower alpha, longer run.
# NOTE: a 50x50 map x 64-dim data is heavy for sequential training, so the
# fine phase is kept modest (~14 passes). Increase LEARN2_RLEN for better quality.
LEARN2_RLEN = 25000
LEARN2_ALPHA = 0.02
LEARN2_RADIUS = 15

# Load dataset
input_data = load_digits()

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=LEARN1_ALPHA,
    n_func=somkit.functions.gaussian,  # or somkit.functions.bubble, somkit.functions.mexican_hat
    initial_radius=LEARN1_RADIUS,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=10,
    topology="hexagonal",  # or "rectangular"
)

# Normalize the input data
# SOM_PAK trains on raw data (no normalization); disabled for conformance.
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
# mirroring SOM_PAK's two consecutive `vsom` calls.
som.train_pak(
    rlen=LEARN1_RLEN, alpha=LEARN1_ALPHA, radius=LEARN1_RADIUS, neighborhood="bubble",
)
som.train_pak(
    rlen=LEARN2_RLEN, alpha=LEARN2_ALPHA, radius=LEARN2_RADIUS, neighborhood="bubble",
)

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

# Plot the U-Matrix with per-unit vcal labels
som_visualizer.plot_umatrix(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="umatrix_digits.png",
    show=False,
)

# Plot Component Planes showing distribution of each feature
som_visualizer.plot_component_planes(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="component_planes_digits.png",
    show=False,
)

# Plot Hit Map showing data density distribution
som_visualizer.plot_hit_map(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="hit_map_digits.png",
    show=False,
)

# Plot Class Distribution Map showing class boundaries
som_visualizer.plot_class_distribution(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="class_distribution_digits.png",
    show=False,
)

# Plot Sammon's Mapping Projection
som_visualizer.plot_sammon_projection(
    show_nodes=True,
    show_data_points=True,
    show_connections=True,
    show_legend=True,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="sammon_projection_digits.png",
    show=False,
)
