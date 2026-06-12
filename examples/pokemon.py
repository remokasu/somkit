"""
データはこちらの記事から拝借しました。
http://zangesuruneko.blog115.fc2.com/blog-entry-21.html
"""

import somkit

# Set SOM parameters
x_size = 100
y_size = 100
random_seed = 123
dynamic_radius = True

# SOM_PAK two-stage sequential learning (cf. SOM_PAK command.sh).
# Stage 1 = coarse ordering: radius ~ map diameter, higher alpha, short run
LEARN1_RLEN = 5000
LEARN1_ALPHA = 0.1
LEARN1_RADIUS = 100
# Stage 2 = fine tuning: small radius, lower alpha, 10x longer run.
# NOTE: a 100x100 map is heavy for sequential training; raise LEARN2_RLEN for
# better quality at the cost of runtime.
LEARN2_RLEN = 50000
LEARN2_ALPHA = 0.02
LEARN2_RADIUS = 30


# Load the 'animal.dat' dataset using the SOMPakDataLoader
input_data = somkit.load_som_pak_data("pokemon.dat")

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=LEARN1_ALPHA,
    n_func=somkit.functions.gaussian,  # or somkit.functions.bubble, somkit.functions.mexican_hat
    initial_radius=LEARN1_RADIUS,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=100,
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

# Save the trained SOM model
som.save_model("pokemon_som_model")

# Evaluate the fine-tuned SOM using various metrics
evaluator = somkit.SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using various visualization methods
visualizer = somkit.SOMVisualizer(
    som, font_path="./font/NotoSansJP-VariableFont_wght.ttf"
)

# Plot the U-Matrix with data points
visualizer.plot_umatrix(
    show_legend=False,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="umatrix_pokemon.png",
    show=False,
)

# Plot Component Planes showing distribution of each feature
visualizer.plot_component_planes(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="component_planes_pokemon.png",
    show=False,
)

# Plot Hit Map showing data density distribution
visualizer.plot_hit_map(
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="hit_map_pokemon.png",
    show=False,
)

# No class-distribution map here: every pokemon has its own unique label
# (663 samples = 663 "classes"), so class-based plots and legends are
# meaningless for this dataset.

# Plot Sammon's Mapping Projection.
# Data points only: projecting all 100x100 = 10,000 node weights is O(n^2)
# per iteration and not feasible at this map size. Legend disabled: one
# entry per pokemon would make the figure unusably tall.
visualizer.plot_sammon_projection(
    show_nodes=False,
    show_data_points=True,
    show_connections=False,
    show_legend=False,
    title=None,  # Optional title for the plot (default: None, no title)
    file_name="sammon_projection_pokemon.png",
    show=False,
)
