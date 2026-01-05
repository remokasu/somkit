import somkit
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()

# Create SOM with RECTANGULAR topology
som_rect = somkit.create_trainer(
    data=data,
    size=(5, 5),
    learning_rate=0.01,
    n_func=somkit.functions.gaussian,
    initial_radius=2.0,
    dynamic_radius=True,
    random_seed=42,
    topology="rectangular"  # NEW PARAMETER
)

# Standardize and initialize
som_rect.standardize_data()
som_rect.initialize_weights_randomly()

# Train
som_rect.train(n_epochs=100, batch_size=1)

print(f"Topology: {som_rect.topology.get_name()}")
print("Training complete!")

# Visualize
visualizer = somkit.SOMVisualizer(som_rect)
visualizer.plot_umatrix(show_data_points=True, file_name="test_rect_umatrix.png", show=False)
visualizer.plot_hit_map(file_name="test_rect_hitmap.png", show=False)
visualizer.plot_component_planes(file_name="test_rect_components.png", show=False)
visualizer.plot_class_distribution(file_name="test_rect_class_dist.png", show=False)

print("Rectangular topology visualization complete!")
