from sklearn.preprocessing import StandardScaler

from som import SOM
from som_evaluator import SOMEvaluator
from som_pak_data_loader import SOMPakDataLoader
from som_visualizer import SOMVisualizer

x_size = 10
y_size = 10
epochs = 1000
topology = 'rectangular'
# topology = 'hexagonal'
# topology = 'circular'
# topology = 'ring'

neighborhood_function = "gaussian"
# neighborhood_function = "mexican_hat"
# neighborhood_function = "bubble"
# neighborhood_function = "cone"

learning_rate = 0.01
initial_radius = 0.02
final_radius = 3

# Load the 'animal.dat' dataset
loader = SOMPakDataLoader("animal.dat")
animal_data = loader.load_data()

data = animal_data.data
target = animal_data.target
target_names = animal_data.target_names

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Create an instance of SOM
som = SOM(
    x_size=x_size,
    y_size=y_size,
    input_dim=data.shape[1],
    epochs=epochs,
    learning_rate=learning_rate,
    initial_radius=initial_radius,
    final_radius=final_radius,
    topology=topology
)


# Set the input data for the SOM
som.set_data(data)

# Initialize the weights using random values
som.initialize_weights_randomly()

# or
# Initialize the weights using PCA
# som.initialize_weights_with_pca()


# Train the SOM using the input data
som.train()

# Evaluate
evaluator = SOMEvaluator(som)

wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM
som_visualizer = SOMVisualizer(som, data, target, target_names)

# plot
som_visualizer.plot_umatrix(show_data_points=True)