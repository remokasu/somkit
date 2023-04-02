from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from som import SOM
from som_visualizer import SOMVisualizer

# Load the iris dataset
iris = load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Create an instance of SOM
som = SOM(x_size=10, y_size=10, input_dim=data.shape[1], epochs=200, learning_rate=0.1)

# Train the SOM
som.train(data)

# Visualize the SOM
som_visualizer = SOMVisualizer(som, data, target, target_names)

# If you need to specify a ttf file, write
# som_visualizer = SOMVisualizer(som, data, target, target_names, font_path="./fonts/ipaexg.ttf")

## The plot is displayed in a grid pattern, and the labels are shown for each cluster.
som_visualizer.plot(grid_type='square', label_type='cluster')

## The plot is displayed in a grid pattern, and the labels are shown for each block.
# som_visualizer.plot(grid_type='square', label_type='block')

## The plot is displayed with a hexagonal grid pattern, and the labels are shown for each cluster.
# som_visualizer.plot(grid_type='hexagonal', label_type='cluster')

## The plot is displayed with a hexagonal grid pattern, and the labels are shown for each block.
# som_visualizer.plot(grid_type='hexagonal', label_type='block')
