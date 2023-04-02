from sklearn.preprocessing import StandardScaler

from som import SOM
from som_pak_data_loader import SOMPakDataLoader
from som_visualizer import SOMVisualizer

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
som = SOM(x_size=10, y_size=10, input_dim=data.shape[1], epochs=200, learning_rate=0.1)

# Train the SOM
som.train(data)

# Visualize the SOM
som_visualizer = SOMVisualizer(som, data, target, target_names)

# If you need to specify a ttf file, write
# som_visualizer = SOMVisualizer(som, data, target, target_names, font_path="./fonts/ipaexg.ttf")

som_visualizer.plot(grid_type='square', label_type='cluster')
som_visualizer.plot(grid_type='square', label_type='block')
som_visualizer.plot(grid_type='hexagonal', label_type='cluster')
som_visualizer.plot(grid_type='hexagonal', label_type='block')
