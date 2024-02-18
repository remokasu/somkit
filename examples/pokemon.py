"""
データはこちらの記事から拝借しました。
http://zangesuruneko.blog115.fc2.com/blog-entry-21.html
"""

import somkit

# Set SOM parameters
x_size = 100
y_size = 100
batch_size = 32
shuffle_each_epoch = True
random_seed = 123
dynamic_radius = True

lean1_n_epochs = 100000
lean1_learning_rate = 0.1
lean1_initial_radius = 100

lean2_n_epochs = 100000
lean2_learning_rate = 0.02
lean2_initial_radius = 30


# Load the 'animal.dat' dataset using the SOMPakDataLoader
input_data = somkit.load_som_pak_data("pokemon.dat")

# Create an instance of SOM with the specified parameters
som = somkit.create_trainer(
    data=input_data,
    size=(x_size, y_size),
    learning_rate=lean1_learning_rate,
    n_func=somkit.functions.gaussian,
    initial_radius=lean1_initial_radius,
    dynamic_radius=dynamic_radius,
    random_seed=random_seed,
    checkpoint_interval=100,
)

# Standardize the input data
som.standardize_data()

# Initialize the weights using random values
som.initialize_weights_randomly()
# or, initialize the weights using PCA
# som.initialize_weights_with_pca()

# Train the SOM using the input data
som.train(
    n_epochs=lean1_n_epochs,
    batch_size=batch_size,
    shuffle_each_epoch=shuffle_each_epoch
)

# Save the trained SOM model
som.save_model("pokemon_som_model")

# Evaluate the trained SOM using various metrics
evaluator = somkit.SOMEvaluator(som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using the U-Matrix plot
visualizer = somkit.SOMVisualizer(
    som,
    font_path="./font/NotoSansJP-VariableFont_wght.ttf"
)

# plot the U-Matrix with data points
visualizer.plot_umatrix(
    show_data_points=True,
    show_legend=False,
    file_name="umatrix_pokemon.png"
)



############################################################################################################
# Load the trained SOM model and train it further
#===========================================================================================================

# Load the trained SOM model
loaded_som = somkit.load_trainer(
    "pokemon_som_model",
    learning_rate=lean2_learning_rate,
    n_func=somkit.functions.gaussian,
    initial_radius=lean2_initial_radius
)

# Train the SOM using the input data
loaded_som.train(
    n_epochs=lean2_n_epochs,
    batch_size=batch_size,
    shuffle_each_epoch=shuffle_each_epoch
)

# Evaluate the loaded SOM using various metrics
evaluator = somkit.SOMEvaluator(loaded_som)
wcss = evaluator.calculate_wcss()
silhouette = evaluator.calculate_silhouette_score()
topological_error = evaluator.calculate_topological_error()

print("WCSS: ", wcss)
print("Silhouette Score: ", silhouette)
print("Topological Error: ", topological_error)

# Visualize the SOM using the U-Matrix plot
visualizer = somkit.SOMVisualizer(
    loaded_som,
    font_path="./font/NotoSansJP-VariableFont_wght.ttf"
)

# plot the U-Matrix with data points
visualizer.plot_umatrix(
    show_data_points=True,
    show_legend=False,
    file_name="umatrix_pokemon2.png",
)