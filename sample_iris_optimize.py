import optuna
from sklearn.datasets import load_iris

from som import SOM
from som_evaluator import SOMEvaluator


def objective(trial):

    epochs = trial.suggest_int("epochs", 30, 100)
    x_size = trial.suggest_int("x_size", 10, 100)
    y_size = x_size
    # x_size = trial.suggest_int("y_size", 10, 100)
    initial_radius = trial.suggest_float("initial_radius", 0.0001, 0.999)
    final_radius = trial.suggest_float("initial_radius", 1, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.999)
    topology = trial.suggest_categorical("topology", ["hexagonal", "rectangular"])
    neighborhood_function = trial.suggest_categorical("rectangular", ["gaussian", "mexican_hat", "bubble", "cone"])

    # Load the iris dataset
    iris = load_iris()

    # Get the input dimension (number of features) of the dataset
    input_dim = iris.data.shape[1]

    # Create an instance of SOM with the specified parameters
    som = SOM(
        data=iris,
        x_size=x_size,
        y_size=y_size,
        input_dim=input_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        initial_radius=initial_radius,
        final_radius=final_radius,
        topology=topology,
        neighborhood_function=neighborhood_function
    )

    # Standardize the input data
    som.standardize_data()

    # Initialize the weights using random values
    som.initialize_weights_randomly()

    # or
    # Initialize the weights using PCA
    # som.initialize_weights_with_pca()

    # Train the SOM using the input data
    som.train()

    # Evaluate the trained SOM using various metrics
    evaluator = SOMEvaluator(som)
    wcss = evaluator.calculate_wcss()
    silhouette = evaluator.calculate_silhouette_score()
    topological_error = evaluator.calculate_topological_error()

    print("WCSS: ", wcss)
    print("Silhouette Score: ", silhouette)
    print("Topological Error: ", topological_error)

    return wcss, silhouette, topological_error


if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "maximize", "minimize"])
    study.optimize(objective, n_trials=1)
    print(study.best_trials)
