import optuna
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from som import SOM
from som_evaluator import SOMEvaluator
from som_visualizer import SOMVisualizer


def objective(trial):

    topology = 'rectangular'

    # epochs = trial.suggest_int("epochs", 30, 100)
    # x_size = trial.suggest_int("x_size", 10, 100)
    # y_size = trial.suggest_int("y_size", 10, 100)
    initial_radius = trial.suggest_float("initial_radius", 0.0001, 0.999)
    final_radius = trial.suggest_float("initial_radius", 1, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.999)
    neighborhood_function = trial.suggest_categorical("neighborhood_function", ["gaussian", "mexican_hat", "bubble", "cone"])

    epochs = 100
    x_size = 10
    y_size = 10
    # initial_radius = 0.02
    # final_radius = 3
    # learning_rate = 0.01

    # Load the iris dataset
    iris = load_iris()
    data = iris.data
    # target = iris.target  # for visualizer
    # target_names = iris.target_names  # for visualizer

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
        topology=topology,
        neighborhood_function=neighborhood_function
    )

    som.set_data(data)
    som.initialize_weights_with_pca()
    som.train()

    evaluator = SOMEvaluator(som)
    wcss = evaluator.calculate_wcss()  # to minimize
    silhouette = evaluator.calculate_silhouette_score()  # to maximize [-1 ~ 1]
    topological_error = evaluator.calculate_topological_error() # to minimize

    print("WCSS: ", wcss)
    print("Silhouette Score: ", silhouette)
    print("Topological Error: ", topological_error)

    return wcss, silhouette, topological_error


if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "maximize", "minimize"])
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
