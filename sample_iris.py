from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from som import SOM
from som_visualizer import SOMVisualizer

# Irisデータセットの読み込み
iris = load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

# データの正規化
scaler = StandardScaler()
data = scaler.fit_transform(data)

# SOMのインスタンス生成
som = SOM(x_size=10, y_size=10, input_dim=4, epochs=200, learning_rate=0.1)

# SOMの学習
som.train(data)

# SOMの可視化
som_visualizer = SOMVisualizer(som, data, target, target_names)
# som_visualizer.plot(grid_type='square', label_type='cluster')
# som_visualizer.plot(grid_type='square', label_type='block')
# som_visualizer.plot(grid_type='hexagonal', label_type='cluster')
# som_visualizer.plot(grid_type='hexagonal', label_type='block')
