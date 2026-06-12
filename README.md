# somkit: A Python Library for Self-Organizing Maps

<div align="center">
  <table>
    <tr>
      <td><img width="250" src="doc/fig_top_1.png" alt="Figure 1"></td>
      <td><img width="250" src="doc/fig_top_2.png" alt="Figure 2"></td>
    </tr>
    <tr>
      <td><img width="250" src="doc/fig_top_3.png" alt="Figure 3"></td>
      <td><img width="250" src="doc/fig_top_4.png" alt="Figure 4"></td>
    </tr>
  </table>
</div>

## Overview

`somkit` is a Python library for Self-Organizing Maps.

## Features

- **Sequential training** via `train_pak`: `rlen` total steps, per-step linear decay (alpha to 0, radius to 1), `bubble`/`gaussian` neighborhood, `linear`/`inverse_t` alpha schedule
- **Two-phase training** (coarse ordering + fine tuning) via `train_two_phase` or two `train_pak` calls
- **Best-map selection** via `SOMTrainer.vfind`: train `n_trials` maps with different seeds, keep the lowest quantization error
- **Per-sample BMU / quantization error** via `compute_visual` / `write_vis` (`.vis` output)
- **Training snapshots**: save the codebook every N steps as numbered `.cod` files
- **File interchange**: read/write `.cod` codebook files (`load_cod`/`save_cod`), load `.dat` data files (`load_som_pak_data`)
- **Per-sample metadata** via `SOMData`: missing-component masks (`x` fields), sample weights, fixed-point BMUs
- **`vcal`-equivalent label calibration** (`calibrate_labels`)
- **Reproducible seeding** via `OrandRNG` (the `orand` linear congruential generator)
- **Multiple topology support**: hexagonal and rectangular
- **Visualization**:
  - U-Matrix (interpolated grid with per-unit calibration labels)
  - Component planes, hit map, class distribution map
  - Sammon's Mapping projection
- **Evaluation metrics**: quantization error, WCSS, Silhouette Score, Topological Error

## Installation

```bash
git clone https://github.com/remokasu/somkit.git
cd somkit
pip install -e .
```

## Quick Start

The canonical usage is two-phase training (coarse ordering then fine tuning). The example below uses the Iris dataset via a `DatasetWrapper`, which adapts any object with `.data`, `.target`, and `.target_names` attributes.

```python
import somkit
from sklearn.datasets import load_iris

data = load_iris()

som = somkit.create_trainer(
    data=data,           # sklearn Bunch, DatasetWrapper, ndarray, or SOMData
    size=(10, 10),
    learning_rate=0.05,
    topology="hexagonal",  # or "rectangular"
)

som.initialize_weights_randomly()  # randinit: per-component [min, max] range

# Two-phase training
som.train_two_phase(
    phase1=dict(rlen=1000, alpha=0.05, radius=10.0, neighborhood="bubble", seed=1),
    phase2=dict(rlen=10000, alpha=0.02, radius=3.0, neighborhood="bubble", seed=1),
)

# Evaluate
evaluator = somkit.SOMEvaluator(som)
print("WCSS:", evaluator.calculate_wcss())
print("Silhouette Score:", evaluator.calculate_silhouette_score())
print("Topological Error:", evaluator.calculate_topological_error())

# Visualize
visualizer = somkit.SOMVisualizer(som)
visualizer.plot_umatrix()              # umat style with vcal labels
visualizer.plot_component_planes()
visualizer.plot_hit_map()
visualizer.plot_class_distribution()
```

## SOM_PAK Compatibility

The core algorithms and the `.cod` / `.dat` / `.vis` file formats are tested against reference outputs of **SOM_PAK 3.1**, the original SOM implementation by the Kohonen lab. The reference files live under `test/golden/`, and the test suite compares somkit's outputs against them; see the tests there for exactly what is covered and with which tolerances.

## Training

### `train_pak`

`train_pak` runs `rlen` total steps, presenting one sample per step by cycling the data with optional shuffling. Learning rate and radius decay **per step**:

- `alpha` decays linearly from the initial value to 0
- `radius` decays linearly from the initial value to 1 (floor)

```python
som.train_pak(
    rlen=10000,            # total steps
    alpha=0.02,            # initial learning rate
    radius=3.0,            # initial neighborhood radius (decays to 1)
    alpha_type="linear",   # "linear" (default) or "inverse_t"
    neighborhood="bubble", # "bubble" (default) or "gaussian"
    seed=1,                # seed for the sample presentation order
)
```

### Two-phase training

The classic SOM schedule trains in two phases: a short coarse-ordering run with a large radius, followed by a longer fine-tuning run with a small radius. `train_two_phase` is syntactic sugar that calls `train_pak` twice on the same trainer, so phase 2 continues from phase 1's weights.

```python
som.train_two_phase(
    phase1=dict(rlen=1000, alpha=0.05, radius=10.0, neighborhood="bubble", seed=1),
    phase2=dict(rlen=10000, alpha=0.02, radius=3.0, neighborhood="bubble", seed=1),
)
```

The `examples/animal.py` file uses these exact parameters.

### Best-map selection (`vfind`)

SOM results depend on the random initialization, so a standard workflow trains several maps with different seeds and keeps the one with the smallest quantization error. `SOMTrainer.vfind` runs that loop:

```python
best = somkit.SOMTrainer.vfind(
    data, (10, 10),
    phase1=dict(rlen=1000, alpha=0.05, radius=10.0),
    phase2=dict(rlen=10000, alpha=0.02, radius=3.0),
    n_trials=5,                # seeds 1..5 (or pass seeds=[...])
    test_data=None,            # None evaluates on the training data
)
best.vfind_best_seed           # winning seed
best.vfind_best_qerror         # its mean per-sample quantization error
best.vfind_qerrors             # {seed: qerror} for every trial
```

Each trial's quantization error is logged at INFO level.

### Training snapshots

`train_pak` can save the codebook every N steps, producing numbered `.cod` files for convergence analysis or animation:

```python
som.train_pak(
    rlen=10000, alpha=0.05, radius=10.0,
    snapshot_interval=1000,
    snapshot_path="run/map.cod",   # -> run/map_01000.cod, run/map_02000.cod, ...
)
```

## Per-sample BMU and quantization error (`visual`)

For a trained map, `compute_visual` returns each sample's best matching unit and its quantization error; `write_vis` saves the same data as a `.vis` file:

```python
res = som.compute_visual()     # VisualResult: coords (n,2), qerrors (n,), labels
som.write_vis("result.vis")    # .vis output
```

Each `.vis` line is `x y qerror [label]`, where the label is the BMU unit's calibrated label. A fully masked sample (no valid components) is written as `-1 -1 -1`.

## Data Files (`.dat`)

### Loading `.dat` files

`load_som_pak_data` reads `.dat` files (the SOM_PAK data format) and returns a `DatasetWrapper` (compatible with `create_trainer`). Each row holds space-separated feature values; a trailing label token is optional.

```python
data = somkit.load_som_pak_data("animal.dat")
```

### Per-sample metadata with `SOMData`

To use features such as missing-component masks (`x` fields in `.dat`), per-sample learning weights, or fixed-point BMUs, pass a `SOMData` container to `create_trainer`.

```python
import numpy as np
from somkit.data_loader import SOMData

X = np.random.rand(100, 8)

# Mask specific components for specific samples (True = ignore)
mask = np.zeros((100, 8), dtype=bool)
mask[5, 2] = True   # sample 5, component 2 is missing

# Per-sample learning weight
sample_weights = np.ones(100)
sample_weights[10] = 2.0  # sample 10 counts double

sdata = SOMData(data=X, mask=mask, weights=sample_weights)
som = somkit.create_trainer(data=sdata, size=(10, 10), learning_rate=0.05)
som.initialize_weights_randomly()
som.train_pak(rlen=5000, alpha=0.05, radius=5.0, neighborhood="bubble", seed=1)
```

`SOMData` also accepts `fixed` / `fixed_valid` to force the BMU for specific samples.

## Codebook I/O

### Saving a `.cod` file

After training, save the codebook. Labels from `calibrate_labels` can be embedded in the file, equivalent to `vcal` output.

```python
labels = som.calibrate_labels(numlabs=1)  # majority label per unit
som.save_cod("result.cod", neigh="bubble", labels=labels)
```

### Loading a `.cod` file

```python
som = somkit.SOMTrainer.load_cod("result.cod")
# attach data before further training or evaluation
som.set_data(data)
```

`load_cod` / `save_cod` are also available as standalone functions:

```python
header, weights = somkit.read_cod("result.cod")
somkit.write_cod("copy.cod", weights, topol="hexa", neigh="bubble")
```

## Weight Initialization

### Random initialization

`initialize_weights_randomly` initializes each component from the per-component `[min, max]` range of the training data.

```python
som.initialize_weights_randomly()        # default
som.initialize_weights_randomly(rng=somkit.functions.OrandRNG(seed=42))  # explicit seed
```

### Linear (PCA-based) initialization

Initializes weights in the subspace spanned by the two largest principal components of the data, arranged in a linear grid.

```python
som.initialize_weights_linearly()
```

## Data Normalization

Normalization is opt-in:

```python
som.normalize_data(method="standard")  # Z-score (mean=0, std=1)
som.normalize_data(method="minmax")    # scale to [0, 1]
som.normalize_data(method="variance")  # divide by std, preserve mean
```

## Topology

Both hexagonal and rectangular topologies are supported. Topology affects BMU search distance, neighborhood function, and visualization grid shape.

```python
som = somkit.create_trainer(data=data, size=(10, 10), learning_rate=0.05, topology="hexagonal")
som = somkit.create_trainer(data=data, size=(10, 10), learning_rate=0.05, topology="rectangular")
```

## Visualization

All visualization methods are on `SOMVisualizer`. Every map shares the same grid orientation (row 0 at top), so the same unit appears at the same position across all plot types.

### U-Matrix

```python
visualizer = somkit.SOMVisualizer(som)
visualizer.plot_umatrix()                    # umat style (default)
visualizer.plot_umatrix(
    show_labels=True,    # vcal majority labels per unit
    numlabs=1,           # max labels per unit (0 = all)
    show_nodes=True,     # dot on units with no label
    file_name="umatrix.png",
    show=False,
)
```

The U-Matrix uses the `(2*x-1, 2*y-1)` interpolated grid: cells between units show the inter-unit distance, and the darker walls mark cluster boundaries.

### Component Planes, Hit Map, Class Distribution

```python
visualizer.plot_component_planes(file_name="planes.png", show=False)
visualizer.plot_hit_map(file_name="hitmap.png", show=False)
visualizer.plot_class_distribution(file_name="classes.png", show=False)
```

### Sammon's Mapping Projection

Projects high-dimensional data and SOM nodes to 2D using Sammon's mapping. Preserves inter-point distances, providing a topology-independent view of the data structure.

```python
visualizer.plot_sammon_projection(
    show_nodes=True,
    show_data_points=True,
    show_connections=True,
    connection_style="spring",   # "spring" (thickness ~ distance) or "line"
    colormap="tab10",            # auto-switches to tab20 for >10 classes
    max_iter=500,
    learning_rate=0.2,
    random_state=42,
    file_name="sammon.png",
    show=False,
)
```

## Evaluation

```python
evaluator = somkit.SOMEvaluator(som)
print("WCSS:", evaluator.calculate_wcss())
print("Silhouette Score:", evaluator.calculate_silhouette_score())
print("Topological Error:", evaluator.calculate_topological_error())
```

## Model Persistence

```python
som.save_model("my_som.h5")   # somkit native HDF5 checkpoint
```

For `.cod` output, use `save_cod` instead (see [Codebook I/O](#codebook-io)).

## Examples

All examples are in `examples/`. Run from the `examples/` directory.

### Animal dataset (two-phase training)

```bash
cd examples
python animal.py
```

### Iris dataset

```bash
python iris.py
```

### Breast cancer, digits, wine datasets

```bash
python breast_cancer.py
python digits.py
python wine.py
```

## Directory Structure

```
somkit/
  somkit/
    trainer/        # SOMTrainer, create_trainer, train_pak, train_two_phase
    functions/      # neighborhood, decay, rng (OrandRNG), learning, initialization
    data_loader/    # SOMData, SOMPakDataLoader, load_som_pak_data
    io/             # cod.py — read_cod, write_cod
    visualizer/     # SOMVisualizer, compute_umatrix_pak
    evaluator/      # SOMEvaluator
    topology/       # HexagonalTopology, RectangularTopology
    preprocessing/  # normalization
    projection/     # Sammon's mapping
    decomposition/  # PCA
  examples/
  test/
```
