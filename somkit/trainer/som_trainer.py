from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from somkit.data_loader import Bunch, DatasetWrapper, SOMData
from somkit.decomposition import PCA
from somkit.exceptions import CodFormatError, SomkitError, SOMDataError
from somkit.functions import gaussian, get_alpha_scheduler, linear_radius
from somkit.functions.initialization import random_init
from somkit.functions.labels import calibrate_labels as _calibrate_labels
from somkit.functions.learning import (
    find_bmu_pak,
    find_bmu_pak_batch,
    presentation_order,
    som_step,
    weighted_alpha,
)
from somkit.functions.neighborhood import get_pak_neighborhood
from somkit.functions.rng import OrandRNG
from somkit.io.cod import read_cod, write_cod
from somkit.io.vis import VisualResult
from somkit.io.vis import write_vis as _write_vis_file
from somkit.preprocessing import fit_transform
from somkit.topology import HexagonalTopology


# Mapping between somkit topology names (Topology.get_name) and SOM_PAK .cod
# topology strings.
_TOPOLOGY_TO_COD = {"hexagonal": "hexa", "rectangular": "rect"}
_COD_TO_TOPOLOGY = {v: k for k, v in _TOPOLOGY_TO_COD.items()}

logger = logging.getLogger(__name__)


__n_radius__ = 1.0
__dynamic_radius__ = True
__checkpoint_interval__ = 1


class SOMTrainer:
    def __init__(
        self,
        data: Bunch | DatasetWrapper | np.ndarray,
        size: Tuple[int, int],
        input_dim: int,
        learning_rate: float,
        n_func: Callable = gaussian,
        initial_radius: float = __n_radius__,
        dynamic_radius: bool = True,
        checkpoint_interval: int = __checkpoint_interval__,
        random_seed: int | None = None,
        rng: np.random.Generator | None = None,
        tau: float | None = None,
        topology: str = "hexagonal",
    ) -> None:
        """
        Initialize the Self-Organizing Map (SOM) with the given parameters.

        :param size: The number of nodes in the x, y dimension.
        :param input_dim: The dimensionality of the input data.
        :param epochs: The number of epochs for training.
        :param learning_rate: The initial learning rate for weight updates.
        :param n_func: The neighborhood function to use for updating weights.
        :param initial_radius: The initial radius of the neighborhood function.
        :param dynamic_radius: Whether to use dynamic radius decay during training.
        :param checkpoint_interval: The interval at which to save checkpoints during training.
        :param random_seed: The random seed to use for reproducible results.
        :param rng: The random number generator to use.
        :param tau: Time constant for exponential decay. If None, defaults to n_epochs.
        :param topology: The topology to use ('hexagonal' or 'rectangular').
        """
        self._org_data = data
        self._somdata = self._to_somdata(data)
        self.data = self._somdata.data
        self.target = getattr(data, "target", np.array([]))
        self.target_names = getattr(data, "target_names", np.array([]))
        self.x_size = size[0]
        self.y_size = size[1]
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.weights = None
        # Set topology based on parameter
        if topology == "rectangular":
            from somkit.topology import RectangularTopology
            self.topology = RectangularTopology()
        else:
            self.topology = HexagonalTopology()
        self.n_func = n_func
        self.initial_radius = initial_radius
        self.n_radius = initial_radius
        self.dynamic_radius = dynamic_radius
        self.tau = tau

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir: str = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.random_seed = random_seed
        self.rng = rng
        if self.rng is None:
            if self.random_seed is not None:
                self.rng = np.random.RandomState(self.random_seed)
            else:
                self.rng = np.random.RandomState()

    # ====================
    # training
    # ====================

    def initialize_weights_randomly(self, rng: OrandRNG | None = None) -> None:
        """Initialize weights the SOM_PAK ``randinit_codes`` way (SPEC-0001 FR-7).

        Each component is drawn uniformly from the data's per-component
        ``[min, max]`` range using the ported ``orand`` generator, in SOM_PAK's
        scan order. This is a behavioral change from the previous unconditional
        ``[0, 1)`` initialization (kept as :meth:`initialize_weights_uniform`).

        Args:
            rng: A seeded :class:`OrandRNG`. If omitted, one is created from
                ``random_seed`` (or seed ``1`` when ``random_seed`` is ``None``;
                note this differs from numpy's non-deterministic ``None``).
        """
        if self.data is None:
            raise ValueError(
                "Data must be set using 'set_data' before initializing weights."
            )
        if rng is None:
            seed = self.random_seed if self.random_seed is not None else 1
            rng = OrandRNG(seed)
        self.weights = random_init(self.data, self.x_size, self.y_size, rng)

    def initialize_weights_uniform(self) -> None:
        """Initialize weights uniformly in ``[0, 1)`` (legacy, non-SOM_PAK).

        This is the historical somkit behavior, preserved as an opt-in. It uses
        the numpy RNG and does not match SOM_PAK; prefer
        :meth:`initialize_weights_randomly` for SOM_PAK conformance.
        """
        self.weights = self.rng.rand(self.x_size, self.y_size, self.input_dim)

    def initialize_weights_with_pca(self) -> None:
        """
        Initialize the weight matrix using the first two principal components
        of the input data. This method can provide a better starting point for
        the SOM training, potentially leading to faster convergence and a more
        accurate representation of the input data.

        Note: This method should be called after setting the input data using the `set_data` method.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before initializing weights with PCA."

        # Calculate the first two principal components of the data using PCA
        pca = PCA(n_components=2)
        pca.fit(self.data)

        # Initialize the weight matrix using the first two principal components
        two_principal_components = pca.components_[:2]
        ranges = [np.linspace(0, 1, num) for num in (self.x_size, self.y_size)]
        grid = np.meshgrid(*ranges, indexing="ij")
        grid = np.stack(grid, axis=-1)

        # Initialize the weight matrix using the first two principal components
        self.weights = np.tensordot(grid, two_principal_components, axes=1) + pca.mean_

    def initialize_weights_linearly(self) -> None:
        """
        Initialize the weight matrix using linear initialization (ordered initialization).

        This is the standard initialization method used in SOM_PAK. The weight vectors
        are initialized to lie in a linear subspace spanned by the two largest principal
        components of the data, arranged in an ordered grid.

        This method provides:
        - Fast and stable convergence
        - Good reproducibility
        - Better initial topology preservation than random initialization

        The weights are initialized as:
        w[i,j] = mean + alpha * PC1 + beta * PC2
        where alpha and beta vary linearly across the grid.

        Note: This method should be called after setting the input data using the `set_data` method.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before initializing weights linearly."

        # Calculate the first two principal components of the data using PCA
        pca = PCA(n_components=2)
        pca.fit(self.data)

        # Get the two largest principal components
        pc1 = pca.components_[0]  # First principal component
        pc2 = pca.components_[1]  # Second principal component
        mean = pca.mean_

        # Calculate the standard deviations along the principal components
        # This helps determine the spread of the initial grid
        data_centered = self.data - mean
        proj1 = np.dot(data_centered, pc1)
        proj2 = np.dot(data_centered, pc2)
        std1 = np.std(proj1)
        std2 = np.std(proj2)

        # Create linearly spaced coefficients for the grid
        # Range from -2*std to +2*std to cover most of the data
        alpha_range = np.linspace(-2 * std1, 2 * std1, self.x_size)
        beta_range = np.linspace(-2 * std2, 2 * std2, self.y_size)

        # Initialize weights
        self.weights = np.zeros((self.x_size, self.y_size, self.input_dim))

        for i in range(self.x_size):
            for j in range(self.y_size):
                # Linear combination: w = mean + alpha*PC1 + beta*PC2
                self.weights[i, j] = mean + alpha_range[i] * pc1 + beta_range[j] * pc2

    def shuffle_data(self):
        """
        Shuffle the input data and target labels (if available) in unison.

        Raises:
            SOMDataError: If a mask is set. Per-sample metadata (mask) would lose
                its row correspondence with the shuffled data. ``train_pak``
                controls presentation order internally (``presentation_order``),
                so shuffling beforehand is unnecessary when using SOMData/mask.
        """
        if self._somdata.mask is not None:
            raise SOMDataError(
                "shuffle_data is not supported when a mask is set "
                "(it would break the data/mask row correspondence); train_pak "
                "shuffles the presentation order internally."
            )
        indices = np.arange(len(self.data))
        self.rng.shuffle(indices)
        self.data = self.data[indices]
        # if self.target is not None:
        if len(self.target) > 0:
            self.target = self.target[indices]

    def standardize_data(self):
        """
        Standardize the input data to have a mean of 0 and a standard deviation of 1.

        This is equivalent to normalize_data(method='standard').
        """
        self.data = fit_transform(self.data, method='standard')

    def normalize_data(self, method='standard'):
        """
        Normalize the input data using the specified method.

        :param method: Normalization method. Options:
            - 'standard': Z-score normalization (mean=0, std=1)
            - 'minmax': Min-Max normalization to [0, 1]
            - 'variance': Variance normalization (divide by std only)
        """
        self.data = fit_transform(self.data, method=method)

    def train(
        self, n_epochs: int, batch_size: int = 1, shuffle_each_epoch: bool = True
    ) -> None:
        """
        Train the SOM using sequential (online) learning.

        :param n_epochs: The number of epochs for training.
        :param batch_size: The batch size for training. If None, online learning will be used.
        :param shuffle_each_epoch: Whether to shuffle the input data before each epoch.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before training."

        if self.weights is None:
            self.initialize_weights_randomly()

        # Set tau to n_epochs if not specified
        tau = self.tau if self.tau is not None else n_epochs

        for epoch in tqdm(range(n_epochs)):
            if shuffle_each_epoch:
                self.shuffle_data()

            # Update learning rate and radius using exponential decay
            self.learning_rate = self._calculate_learning_rate(epoch, tau)
            if self.dynamic_radius:
                self.n_radius = self._calculate_radius(epoch, tau)

            batch_indices = np.arange(0, self.data.shape[0], batch_size)
            for batch_index in batch_indices:
                batch = self.data[batch_index : batch_index + batch_size]
                # bmu = [self._find_bmu(sample)[0] for sample in batch]
                bmu_indices = [self._find_bmu(sample)[1] for sample in batch]
                self._update_weights_batch(batch, bmu_indices, self.get_radius())

            # Save checkpoint at specified intervals
            if epoch % self.checkpoint_interval == 0:
                checkpoint_path = self._get_checkpoint_file_path(epoch)
                self._save_checkpoint(checkpoint_path)

        # self._compute_performance_metrics(self.data)

    def train_batch(
        self, n_epochs: int, shuffle_each_epoch: bool = True
    ) -> None:
        """
        Train the SOM using batch learning algorithm.

        In batch learning, all data samples are processed before updating weights.
        This provides more stable convergence and doesn't require a learning rate parameter.

        The batch SOM algorithm:
        1. Find BMU for each data sample
        2. For each node, collect all samples that have it as BMU
        3. Update node weight as weighted average of collected samples
           (weighted by neighborhood function)

        :param n_epochs: The number of epochs for training.
        :param shuffle_each_epoch: Whether to shuffle the input data before each epoch.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before training."

        if self.weights is None:
            self.initialize_weights_randomly()

        # Set tau to n_epochs if not specified
        tau = self.tau if self.tau is not None else n_epochs

        for epoch in tqdm(range(n_epochs)):
            if shuffle_each_epoch:
                self.shuffle_data()

            # Update radius using exponential decay
            if self.dynamic_radius:
                self.n_radius = self._calculate_radius(epoch, tau)

            # Batch update
            self._batch_update(self.get_radius())

            # Save checkpoint at specified intervals
            if epoch % self.checkpoint_interval == 0:
                checkpoint_path = self._get_checkpoint_file_path(epoch)
                self._save_checkpoint(checkpoint_path)

    def train_pak(
        self,
        rlen: int,
        alpha: float,
        radius: float,
        *,
        alpha_type: str = "linear",
        neighborhood: str = "bubble",
        random_order: bool = True,
        seed: int | None = None,
        rng: OrandRNG | None = None,
        snapshot_interval: int | None = None,
        snapshot_path: str | None = None,
        progress: bool = True,
    ) -> None:
        """Train the SOM the SOM_PAK ``som_training`` way (SPEC-0001 FR-2).

        This is the SOM_PAK-conformant sequential learning path: ``rlen`` total
        steps, one sample presented per step (cycling the data), with the
        learning rate and neighborhood radius decaying **per step** (linear
        radius down to 1, linear/inverse-t alpha). It differs from
        :meth:`train`/:meth:`train_batch`, which are epoch-based, non-conformant
        extensions kept for backward compatibility.

        Args:
            rlen: Total number of training steps (SOM_PAK ``rlen``).
            alpha: Initial learning rate.
            radius: Initial neighborhood radius (decays linearly toward 1).
            alpha_type: ``"linear"`` (default) or ``"inverse_t"`` (FR-1).
            neighborhood: ``"bubble"`` (default, SOM_PAK) or ``"gaussian"`` (FR-5).
            random_order: Whether to present samples in a shuffled (OrandRNG)
                order (SOM_PAK ``-rand``); otherwise sequential cycling.
            seed: Seed for the presentation-order RNG when ``rng`` is omitted.
                Falls back to ``random_seed`` then ``1``.
            rng: An explicit :class:`OrandRNG` for the presentation order
                (overrides ``seed``). Initialization uses a separate RNG (see
                :meth:`initialize_weights_randomly`), matching SOM_PAK's separate
                ``randinit`` / ``vsom`` random streams.
            snapshot_interval: Save the codebook every this many steps
                (SPEC-0004 FR-3; SOM_PAK ``-snapinterval``). Saved at steps
                where ``le % interval == 0 and le > 0`` (som_rout.c:652), so
                neither step 0 nor the final step produces a snapshot.
                ``None`` (or ``0``) disables snapshots.
            snapshot_path: Base ``.cod`` path for snapshots; step files are
                written next to it as ``{stem}_{step:05d}.cod``. Required
                together with ``snapshot_interval``.
            progress: Show the tqdm progress bar. :meth:`vfind` disables it to
                avoid one bar per trial.

        Raises:
            ValueError: If data is not set, ``rlen <= 0``, or
                ``alpha_type``/``neighborhood`` is unknown.
            SomkitError: If only one of ``snapshot_interval``/``snapshot_path``
                is given, or the snapshot directory does not exist.
        """
        if self.data is None:
            raise ValueError(
                "Data must be set using 'set_data' before training."
            )
        if rlen <= 0:
            raise ValueError(f"rlen must be a positive integer, got {rlen!r}.")
        if self.weights is None:
            self.initialize_weights_randomly()

        alpha_fn = get_alpha_scheduler(alpha_type)
        neighborhood_fn = get_pak_neighborhood(neighborhood)

        if rng is None:
            order_seed = (
                seed
                if seed is not None
                else (self.random_seed if self.random_seed is not None else 1)
            )
            rng = OrandRNG(order_seed)

        snapshot_template = self._resolve_snapshot_template(
            snapshot_interval, snapshot_path
        )

        n_samples = self.data.shape[0]
        order = presentation_order(n_samples, rlen, rng, random_order)
        # Per-sample metadata (None -> SPEC-0001 fast path, bit-identical).
        mask = self._somdata.mask
        sample_weights = self._somdata.weights
        fixed = self._somdata.fixed
        fixed_valid = self._somdata.fixed_valid
        self._validate_fixed(fixed, fixed_valid)

        for le in tqdm(range(rlen), disable=not progress):
            idx = order[le]
            sample = self.data[idx]
            m = mask[idx] if mask is not None else None
            trad = linear_radius(le, rlen, radius)
            talp = alpha_fn(le, rlen, alpha)
            if sample_weights is not None:  # FR-2 (som_rout.c:624-626)
                talp = weighted_alpha(talp, sample_weights[idx])
            if fixed is not None and (fixed_valid is None or fixed_valid[idx]):
                bmu = (int(fixed[idx, 0]), int(fixed[idx, 1]))  # FR-3, winner skipped
            elif m is not None and m.all():
                bmu = None  # fully-masked sample: no winner (som_rout.c:637-642)
            else:
                bmu = find_bmu_pak(self.weights, sample, mask=m)
            if bmu is not None:
                som_step(
                    self.weights, sample, bmu, trad, talp, neighborhood_fn,
                    self.topology, mask=m,
                )
            # Snapshot after this step's update, exactly as SOM_PAK does: the
            # save sits after adapt at the skip_teach label — which a skipped
            # (fully-masked) step also reaches — with the condition
            # (le % interval == 0) && (le > 0), som_rout.c:650-660. Neither
            # step 0 nor the final codebook produces a snapshot.
            if snapshot_template and le > 0 and le % snapshot_interval == 0:
                write_cod(
                    snapshot_template.format(step=le),
                    self.weights,
                    topol=_TOPOLOGY_TO_COD[self.topology.get_name()],
                    neigh=neighborhood,
                    # Two comment lines, as SOM_PAK save_snapshot writes
                    # (lvq_pak.c:529-531).
                    comments=["SNAPSHOT FILE", f"iterations: {le} ({rlen} total)"],
                )

        # Expose the final schedule values for compatibility with getters.
        self.n_radius = linear_radius(rlen - 1, rlen, radius)
        self.learning_rate = alpha_fn(rlen - 1, rlen, alpha)

    @staticmethod
    def _resolve_snapshot_template(
        snapshot_interval: int | None, snapshot_path: str | None
    ) -> str | None:
        """Validate the snapshot arguments and build the filename template.

        Args:
            snapshot_interval: Steps between snapshots (``None`` = disabled).
            snapshot_path: Base ``.cod`` path; snapshots become
                ``{stem}_{step:05d}.cod`` in the same directory.

        Returns:
            A ``str.format`` template with a ``{step}`` field, or ``None``
            when snapshots are disabled.

        Raises:
            SomkitError: If only one of the two arguments is given, or the
                target directory does not exist.
        """
        # interval None/0 both mean "disabled" (SPEC-0004 FR-3).
        if not snapshot_interval and snapshot_path is None:
            return None
        if not snapshot_interval or snapshot_path is None:
            raise SomkitError(
                "snapshot_interval and snapshot_path must be given together."
            )
        directory, filename = os.path.split(snapshot_path)
        if directory and not os.path.isdir(directory):
            raise SomkitError(
                f"snapshot directory does not exist: {directory!r}."
            )
        stem = filename[:-4] if filename.endswith(".cod") else filename
        return os.path.join(directory, stem + "_{step:05d}.cod")

    def _validate_fixed(
        self, fixed: np.ndarray | None, fixed_valid: np.ndarray | None
    ) -> None:
        """Validate fixed-point coordinates are within the map (SPEC-0002 FR-3).

        Range check is done here (not in SOMData) because it depends on the
        grid size, which SOMData does not know.

        Raises:
            SOMDataError: If any active fixed coordinate is out of range.
        """
        if fixed is None:
            return
        active = fixed if fixed_valid is None else fixed[fixed_valid]
        if active.size == 0:
            return
        x_min, x_max = int(active[:, 0].min()), int(active[:, 0].max())
        y_min, y_max = int(active[:, 1].min()), int(active[:, 1].max())
        if x_min < 0 or x_max >= self.x_size or y_min < 0 or y_max >= self.y_size:
            raise SOMDataError(
                f"fixed coordinates out of range for map ({self.x_size}, "
                f"{self.y_size}): x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]."
            )

    def train_two_phase(self, phase1: dict, phase2: dict) -> None:
        """Run SOM_PAK's two-phase training (coarse ordering -> fine tuning).

        This is high-level sugar over :meth:`train_pak`: ``phase1`` initializes
        and coarsely orders the map (large radius, higher alpha, shorter rlen),
        then ``phase2`` continues from those weights to fine-tune (small radius,
        lower alpha, longer rlen). It simply calls :meth:`train_pak` twice with
        no duplicated logic, mirroring SOM_PAK's ``command.sh`` Step 2 (two
        consecutive ``vsom`` runs).

        Args:
            phase1: Keyword arguments for the coarse :meth:`train_pak` call
                (must include ``rlen``/``alpha``/``radius``). Example:
                ``dict(rlen=1000, alpha=0.05, radius=10.0)``.
            phase2: Keyword arguments for the fine :meth:`train_pak` call.
                Example: ``dict(rlen=10000, alpha=0.02, radius=3.0)``.

        Raises:
            ValueError: Propagated from :meth:`train_pak` (e.g. ``rlen <= 0``,
                unknown ``alpha_type``/``neighborhood``).
            TypeError: Propagated from Python's argument binding when a required
                argument (``rlen``, ``alpha``, or ``radius``) is absent from a
                phase dict.
        """
        self.train_pak(**phase1)
        self.train_pak(**phase2)

    @classmethod
    def vfind(
        cls,
        data,
        size: Tuple[int, int],
        *,
        phase1: dict,
        phase2: Optional[dict] = None,
        n_trials: int,
        test_data: Union[np.ndarray, SOMData, None] = None,
        seeds: Optional[List[int]] = None,
        learning_rate: float = 0.05,
        topology: str = "hexagonal",
        neighborhood: str = "bubble",
        alpha_type: str = "linear",
    ) -> "SOMTrainer":
        """Train ``n_trials`` maps with different seeds and return the best one.

        The somkit port of SOM_PAK ``vfind`` (SPEC-0004 FR-1): for each seed, a
        fresh map is randomly initialized and trained (``phase1`` then
        ``phase2`` on the same codebook), and the map with the smallest mean
        per-sample quantization error on ``test_data`` wins. As in SOM_PAK, a
        single random stream per trial drives both the initialization and the
        sample presentation order, and ties keep the earlier trial (strict
        ``<`` comparison, vfind.c:290).

        Args:
            data: Training data (array, sklearn Bunch, or :class:`SOMData`).
            size: Map size ``(x_size, y_size)``.
            phase1: Keyword arguments for the coarse :meth:`train_pak` call,
                e.g. ``dict(rlen=1000, alpha=0.05, radius=10.0)``. Must not
                contain ``seed``/``rng`` (the trial seed controls the RNG).
            phase2: Optional fine-tuning :meth:`train_pak` arguments; ``None``
                trains in a single phase.
            n_trials: Number of trials (required).
            test_data: Data the quantization error is evaluated on; ``None``
                falls back to the training data.
            seeds: Explicit seed list; ``None`` uses ``1..n_trials``
                (SOM_PAK's ``init_random`` trial numbering).
            learning_rate: Initial learning rate handed to the trainer
                constructor (the effective rate comes from the phase dicts).
            topology: ``"hexagonal"`` or ``"rectangular"``.
            neighborhood: ``"bubble"`` or ``"gaussian"``.
            alpha_type: ``"linear"`` or ``"inverse_t"``.

        Returns:
            The best trial's :class:`SOMTrainer`, with extra attributes
            attached: ``vfind_best_seed`` (int), ``vfind_best_qerror``
            (float, mean per-sample qerror) and ``vfind_qerrors``
            (dict[int, float], seed -> mean qerror for every trial).

        Raises:
            SomkitError: If ``n_trials < 1``, ``len(seeds) != n_trials``, or a
                phase dict contains ``seed``/``rng``.

        References:
            vfind.c:245-304 (trial loop), vfind.c:290 (strict ``<``),
            som_rout.c:680-733 (find_qerror; sum vs mean is argmin-equivalent
            for a fixed test set, see SPEC-0004 FR-1).
        """
        if n_trials < 1:
            raise SomkitError(f"n_trials must be >= 1, got {n_trials}.")
        if seeds is None:
            seeds = list(range(1, n_trials + 1))
        elif len(seeds) != n_trials:
            raise SomkitError(
                f"len(seeds)={len(seeds)} contradicts n_trials={n_trials}; "
                "drop n_trials mismatches or fix the seed list."
            )
        for name, phase in (("phase1", phase1), ("phase2", phase2)):
            if phase and ("seed" in phase or "rng" in phase):
                raise SomkitError(
                    f"{name} must not contain 'seed'/'rng'; the trial seed "
                    "drives a single random stream per trial (vfind.c)."
                )
        if test_data is None:
            logger.info("vfind: test_data not given; evaluating on training data")

        best: Optional["SOMTrainer"] = None
        best_qerror = float("inf")
        best_seed: Optional[int] = None
        qerrors: Dict[int, float] = {}
        for seed in seeds:
            trainer = create_trainer(
                data=data,
                size=size,
                learning_rate=learning_rate,
                random_seed=seed,
                topology=topology,
            )
            # One stream per trial for both init and presentation order,
            # matching SOM_PAK's single init_random(not) stream.
            rng = OrandRNG(seed)
            trainer.initialize_weights_randomly(rng=rng)
            trainer.train_pak(
                **phase1, neighborhood=neighborhood, alpha_type=alpha_type,
                rng=rng, progress=False,
            )
            if phase2 is not None:
                trainer.train_pak(
                    **phase2, neighborhood=neighborhood, alpha_type=alpha_type,
                    rng=rng, progress=False,
                )

            result = trainer.compute_visual(test_data)
            valid = result.qerrors >= 0  # exclude empty (fully masked) samples
            if not valid.any():
                raise SomkitError(
                    f"vfind trial seed={seed}: every test sample is empty "
                    "(fully masked); cannot compute the quantization error."
                )
            qerror = float(result.qerrors[valid].mean())
            qerrors[seed] = qerror
            logger.info("vfind trial seed=%d qerror=%g", seed, qerror)
            if qerror < best_qerror:  # strict <: ties keep the earlier trial
                best, best_qerror, best_seed = trainer, qerror, seed

        logger.info("vfind best seed=%d qerror=%g", best_seed, best_qerror)
        best.vfind_best_seed = best_seed
        best.vfind_best_qerror = best_qerror
        best.vfind_qerrors = qerrors
        return best

    def _find_bmu(self, sample: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Find the Best Matching Unit (BMU) in the SOM for the given input sample.

        :param sample: The input sample for which to find the BMU.
        :return bmu: The weights of the BMU.
        :return bmu_idx: The indices of the BMU in the SOM grid.
        """
        # Calculate Euclidean distances between sample and all weight vectors
        distances = np.linalg.norm(self.weights - sample, axis=2)

        # Find the node with minimum distance (BMU)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.x_size, self.y_size))
        bmu = self.weights[bmu_idx]

        return bmu, bmu_idx

    def _update_weights_batch(
        self,
        batch: np.ndarray,
        bmu_indices: List[Tuple[int, int]],
        current_radius: float,
    ) -> None:
        """
        Update the weights of the SOM using the given batch of input samples (sequential mode).

        Weight update formula: w_i(t+1) = w_i(t) + η(t) · h_ij(t) · (x - w_i(t))

        :param batch: A batch of input samples.
        :param bmu_indices: The BMU indices in the SOM grid for each input sample.
        :param current_radius: The current radius of the neighborhood function.
        """
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        grid = np.concatenate((x, y), axis=1)

        for sample, bmu_idx in zip(batch, bmu_indices):
            # Use topology-specific distance function
            distance = self.topology.topology_function(
                grid[:, 0], grid[:, 1], bmu_idx[0], bmu_idx[1]
            )
            influence = self.n_func(current_radius, distance, self.get_radius())

            mask = distance <= current_radius
            influence = influence[mask].reshape(-1, 1)

            affected_nodes = grid[mask]

            affected_weights = self.weights[
                affected_nodes[:, 0], affected_nodes[:, 1], :
            ]
            # Use the current learning rate (already decayed) without dividing by batch size
            # Formula: w_i(t+1) = w_i(t) + η(t) · h_ij(t) · (x - w_i(t))
            new_weights = affected_weights + self.learning_rate * influence * (sample - affected_weights)
            self.weights[affected_nodes[:, 0], affected_nodes[:, 1], :] = new_weights

    def _batch_update(self, current_radius: float) -> None:
        """
        Perform batch SOM weight update (vectorized version).

        In batch SOM, weights are updated using the formula:
        w_i(t+1) = Σ_j h_ij(t) · x_j / Σ_j h_ij(t)

        where:
        - w_i is the weight vector of node i
        - h_ij is the neighborhood function between node i and BMU of sample j
        - x_j is data sample j

        This is equivalent to a weighted average where each node's new weight
        is the weighted mean of all data samples, weighted by the neighborhood function.

        :param current_radius: The current radius of the neighborhood function.
        """
        # Find BMUs for all data samples
        bmus = self.get_bmus(self.data)  # List of (i, j) tuples
        bmu_array = np.array(bmus)  # Shape: (n_samples, 2)
        bmu_x = bmu_array[:, 0]  # Shape: (n_samples,)
        bmu_y = bmu_array[:, 1]  # Shape: (n_samples,)

        # Create grid of all node coordinates
        # node_x, node_y: Shape (x_size, y_size)
        node_x, node_y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size), indexing='ij')

        # Reshape for broadcasting: (x_size, y_size, 1)
        node_x = node_x[:, :, np.newaxis]
        node_y = node_y[:, :, np.newaxis]

        # BMU coordinates for broadcasting: (1, 1, n_samples)
        bmu_x = bmu_x[np.newaxis, np.newaxis, :]
        bmu_y = bmu_y[np.newaxis, np.newaxis, :]

        # Calculate distances from all nodes to all BMUs
        # distances: Shape (x_size, y_size, n_samples)
        distances = self.topology.topology_function(
            node_x, node_y, bmu_x, bmu_y
        )

        # Calculate neighborhood influence for all node-BMU pairs
        # influences: Shape (x_size, y_size, n_samples)
        influences = self.n_func(current_radius, distances, self.get_radius())

        # Calculate weighted sum: Σ h_ij * x_j
        # influences: (x_size, y_size, n_samples, 1)
        # data: (1, 1, n_samples, n_features)
        # numerator: (x_size, y_size, n_features)
        influences_expanded = influences[:, :, :, np.newaxis]  # (x_size, y_size, n_samples, 1)
        data_expanded = self.data[np.newaxis, np.newaxis, :, :]  # (1, 1, n_samples, n_features)
        numerator = np.sum(influences_expanded * data_expanded, axis=2)

        # Calculate sum of influences: Σ h_ij
        # denominator: (x_size, y_size, 1)
        denominator = np.sum(influences, axis=2, keepdims=True)

        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)

        # Update weights as weighted average
        self.weights = numerator / denominator

    def _calculate_learning_rate(self, t: int, tau: float) -> float:
        """
        Calculate the learning rate at time t using exponential decay.

        η(t) = η_0 · exp(-t / τ)

        :param t: Current iteration/epoch.
        :param tau: Time constant for decay.
        :return: The learning rate at time t.
        """
        return self.initial_learning_rate * np.exp(-t / tau)

    def _calculate_radius(self, t: int, tau: float) -> float:
        """
        Calculate the neighborhood radius at time t using exponential decay.

        σ(t) = σ_0 · exp(-t / τ)

        :param t: Current iteration/epoch.
        :param tau: Time constant for decay.
        :return: The neighborhood radius at time t.
        """
        return self.initial_radius * np.exp(-t / tau)

    def _decay_function(self, n_epochs: int, epoch: int) -> float:
        """
        Calculate the decay function value for the given epoch.
        (Deprecated: Use _calculate_radius instead)

        :param n_epochs: The total number of epochs for training.
        :param epoch: The current epoch of training.
        :return: The decay function value for the given epoch.
        """
        return np.exp(-epoch / n_epochs) * max(self.x_size, self.y_size) / 2.0

    def get_bmus(self, data: ndarray) -> List[Tuple[int, int]]:
        """
        Get the Best Matching Units (BMUs) for each input sample in the given data.

        :param data: A 2D numpy array containing the input data.
        :return: A list of BMU indices in the SOM grid for each input sample.
        """
        data_expanded = data[:, np.newaxis, np.newaxis, :]
        distance_map = np.linalg.norm(self.weights - data_expanded, axis=3)
        bmu_indices = np.unravel_index(
            np.argmin(distance_map.reshape(data.shape[0], -1), axis=1),
            (self.x_size, self.y_size),
        )
        return list(zip(bmu_indices[0], bmu_indices[1]))

    def winner(self, data_points: Union[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Find the winning node(s) in the SOM for the given data point(s).

        :param data_points: A single data point or an array of data points with shape (n_points, n_features).
        :return: The coordinates of the winning node(s) as an array of shape (n_points, 2).
                 If only one data point is provided, a 1D array of shape (2,) is returned.
        """
        data_points = np.asarray(data_points)
        if data_points.ndim == 1:
            data_points = data_points[np.newaxis, :]

        # Calculate Euclidean distances between each data point and the nodes in the weights matrix
        distances = np.linalg.norm(
            self.weights - data_points[:, np.newaxis, np.newaxis, :], axis=-1
        )

        # Find the indices of the minimum distance(s) in the flattened distances array
        winner_indices = np.argmin(distances.reshape(data_points.shape[0], -1), axis=1)

        # Convert the indices of the minimum distance(s) to coordinates of the winning node(s)
        winner_coordinates = np.column_stack(
            np.unravel_index(winner_indices, self.weights.shape[:2])
        )

        if winner_coordinates.shape[0] == 1:
            return winner_coordinates[0]
        else:
            return winner_coordinates

    # ====================
    # save
    # ====================

    def _get_checkpoint_file_path(self, epoch: int) -> str:
        """
        Get the file path for the checkpoint file for the given epoch.

        :param epoch: The epoch for which to get the checkpoint file path.
        :return: The file path for the checkpoint file for the given epoch.
        """
        return os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.h5")

    def _save_checkpoint(self, file_path: str) -> None:
        """
        Save the trained SOM model to a file.

        :param file_path: The path to the file where the model will be saved.
        """
        random_state = self.rng.get_state()

        target_names = np.array([])
        if len(self.target_names) > 0:
            if isinstance(self.target_names[0], str):
                target_names = np.array(
                    [name.encode("utf-8") for name in self.target_names]
                )

        with h5py.File(file_path, "w") as f:
            f.attrs["x_size"] = self.x_size
            f.attrs["y_size"] = self.y_size
            f.attrs["input_dim"] = self.input_dim
            f.attrs["n_radius"] = self.n_radius
            f.attrs["learning_rate"] = self.learning_rate
            f.create_dataset("data", data=self.data)
            f.create_dataset("target", data=self.target)
            f.create_dataset("target_names", data=target_names)
            f.create_dataset("weights", data=self.weights)
            grp = f.create_group("random_state")
            grp["0"] = random_state[0]
            grp["1"] = f.create_dataset("stete", data=random_state[1])
            grp["2"] = random_state[2]
            grp["3"] = random_state[3]
            grp["4"] = random_state[4]

    def save_model(self, file_path: str) -> None:
        """
        Save the trained SOM model to a file.

        :param file_path: The path to the file where the model will be saved.
        """
        self._save_checkpoint(file_path)

    def save_cod(
        self, file_path: str, *, neigh: str = "bubble", labels: np.ndarray | None = None
    ) -> None:
        """Save the codebook in SOM_PAK ``.cod`` format (SPEC-0002 FR-4).

        Unlike :meth:`save_model` (somkit's h5 checkpoint), this writes a
        SOM_PAK-compatible codebook that the SOM_PAK C tools can read.

        Args:
            file_path: Output ``.cod`` path.
            neigh: SOM_PAK neighborhood string to record in the header
                (``"bubble"`` default / ``"gaussian"``).
            labels: Optional ``(x_size, y_size)`` object array of per-unit label
                lists (e.g. from :meth:`calibrate_labels`); written after each
                unit's components.

        Raises:
            SomkitError: If the weights have not been initialized/trained.
        """
        if self.weights is None:
            raise SomkitError(
                "weights are not initialized; train or initialize before save_cod."
            )
        topol = _TOPOLOGY_TO_COD[self.topology.get_name()]
        write_cod(file_path, self.weights, topol=topol, neigh=neigh, labels=labels)

    def compute_visual(
        self,
        data: Union[np.ndarray, SOMData, None] = None,
        unit_labels: np.ndarray | None = None,
    ) -> VisualResult:
        """Compute per-sample BMU coordinates and quantization errors (SPEC-0004 FR-2).

        The somkit port of SOM_PAK ``visual``: for every sample, the BMU grid
        coordinates and the L2 distance to the BMU over unmasked components.
        Fully masked samples have no BMU and get ``(-1, -1)`` / ``-1.0``.

        Args:
            data: Samples to evaluate. ``None`` uses the training data
                (including its mask); a :class:`SOMData` brings its own mask;
                a plain array has no mask.
            unit_labels: Optional ``(x_size, y_size)`` object array of per-unit
                label lists (e.g. from :meth:`calibrate_labels`). Each sample
                receives its BMU unit's labels (SOM_PAK ``visual`` copies the
                winner's labels, visual.c:128). ``None`` derives the labels via
                :meth:`calibrate_labels` when targets exist, else no labels.

        Returns:
            A :class:`~somkit.io.vis.VisualResult`.

        Raises:
            SomkitError: If the weights are not initialized or the data
                dimension does not match the codebook.

        References:
            visual.c:47-155 (coords from the winner, ``sqrt(win_info.diff)``
            qerror, empty samples as ``(-1, -1, -1)``).
        """
        if self.weights is None:
            raise SomkitError(
                "weights are not initialized; train or initialize before "
                "compute_visual."
            )
        if data is None:
            if self.data is None or len(self.data) == 0:
                raise SomkitError(
                    "no data attached to the trainer; call set_data() or pass "
                    "data to compute_visual explicitly."
                )
            samples = self.data
            mask = self._somdata.mask
        elif isinstance(data, SOMData):
            samples = data.data
            mask = data.mask
        else:
            samples = np.asarray(data)
            mask = None
        if samples.ndim != 2 or samples.shape[1] != self.weights.shape[2]:
            raise SomkitError(
                f"data of shape {samples.shape} does not match the codebook "
                f"dimension {self.weights.shape[2]}."
            )

        coords = find_bmu_pak_batch(self.weights, samples, mask=mask)
        diff = samples - self.weights[coords[:, 0], coords[:, 1]]
        if mask is not None:
            diff = np.where(mask, 0.0, diff)
        qerrors = np.sqrt(np.einsum("nd,nd->n", diff, diff))

        empty = mask.all(axis=1) if mask is not None else None
        if empty is not None and empty.any():
            coords[empty] = -1
            qerrors[empty] = -1.0

        if unit_labels is None and self._has_targets():
            unit_labels = self.calibrate_labels()
        labels: List[List[str]] | None = None
        if unit_labels is not None:
            labels = [
                []
                if (empty is not None and empty[i])
                else list(unit_labels[coords[i, 0], coords[i, 1]])
                for i in range(len(samples))
            ]
        return VisualResult(coords=coords, qerrors=qerrors, labels=labels)

    def write_vis(
        self,
        file_path: str,
        data: Union[np.ndarray, SOMData, None] = None,
        *,
        neigh: str = "bubble",
        unit_labels: np.ndarray | None = None,
    ) -> None:
        """Write per-sample BMU/qerror data as a SOM_PAK ``.vis`` file (FR-2).

        Args:
            file_path: Output ``.vis`` path.
            data: Samples to evaluate (see :meth:`compute_visual`).
            neigh: SOM_PAK neighborhood string recorded in the header. The
                trainer does not store the neighborhood, so pass the same
                value used as ``train_pak``'s ``neighborhood``.
            unit_labels: Optional per-unit labels (see :meth:`compute_visual`).

        Raises:
            SomkitError: Propagated from :meth:`compute_visual`.
        """
        result = self.compute_visual(data, unit_labels=unit_labels)
        topol = _TOPOLOGY_TO_COD[self.topology.get_name()]
        _write_vis_file(
            file_path,
            result,
            topol=topol,
            xdim=self.x_size,
            ydim=self.y_size,
            neigh=neigh,
        )

    def _has_targets(self) -> bool:
        """Whether usable target labels are attached to the training data."""
        return (
            self.target is not None
            and len(self.target) > 0
            and self.target_names is not None
            and len(self.target_names) > 0
        )

    def calibrate_labels(self, numlabs: int = 1) -> np.ndarray:
        """Label each unit by majority vote of its data samples (SOM_PAK vcal).

        Computes the BMU of every data sample (using the SOM_PAK-conformant
        :func:`find_bmu_pak` tie-break) and assigns each unit the most frequent
        labels of the samples mapped to it. Reuses ``target``/``target_names``.

        Args:
            numlabs: Max labels per unit; ``0`` means all. Default ``1``.

        Returns:
            An ``(x_size, y_size)`` object array of per-unit label lists
            (frequency-descending). Pass it to :meth:`save_cod` as ``labels=``.

        Raises:
            SomkitError: If the weights have not been initialized/trained.
        """
        if self.weights is None:
            raise SomkitError(
                "weights are not initialized; train before calibrate_labels."
            )
        bmus = [tuple(b) for b in find_bmu_pak_batch(self.weights, self.data)]
        # set_data may leave target/target_names as None; treat as "no labels".
        target = self.target if self.target is not None else np.array([])
        target_names = (
            self.target_names if self.target_names is not None else np.array([])
        )
        if len(target) > 0 and len(target_names) > 0:
            sample_labels = [str(target_names[t]) for t in target]
        else:
            sample_labels = [""] * len(self.data)
        return _calibrate_labels(
            bmus, sample_labels, self.x_size, self.y_size, numlabs=numlabs
        )

    @classmethod
    def load_cod(cls, file_path: str) -> "SOMTrainer":
        """Load a SOM_PAK ``.cod`` codebook into a new trainer (SPEC-0002 FR-4).

        The returned trainer holds the codebook weights, grid size and topology
        from the file but has **no data** (use :meth:`set_data` to attach data
        before further training or evaluation).

        Note:
            Constructing the trainer creates a ``checkpoints/`` directory in the
            working directory (a side effect of ``__init__``).

        Args:
            file_path: Path to the ``.cod`` file.

        Returns:
            A :class:`SOMTrainer` with ``weights`` set from the file.

        Raises:
            CodFormatError: If the ``.cod`` file is malformed or has an unknown
                topology string.
        """
        header, weights = read_cod(file_path)
        dim = header["dim"]
        topol = header["topol"]
        if topol not in _COD_TO_TOPOLOGY:
            raise CodFormatError(
                f"{file_path}: unknown topology {topol!r}. "
                f"Expected one of {sorted(_COD_TO_TOPOLOGY)}."
            )
        som = cls(
            data=np.empty((0, dim)),
            size=(header["xdim"], header["ydim"]),
            input_dim=dim,
            learning_rate=0.0,
            topology=_COD_TO_TOPOLOGY[topol],
        )
        som.set_weights(weights)
        return som

    # ====================
    # getter and setter
    # ====================

    def get_data(self):
        return self.data

    @staticmethod
    def _to_somdata(data) -> SOMData:
        """Normalize ndarray / Bunch / DatasetWrapper / SOMData into a SOMData.

        ndarray and Bunch/DatasetWrapper carry no per-sample metadata, so they
        become a ``SOMData`` with ``data`` only (all metadata None) — the trainer
        then takes the SPEC-0001 bit-identical fast path. A ``SOMData`` is kept
        as-is (its mask/weights/fixed are consumed by train_pak).
        """
        if isinstance(data, SOMData):
            return data
        arr = (
            data.data
            if hasattr(data, "data") and not isinstance(data, np.ndarray)
            else data
        )
        return SOMData(data=np.asarray(arr))

    def set_data(self, data: Bunch | DatasetWrapper | np.ndarray | SOMData) -> None:
        """
        Set the input data for the SOM.

        :param data: The input data for the SOM (ndarray, Bunch, or SOMData).
        """
        self._somdata = self._to_somdata(data)
        self.data = self._somdata.data
        self.target = getattr(data, "target", np.array([]))
        self.target_names = getattr(data, "target_names", np.array([]))

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the weights of the SOM.

        :param weights: The weights of the SOM.
        """
        self.weights = weights

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the SOM.

        :return: The weights of the SOM.
        """
        return self.weights

    def set_function(self, n_func: Callable) -> None:
        """
        Set the neighborhood function for the SOM.

        :param n_func: The neighborhood function to use for updating weights.
        """
        self.n_func = n_func

    def update_radius(self, radius: float):
        self.n_radius = radius

    def get_radius(self):
        return self.n_radius


def create_trainer(
    data: Bunch | DatasetWrapper | np.ndarray | SOMData,
    size: Tuple[int, int],
    learning_rate: float,
    n_func: Callable = gaussian,
    initial_radius: float = __n_radius__,
    dynamic_radius: bool = __dynamic_radius__,
    checkpoint_interval: int = __checkpoint_interval__,
    random_seed: int | None = None,
    tau: float | None = None,
    topology: str = "hexagonal",
):
    if isinstance(data, np.ndarray):
        input_dim = data.shape[1]
    elif isinstance(data, SOMData):
        input_dim = data.data.shape[1]
    elif isinstance(data, Bunch) or isinstance(data, DatasetWrapper):
        input_dim = data.data.shape[1]
    else:
        raise ValueError(
            "Invalid input data type. The input data must be a numpy array, "
            "a Bunch/DatasetWrapper, or a SOMData."
        )

    return SOMTrainer(
        data,
        size,
        input_dim,
        learning_rate,
        n_func=n_func,
        initial_radius=initial_radius,
        dynamic_radius=dynamic_radius,
        checkpoint_interval=checkpoint_interval,
        random_seed=random_seed,
        tau=tau,
        topology=topology,
    )


def load_trainer(
    checkpoint_file_path: str,
    learning_rate: float,
    n_func: Callable,
    initial_radius: float | None = None,
    dynamic_radius: bool = __dynamic_radius__,
    tau: float | None = None,
) -> SOMTrainer:
    with h5py.File(checkpoint_file_path, "r") as f:
        _x_size = f.attrs["x_size"]
        _y_size = f.attrs["y_size"]
        _input_dim = f.attrs["input_dim"]
        _weights = f["weights"][:]
        _n_radius = f.attrs["n_radius"]
        _learning_rate = f.attrs["learning_rate"]
        _data = f["data"][:]
        _target = f["target"][:]
        _target_names = f["target_names"][:]
        _state = (
            f["random_state"]["0"][()].decode("utf-8"),
            f["random_state"]["1"][:],
            f["random_state"]["2"][()],
            f["random_state"]["3"][()],
            f["random_state"]["4"][()],
        )

    if len(_target_names) > 0:
        target_names = [name.decode("utf-8") for name in _target_names]

    rng = np.random.RandomState()
    rng.set_state(_state)

    if initial_radius is None:
        initial_radius = _n_radius

    som = SOMTrainer(
        data=_data,
        size=(_x_size, _y_size),
        input_dim=_input_dim,
        learning_rate=_learning_rate,
        n_func=n_func,
        initial_radius=initial_radius,
        dynamic_radius=dynamic_radius,
        rng=rng,
        tau=tau,
    )
    som.weights = _weights
    som.target = _target
    som.target_names = target_names
    return som
