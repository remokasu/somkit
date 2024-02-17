from __future__ import annotations

import numpy as np

cuda_available = False
array_types = np.ndarray

try:
    import cupy as cp

    cupy = cp
    cuda_available = True
    array_types = (cp.ndarray, np.ndarray)
except ImportError:
    pass


def get_device() -> cp.cuda.Device or None:
    if cuda_available:
        return cp.cuda.Device()
    else:
        return None


def get_device_count() -> int:
    if cuda_available:
        return cp.cuda.runtime.getDeviceCount()
    else:
        return 0


def get_array_module(data: np.ndarray) -> cp or np:
    """
    cupy.get_array_module(data) if cuda_available else numpy
    """
    if cuda_available:
        return cp.get_array_module(data)
    else:
        return np


def asnumpy(data: cp.ndarray) -> np.ndarray:
    """
    to numpy array
    """
    if cuda_available:
        return cp.asnumpy(data)
    else:
        return data


def ascupy(data: np.ndarray) -> cp.ndarray:
    """
    to cupy array
    """
    if cuda_available:
        return cp.asarray(data)
    else:
        raise ImportError(
            "CuPy is not available. Please install CuPy to use GPU acceleration."
        )
