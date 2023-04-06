

class NeighborhoodFunction(ABC):

    @abstractmethod
    def compute(self, distance: np.ndarray, radius: float) -> np.ndarray:
        pass