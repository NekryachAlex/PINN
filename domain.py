# domain.py

import numpy as np
import torch

class DomainGenerator:
    """
    Генератор точек в области [0,1] для interior и boundary.
    Если требуется data loss, можно передать data_sampler и data_values_sampler.
    """

    def __init__(self,
                 interior_bounds: np.ndarray,
                 boundary_sampler: callable = None,
                 data_sampler: callable = None,
                 data_values_sampler: callable = None):
        """
        :param interior_bounds: np.ndarray shape=(1,2) с [min, max] для x
        :param boundary_sampler: функция генерирующая точки на границе (x=0 или x=1)
        :param data_sampler: функция для точек, где есть “истинные” значения
        :param data_values_sampler: функция, возвращающая истинные u в этих точках
        """
        self.interior_bounds = interior_bounds
        self.boundary_sampler = boundary_sampler
        self.data_sampler = data_sampler
        self.data_values_sampler = data_values_sampler

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """
        Равномерное сэмплирование n_samples точек внутри [0,1].
        """
        # dims = 1
        samples = np.random.rand(n_samples, 1)
        a, b = self.interior_bounds[0, 0], self.interior_bounds[0, 1]
        samples = samples * (b - a) + a
        return torch.tensor(samples, dtype=torch.float32)

    def sample_boundary(self, n_samples: int) -> torch.Tensor:
        """
        Если boundary_sampler прописан, вызывает его, иначе возвращает None.
        """
        if self.boundary_sampler is None:
            return None
        samples = self.boundary_sampler(n_samples)
        return torch.tensor(samples, dtype=torch.float32)

    def sample_data(self, n_samples: int) -> torch.Tensor:
        """
        Точки, где известны истинные u, если нужны для data loss.
        """
        if self.data_sampler is None:
            return None
        samples = self.data_sampler(n_samples)
        return torch.tensor(samples, dtype=torch.float32)

    def sample_data_values(self, n_samples: int) -> torch.Tensor:
        """
        Истинные значения u в точках data, для data loss.
        """
        if self.data_values_sampler is None:
            return None
        values = self.data_values_sampler(n_samples)
        return torch.tensor(values, dtype=torch.float32)
