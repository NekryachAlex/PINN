# losses.py

import torch
import torch.nn as nn

class LossFunction:
    """
    Базовый класс для loss-функций PINN:
    - residual_loss
    - boundary_loss
    - data_loss
    """

    def __init__(self, weights: dict = None):
        """
        :param weights: например {'residual':1.0, 'boundary':1.0, 'data':0.0}
        """
        default = {'residual': 1.0, 'boundary': 1.0, 'data': 0.0}
        self.weights = default if weights is None else {**default, **weights}
        self.mse = nn.MSELoss()

    def residual_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """
        L2-ошибка для PDE-остатка: ||residual||²
        """
        return self.weights['residual'] * self.mse(residual, torch.zeros_like(residual))

    def boundary_loss(self, pred_b: torch.Tensor, true_b: torch.Tensor) -> torch.Tensor:
        """
        L2 на границе (Dirichlet).
        """
        return self.weights['boundary'] * self.mse(pred_b, true_b)

    def data_loss(self, pred_d: torch.Tensor, true_d: torch.Tensor) -> torch.Tensor:
        """
        L2 на “точечных” данных.
        """
        return self.weights['data'] * self.mse(pred_d, true_d)
