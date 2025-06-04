# pinn.py

import abc
from typing import Callable, List
import torch
import torch.nn as nn

class PINN(abc.ABC, nn.Module):
    """
    Абстрактный базовый класс для Physics-Informed Neural Network.
    Определяет общий интерфейс и шаблонные методы.
    """

    def __init__(self, layers: List[int], activation: Callable = nn.Tanh):
        """
        :param layers: список из размеров слоёв [input_dim, ..., output_dim]
        :param activation: класс активации (по умолчанию nn.Tanh)
        """
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.net = self.build_network(layers, activation)

    def build_network(self, layers: List[int], activation: Callable) -> nn.Sequential:
        """
        Собирает последовательно Linear и активации.
        """
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: возвращает u(x).
        """
        pass

    @abc.abstractmethod
    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Должен вернуть остаток PDE в точках x (batch×1 → batch×1).
        """
        pass

    def loss(self,
             interior_points: torch.Tensor,
             boundary_points: torch.Tensor,
             equation: 'Equation',
             loss_fn: 'LossFunction',
             data_points: torch.Tensor = None,
             data_values: torch.Tensor = None) -> torch.Tensor:
        """
        Общая loss-функция для PINN:
        - PDE-residual на interior_points
        - boundary loss на boundary_points
        - (опционально) data loss
        """
        losses = []
        # 1) Residual loss (PDE)
        res = self.compute_residual(interior_points)
        losses.append(loss_fn.residual_loss(res))

        # 2) Boundary loss (Dirichlet u(0)=0, u(1)=0)
        if boundary_points is not None:
            pred_b = self.forward(boundary_points)
            true_b = equation.boundary_condition(boundary_points)
            losses.append(loss_fn.boundary_loss(pred_b, true_b))

        # 3) Data loss (если есть “точечные” измерения u_true)
        if data_points is not None and data_values is not None:
            pred_d = self.forward(data_points)
            losses.append(loss_fn.data_loss(pred_d, data_values))

        total_loss = torch.stack(losses).sum()
        return total_loss

    def train_model(self,
                    domain: 'DomainGenerator',
                    equation: 'Equation',
                    loss_fn: 'LossFunction',
                    optimizer: torch.optim.Optimizer,
                    events: list,
                    num_epochs: int = 10000,
                    n_interior: int = 1000,
                    n_boundary: int = 100,
                    device: torch.device = torch.device('cpu')):
        """
        Процесс обучения PINN:
        - Каждая эпоха генерирует n_interior точек внутри [0,1],
          n_boundary точек на границе (x=0 и x=1),
          (опционально) data_points для data loss.
        - Считает loss, делает шаг оптимизации, вызывает колбэки.
        """
        self.to(device)
        equation.to(device)

        for epoch in range(1, num_epochs + 1):
            # 1) генерируем точки
            interior_pts = domain.sample_interior(n_interior).to(device)       # shape (n_interior,1)
            boundary_pts = domain.sample_boundary(n_boundary)
            if boundary_pts is not None:
                boundary_pts = boundary_pts.to(device)

            # data points (можно задать отдельно, но пока None)
            data_pts = domain.sample_data(n_interior)
            data_vals = domain.sample_data_values(n_interior)
            if data_pts is not None:
                data_pts = data_pts.to(device)
                data_vals = data_vals.to(device)

            # 2) обнуляем градиенты
            optimizer.zero_grad()

            # 3) считаем loss
            total_loss = self.loss(interior_pts, boundary_pts, equation, loss_fn,
                                   data_points=data_pts, data_values=data_vals)

            # 4) шаг оптимизации
            total_loss.backward()
            optimizer.step()

            # 5) колбэки
            for event in events:
                event.on_epoch_end(epoch, self, total_loss.cpu().item())



# Конкретная PINN для 1D Пуассона
class PoissonPINN(PINN):
    """
    PINN для решения −u''(x) = f(x),  x∈[0,1],  u(0)=0, u(1)=0.
    """

    def __init__(self, layers: List[int], equation: 'PoissonEquation', activation: Callable = nn.Tanh):
        """
        :param layers: [1, ..., 1] (напр. [1, 50, 50, 1])
        :param equation: объект PoissonEquation (знает f(x) и BC)
        """
        super().__init__(layers, activation)
        self.equation = equation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch,1) → возвращает u(x) shape (batch,1)
        """
        return self.net(x)

    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Для Poisson: u''(x) − (−f(x)) = 0 → residual = u''(x) + f(x).
        Но в нашем Equation сделаем residual = u''(x) + f(x).
        """
        return self.equation.pde_operator(self, x)
