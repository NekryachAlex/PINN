# equation.py

import abc
import torch
import numpy as np

class Equation(abc.ABC, torch.nn.Module):
    """
    Базовый класс для PDE. Дочерние классы должны реализовать:
    - pde_operator(model, x) → residual
    - boundary_condition(x) → true u на границе
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def pde_operator(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет residual PDE в точках x.
        """
        pass

    @abc.abstractmethod
    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает истинные значения u на граничных точках x.
        """
        pass


class PoissonEquation(Equation):
    """
    Одномерное уравнение Пуассона:
        −u''(x) = f(x),  x∈[0,1],  u(0)=0, u(1)=0.
    """

    def __init__(self, f: callable):
        """
        :param f: функция f(x), возвращающая torch.Tensor shape=(batch,1).
        """
        super().__init__()
        self.f = f

    def pde_operator(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch,1). Мы вычисляем:
            u = model(x)          (batch,1)
            u_x = du/dx           (batch,1)
            u_xx = d^2u/dx^2      (batch,1)
        residual = u_xx + f(x)   (batch,1)  [потому что −u_xx = f → u_xx + f = 0]
        """
        x = x.clone().detach().requires_grad_(True)
        u = model(x)  # (batch,1)

        # 1-й производная du/dx
        grad_u = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]  # (batch,1)

        # 2-й производная d²u/dx²
        grad_u_x = torch.autograd.grad(
            outputs=grad_u, inputs=x,
            grad_outputs=torch.ones_like(grad_u),
            create_graph=True
        )[0]  # (batch,1)

        u_xx = grad_u_x  # (batch,1)
        fx = self.f(x)   # (batch,1)

        residual = u_xx + fx
        return residual

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dirichlet: u(0)=0, u(1)=0. 
        Предполагаем, что boundary_sampler дает x∈{0,1}.
        Тогда возвращаем нули нужного размера.
        """
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 1), dtype=torch.float32, device=x.device)
