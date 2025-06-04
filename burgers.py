import torch
from equation import Equation

class BurgersEquation(Equation):
    def __init__(self, nu: float = 0.01 / np.pi):
        super().__init__()
        self.nu = nu

    def pde_operator(self, model, x: torch.Tensor) -> torch.Tensor:
        """
        Допустим, x имеет две компоненты (t, x_spatial).
        model(x) → u(t, x).
        Рассчитываем:
            u_t + u * u_x - nu * u_xx
        """
        x.requires_grad_(True)
        u = model(x)

        # Частные производные:
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]

        # Вторая производная по x:
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Зададим, что u(t, x=−1) = u(t, x=1) = 0 (Dirichlet).
        Предполагаем, что boundary sampler даёт точки, в которых x_spatial = ±1.
        Тогда возвращаем тензор нулей того же размера, что и model(x).
        """
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 1), dtype=torch.float32)
    
    @property
    def initial_condition(self):
        # u(0, x) = -sin(pi x)
        def u0(x: torch.Tensor) -> torch.Tensor:
            # x имеет форму [batch, 2]; интересует второй столбец (пространственная координата)
            x_spatial = x[:, 1:2]
            return -torch.sin(torch.pi * x_spatial)
        return u0
