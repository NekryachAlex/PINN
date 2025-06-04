# run_pinn.py

import yaml
import torch
import torch.optim as optim
import numpy as np

from pinn import PoissonPINN
from domain import DomainGenerator
from equation import PoissonEquation
from losses import LossFunction
from events import (
    PlotLoss,
    PlotSolution,
    Checkpoint,
    RelativeRMSE,
    OnnxExport,
    ProgressBar
)
from utils import default_poisson_visualizer


def setup_training(config_path: str = "config.yaml"):
    """
    Загружает конфиг, создаёт все объекты (domain, equation, loss_fn, model, optimizer, events, device и т.п.)
    и возвращает их в виде кортежа.
    """
    # 1. Загружаем конфиг
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Извлекаем параметры из конфига
    pinn_class_name = config["pinn_class"]       # "PoissonPINN"
    layers = config["layers"]                    # [1, 50, 50, 1]
    n_interior = config["n_interior"]            # 1024
    n_boundary = config["n_boundary"]            # 2

    opt_name = config["optimizer"]["name"]       # "Adam"
    lr = config["optimizer"]["lr"]               # 0.001
    loss_weights = config["loss_weights"]        # {'residual':1.0, 'boundary':1.0, 'data':0.0}
    num_epochs = config["num_epochs"]            # 3000
    device_name = config.get("device", "cpu")    # "cuda" или "cpu"

    # Частоты из конфига (опционально)
    plot_loss_freq = config.get("plot_loss_freq", 100)
    plot_solution_freq = config.get("plot_solution_freq", 20)
    checkpoint_save_freq = config.get("checkpoint_save_freq", 500)
    relative_rmse_freq = config.get("relative_rmse_freq", 100)

    # 3. DomainGenerator для [0,1]
    interior_bounds = np.array([[0.0, 1.0]])

    def sample_poisson_boundary(n):
        """
        Всегда возвращаем ровно две граничные точки: x=0 и x=1.
        """
        return np.array([[0.0], [1.0]], dtype=np.float32)

    # Data loss не используем → data_sampler=None
    domain = DomainGenerator(
        interior_bounds=interior_bounds,
        boundary_sampler=sample_poisson_boundary,
        data_sampler=None,
        data_values_sampler=None
    )

    # 4. Определяем f(x) = sin(pi x)
    def f_func(x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x_tensor)

    equation = PoissonEquation(f=f_func)

    # 5. LossFunction
    loss_fn = LossFunction(weights=loss_weights)

    # 6. Выбираем класс PINN
    pinn_classes = {
        "PoissonPINN": PoissonPINN,
        # при необходимости можно добавить другие PINN-классы
    }
    if pinn_class_name not in pinn_classes:
        raise ValueError(f"PINN класс '{pinn_class_name}' не найден. Доступные: {list(pinn_classes.keys())}")
    PINNClass = pinn_classes[pinn_class_name]

    # 7. Инициализируем модель
    model = PINNClass(layers=layers, equation=equation, activation=torch.nn.Tanh)

    # 8. Создаём оптимизатор
    optimizers_map = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
    }
    if opt_name not in optimizers_map:
        raise ValueError(f"Оптимизатор '{opt_name}' не поддерживается. Доступные: {list(optimizers_map.keys())}")
    optimizer = optimizers_map[opt_name](model.parameters(), lr=lr)

    # 9. Настраиваем колбэки (events)
    events = []

    # 9.1. PlotLoss
    plot_loss = PlotLoss(
        save_path="plots/loss.png",
        display=False,
        plot_freq=plot_loss_freq
    )
    events.append(plot_loss)

    # 9.2. PlotSolution
    x_vis = np.linspace(0, 1, 200).reshape(-1, 1)   # (200,1)
    sample_points_vis = torch.tensor(x_vis, dtype=torch.float32)

    plot_solution = PlotSolution(
        sample_points=sample_points_vis,
        visualizer=default_poisson_visualizer,
        save_dir="plots/solution",
        display=False,
        plot_freq=plot_solution_freq
    )
    events.append(plot_solution)

    # 9.3. Checkpoint
    checkpoint = Checkpoint(
        filepath_template="checkpoints/poisson_epoch_{epoch}_loss_{loss:.4f}.pt",
        save_freq=checkpoint_save_freq
    )
    events.append(checkpoint)

    # 9.4. RelativeRMSE
    def true_poisson_solution(x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x_tensor) / (torch.pi**2)

    relative_rmse = RelativeRMSE(
        sample_points=sample_points_vis,
        true_solution_fn=true_poisson_solution,
        print_freq=relative_rmse_freq
    )
    events.append(relative_rmse)

    # 9.5. ONNX Export (если включено в конфиг)
    if config.get("onnx", {}).get("export", False):
        onnx_cfg = config["onnx"]
        export_path = onnx_cfg.get("export_path", "models/poisson.onnx")
        opset = onnx_cfg.get("opset_version", 13)
        sample_for_onnx = torch.tensor([[0.5]], dtype=torch.float32)
        onnx_export = OnnxExport(
            export_path=export_path,
            sample_input=sample_for_onnx,
            opset_version=opset,
            export_epoch=num_epochs
        )
        events.append(onnx_export)

    # 9.6. ProgressBar (каждая эпоха)
    progress_bar = ProgressBar(total_epochs=num_epochs, bar_length=30)
    events.append(progress_bar)

    # 10. Устанавливаем устройство (CPU или GPU)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    # Возвращаем все объекты, которые понадобятся для train_model
    return {
        "domain": domain,
        "equation": equation,
        "loss_fn": loss_fn,
        "model": model,
        "optimizer": optimizer,
        "events": events,
        "num_epochs": num_epochs,
        "n_interior": n_interior,
        "n_boundary": n_boundary,
        "device": device
    }


if __name__ == "__main__":
    # Здесь у вас уже одна единственная строка, которая собирает все нужные объекты:
    params = setup_training("config.yaml")

    # И после этого остаётся только один вызов train_model:
    params["model"].train_model(
        domain=params["domain"],
        equation=params["equation"],
        loss_fn=params["loss_fn"],
        optimizer=params["optimizer"],
        events=params["events"],
        num_epochs=params["num_epochs"],
        n_interior=params["n_interior"],
        n_boundary=params["n_boundary"],
        device=params["device"]
    )
