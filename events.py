# events.py

import abc
import os
import sys
import torch
import matplotlib.pyplot as plt

class Event(abc.ABC):
    """
    Базовый класс для колбэков: on_epoch_end(self, epoch, model, loss_value)
    """
    @abc.abstractmethod
    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        pass


class PlotLoss(Event):
    """
    Рисует график loss по эпохам и (опционально) сохраняет/показывает.
    """

    def __init__(self, save_path: str = None, display: bool = False, plot_freq: int = 100):
        """
        :param save_path: путь для сохранения итогового изображения (каждый раз перезаписывается)
        :param display: если True, вызывает plt.show()
        :param plot_freq: как часто (по эпохам) обновлять график
        """
        self.history = []
        self.save_path = save_path
        self.display = display
        self.plot_freq = plot_freq

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        self.history.append(loss_value)
        if epoch % self.plot_freq != 0:
            return

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.history) + 1), self.history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Epoch (epoch={epoch})')
        plt.legend()
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(self.save_path)
        if self.display:
            plt.show()
        plt.close()


class PlotSolution(Event):
    """
    По заданным точкам sample_points рисует график u(x) и (опционально) сохраняет.
    """

    def __init__(self,
                 sample_points: torch.Tensor,
                 visualizer: callable,
                 save_dir: str = None,
                 display: bool = False,
                 plot_freq: int = 1):
        """
        :param sample_points: torch.Tensor shape=(N,1)
        :param visualizer: функция visualizer(x_np, pred_np)
        :param save_dir: директория, куда сохранять (файлы solution_epoch_{epoch}.png)
        :param display: если True — plt.show()
        :param plot_freq: печатать/сохранять каждые plot_freq эпох
        """
        self.sample_points = sample_points
        self.visualizer = visualizer
        self.save_dir = save_dir
        self.display = display
        self.plot_freq = plot_freq

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        if epoch % self.plot_freq != 0:
            return

        device = next(model.parameters()).device
        pts = self.sample_points.to(device)
        with torch.no_grad():
            preds = model(pts).cpu().numpy()  # (N,1)
        x_np = self.sample_points.cpu().numpy().flatten()  # (N,)

        plt.figure(figsize=(6, 4))
        self.visualizer(x_np, preds.flatten())
        plt.title(f'Solution at epoch {epoch}')
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            fname = os.path.join(self.save_dir, f"solution_epoch_{epoch}.png")
            plt.savefig(fname)
        if self.display:
            plt.show()
        plt.close()


class Checkpoint(Event):
    """
    Сохраняет модель каждые save_freq эпох или при улучшении loss.
    """

    def __init__(self, filepath_template: str, save_freq: int = 500):
        """
        :param filepath_template: шаблон пути, например "checkpoints/poisson_epoch_{epoch}_loss_{loss:.4f}.pt"
        :param save_freq: сохранять каждые save_freq эпох
        """
        self.filepath_template = filepath_template
        self.save_freq = save_freq
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        if (epoch % self.save_freq == 0) or (loss_value < self.best_loss):
            os.makedirs(os.path.dirname(self.filepath_template), exist_ok=True)
            save_path = self.filepath_template.format(epoch=epoch, loss=loss_value)
            torch.save(model.state_dict(), save_path)
            self.best_loss = min(self.best_loss, loss_value)


class RelativeRMSE(Event):
    """
    Колбэк для вычисления относительной RMSE между предсказанием и истинным решением.
    """

    def __init__(self, sample_points: torch.Tensor, true_solution_fn: callable, print_freq: int = 100):
        """
        :param sample_points: torch.Tensor shape=(N,1)
        :param true_solution_fn: функция true_solution_fn(x_tensor: torch.Tensor) -> torch.Tensor shape (N,1)
        :param print_freq: печатать каждые print_freq эпох
        """
        self.sample_points = sample_points
        self.true_solution_fn = true_solution_fn
        self.print_freq = print_freq

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        if epoch % self.print_freq != 0:
            return

        device = next(model.parameters()).device
        pts = self.sample_points.to(device)
        with torch.no_grad():
            preds = model(pts)                     # (N,1)
            true_vals = self.true_solution_fn(pts)  # (N,1)

        true_vals = true_vals.to(device)
        diff = preds - true_vals
        num = torch.norm(diff)               # L2
        den = torch.norm(true_vals)          # L2
        eps = 1e-8
        rel_rmse = (num / (den + eps)).item()
        print(f"[Epoch {epoch:5d}] Relative RMSE = {rel_rmse:.6e}")


class OnnxExport(Event):
    """
    Колбэк для экспорта модели в ONNX в заданную эпоху (export_epoch).
    """

    def __init__(self, export_path: str, sample_input: torch.Tensor,
                 opset_version: int = 13, export_epoch: int = None):
        """
        :param export_path: путь для ONNX-файла, например "models/poisson.onnx"
        :param sample_input: torch.Tensor shape=(M,1), вход для фиксации графа
        :param opset_version: версия ONNX opset (по умолчанию 13)
        :param export_epoch: если None, экспортируем в последнюю эпоху; если указано — только при epoch == export_epoch
        """
        self.export_path = export_path
        self.sample_input = sample_input
        self.opset_version = opset_version
        self.export_epoch = export_epoch

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        do_export = False
        if self.export_epoch is None:
            do_export = True
        elif epoch == self.export_epoch:
            do_export = True

        if not do_export:
            return

        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
        model_cpu = model.cpu()
        sample = self.sample_input.cpu()

        torch.onnx.export(
            model_cpu,
            sample,
            self.export_path,
            input_names=["x"],
            output_names=["u"],
            opset_version=self.opset_version,
        )
        print(f"ONNX model exported to {self.export_path}")


class ProgressBar(Event):
    """
    Колбэк, который печатает текстовую «шкалу» прогресса и количество пройденных/оставшихся эпох.
    """

    def __init__(self, total_epochs: int, bar_length: int = 30):
        """
        :param total_epochs: общее число эпох тренировки
        :param bar_length: длина «шкалы» в символах
        """
        self.total_epochs = total_epochs
        self.bar_length = bar_length

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, loss_value: float):
        progress = epoch / self.total_epochs
        filled = int(self.bar_length * progress)
        bar = '=' * filled + ' ' * (self.bar_length - filled)
        # \r — чтобы затирать предыдущую строку, без перевода строки
        sys.stdout.write(f"\rEpoch {epoch}/{self.total_epochs} [{bar}]")
        if epoch == self.total_epochs:
            sys.stdout.write('\n')
        sys.stdout.flush()
