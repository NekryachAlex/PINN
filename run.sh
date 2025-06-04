#!/usr/bin/env bash
set -e

# 1) Устанавливаем зависимости (если ещё не установлены)
#    Можно закомментировать эти строки, если уже всё установлено.
#pip install --upgrade pip
#pip install torch torchvision matplotlib numpy pyyaml onnx onnxruntime
# 2) Создаём необходимые директории для артефактов (plots, checkpoints, models)
mkdir -p plots/solution
mkdir -p checkpoints
mkdir -p models

# 3) Запускаем обучение
#    Конфиг читается из config.yaml
python run_pinn.py

# 4) По завершении можно вывести базовую информацию о результатах
echo ""
echo "Training finished."
echo "  - loss curve:    plots/loss.png"
echo "  - solution plots: plots/solution/solution_epoch_*.png"
echo "  - checkpoints:    checkpoints/"
echo "  - ONNX model:     models/poisson.onnx"
