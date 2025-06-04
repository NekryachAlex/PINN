# utils.py

import matplotlib.pyplot as plt

def default_poisson_visualizer(x: list, u_pred: list):
    """
    Визуализатор решения 1D: x_np (size N), u_pred (size N).
    """
    plt.plot(x, u_pred, 'b-', label='PINN approximation')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
