import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def display_results(widths, eta_values, train_loss_histories, title=''):
    for idx, width in enumerate(widths):
        t_losses = [np.mean(train_loss_histories[width][lr][-200:]) for lr in eta_values]
        plt.plot(eta_values, t_losses, marker='s', alpha=0.6, linestyle='--', label=f'Width {width}')
    plt.xscale('log', base=2)
    plt.xlabel("Learning Rate")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title(title)
    plt.show()

def hermite_poly(n):
    norm = math.sqrt(math.factorial(n))
    def evaluate(x):
        return torch.special.hermite_polynomial_he(x, n) / norm
    return evaluate

def make_teacher(d: int, device, seed=42, teacher_width = 10000):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W1 = torch.randn(d, teacher_width, generator=g, device=device)
    W2 = torch.randn(teacher_width, teacher_width, generator=g, device=device) / math.sqrt(teacher_width)
    W3 = torch.randn(teacher_width, 1, generator=g, device=device) / math.sqrt(teacher_width)
    return (W1, W2, W3)