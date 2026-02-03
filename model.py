import math
import torch
import torch.nn as nn

class MLP2Hidden(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(d_in, width, bias=False)   # W1
        self.fc2 = nn.Linear(width, width, bias=False)  # W2
        self.fc3 = nn.Linear(width, d_out, bias=False)  # W3
        self.act = nn.SELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    def init_weights_mup(self):
        n = self.width
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 1.0)                  # Var=1
            self.fc2.weight.normal_(0.0, 1.0 / math.sqrt(n))   # Var=1/n
            self.fc3.weight.normal_(0.0, 1.0 / n)              # Var=1/n^2
    
    def init_weights_ntk(self):
        n = self.width
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 1.0)                  # Var=1
            self.fc2.weight.normal_(0.0, 1.0 / math.sqrt(n))   # Var=1/n
            self.fc3.weight.normal_(0.0, 1.0 / math.sqrt(n))   # Var=1/n

    def init_optimizer_mup(self, eta: float):
        n = self.width
        return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta * n},  # eta1
                {"params": [self.fc2.weight], "lr": eta},      # eta2
                {"params": [self.fc3.weight], "lr": eta / n},  # eta3
            ],
        )
    
    def init_optimizer_ntk(self, eta: float):
        n = self.width
        return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
                {"params": [self.fc3.weight], "lr": eta / n},  # eta3
            ],
        )


class MLP1Hidden(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(d_in, width, bias=False)   # W1
        self.fc2 = nn.Linear(width, d_out, bias=False)  # W2
        self.act = nn.SELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)

    def init_weights_mup(self):
        n = self.width
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 1.0)                  # Var=1
            self.fc2.weight.normal_(0.0, 1.0 / n)              # Var=1/n^2
    
    def init_weights_ntk(self):
        n = self.width
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 1.0)                  # Var=1
            self.fc2.weight.normal_(0.0, 1.0 / math.sqrt(n))   # Var=1/n

    def init_optimizer_mup(self, eta: float):
        n = self.width
        return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta * n},  # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
            ],
        )
    
    def init_optimizer_ntk(self, eta: float):
        n = self.width
        return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
            ],
        )