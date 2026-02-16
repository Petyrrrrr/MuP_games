import math
import torch
import torch.nn as nn

class MLP2NTP(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int, type: str = "NTP"):
        super().__init__()
        self.width = width
        self.d_in = d_in
        self.fc1 = nn.Linear(d_in, width, bias=False)   # W1
        self.fc2 = nn.Linear(width, width, bias=False)  # W2
        self.fc3 = nn.Linear(width, d_out, bias=False)  # W3
        self.act = nn.SELU()
        self.type = type

    def forward(self, x):
        if self.type == "NTP":
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x) / math.sqrt(self.width)) 
        elif self.type == "MUP":
            x = self.act(self.fc1(x) * math.sqrt(self.width))
            x = self.act(self.fc2(x)) 
        return self.fc3(x) / math.sqrt(self.width)

    def init_weights(self):
        if self.type == "NTP":
            std = 1.0
        elif self.type == "MUP":
            std = 1.0 / math.sqrt(self.width)
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, std)                 
            self.fc2.weight.normal_(0.0, std)   
            self.fc3.weight.normal_(0.0, std)  

    def init_optimizer(self, eta: float, optimizer: str = "SGD"):
        if optimizer == "SGD":
            return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
                {"params": [self.fc2.weight], "lr": eta},  # eta2
                {"params": [self.fc3.weight], "lr": eta},  # eta3
            ],
        )
       
    def ntk_kernel(self, x_data):
        P = x_data.shape[0]
        assert x_data.shape == (P, self.d_in)
        ref = next(self.parameters())
        x_data = x_data.to(device=ref.device, dtype=ref.dtype)
        params = [p for p in self.parameters() if p.requires_grad]
        self.eval() 
        def flat_grad_vector():
            return torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in params
            ], dim=0)

        def get_jacobian(x: torch.Tensor) -> torch.Tensor:
            out = self.forward(x.unsqueeze(0)).reshape(-1)  # general: flatten all outputs
            m = out.numel()
            rows = []
            for k in range(m):
                self.zero_grad(set_to_none=True)
                out[k].backward( retain_graph = (k < m - 1) )
                rows.append(flat_grad_vector())
            return torch.cat(rows, dim=0)  # shape (m * num_params,)

        with torch.enable_grad():
            jacs = torch.stack([get_jacobian(x_data[i]) for i in range(P)], dim=0)  # (P, D)
            K = jacs @ jacs.T  # (P, P)
        self.train()
        return K

class MLP2Hidden(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int):
        super().__init__()
        self.width = width
        self.d_in = d_in
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

    def init_optimizer_mup(self, eta: float, optimizer: str = "SGD"):
        n = self.width
        if optimizer == "SGD":
            return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta * n},  # eta1
                {"params": [self.fc2.weight], "lr": eta},      # eta2
                {"params": [self.fc3.weight], "lr": eta / n},  # eta3
            ],
        )
        elif optimizer == "Adam":
            return torch.optim.Adam(
            [
                {"params": [self.fc1.weight], "lr": eta },         # eta1
                {"params": [self.fc2.weight], "lr": eta / n},      # eta2
                {"params": [self.fc3.weight], "lr": eta / n},      # eta3
            ],
        )
    
    def init_optimizer_ntk(self, eta: float, optimizer: str = "SGD"):
        n = self.width
        if optimizer == "SGD":
            return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
                {"params": [self.fc3.weight], "lr": eta / n},  # eta3
            ],
        )
        elif optimizer == "Adam":
            raise ValueError("Adam optimizer is not supported for NTK")
            return torch.optim.Adam(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
                {"params": [self.fc3.weight], "lr": eta / n},  # eta3
            ],
        )

    def ntk_kernel(self, x_data):
        P = x_data.shape[0]
        assert x_data.shape == (P, self.d_in)
        ref = next(self.parameters())
        x_data = x_data.to(device=ref.device, dtype=ref.dtype)
        params = [p for p in self.parameters() if p.requires_grad]

        self.eval() 
        
        def flat_grad_vector():
            return torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in params
            ], dim=0)

        def get_jacobian(x: torch.Tensor) -> torch.Tensor:
            # Forward ONCE per x (so all output elements share the same forward randomness)
            out = self.forward(x.unsqueeze(0)).reshape(-1)  # general: flatten all outputs
            m = out.numel()
            rows = []
            for k in range(m):
                self.zero_grad(set_to_none=True)
                out[k].backward(retain_graph=(k < m - 1))
                rows.append(flat_grad_vector())
            return torch.cat(rows, dim=0)  # shape (m * num_params,)

        with torch.enable_grad():
            jacs = torch.stack([get_jacobian(x_data[i]) for i in range(P)], dim=0)  # (P, D)
            K = jacs @ jacs.T  # (P, P)

        assert K.shape == (P, P)

        self.train()
        return K


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

    def init_optimizer_mup(self, eta: float, optimizer: str = "SGD"):
        n = self.width
        if optimizer == "SGD":
            return torch.optim.SGD(
            [
                {"params": [self.fc1.weight], "lr": eta * n},  # eta1
                {"params": [self.fc2.weight], "lr": eta / n},  # eta2
            ],
        )
        elif optimizer == "Adam":
            return torch.optim.Adam(
            [
                {"params": [self.fc1.weight], "lr": eta},      # eta1
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