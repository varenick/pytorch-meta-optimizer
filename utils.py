import math
import torch

def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = ((x.abs() + eps).log() / p * indicator - (1 - indicator)).view((-1, 1))
    x2 = (x.sign() * indicator + math.exp(p) * x * (1 - indicator)).view((-1, 1))

    return torch.cat((x1, x2), 1)

def copy_params(source, dest):
    for param_source, param_dest in zip(source.parameters(), dest.parameters()):
        param_dest.data.copy_(param_source.data)
