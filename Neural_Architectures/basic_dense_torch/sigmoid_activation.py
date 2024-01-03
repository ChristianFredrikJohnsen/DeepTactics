"""
File containing the sigmoid activation function and its derivative.
"""
import torch

def sigmoid_prime(z: torch.Tensor):
    """
    The derivative of the sigmoid activation function.\n
    Input is usually a tensor of shape (num_neurons, batch_size)
    """
    sig = 1 / (1 + torch.exp(-z))
    return sig * (1 - sig)

def sigmoid(z: torch.Tensor):
    """
    Applies the sigmoid activation function to each element in the tensor.\n
    Input is usually a tensor of shape (num_neurons, batch_size)
    """
    return 1 / (1 + torch.exp(-z))

def quick_sigmoid(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the sigmoid function and its derivative at the same time.
    Reuses the calculation of the sigmoid function to save time.\n
    Input is usually a tensor of shape (num_neurons, batch_size)\n
    CAUTION: In-place operation is used for maximum efficiency (The input tensor is changed)\n
    Returns: ( σ(z), σ'(z) )
    """
    sig = 1 / (1 + z.neg_().exp_()) # In-place operation is used. Less data needs to be moved around, about 30% faster than the non-in-place version.
    return (sig, sig * (1 - sig))