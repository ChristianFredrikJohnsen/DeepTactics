"""
This file contains information about the MSE loss function.\n
The loss function is used to determine how far off the network's output is from the expected output.\n
The file contains a method for calculating loss, and a method for calculating the derivative of the loss function
with respect to the output of a neural network.
"""
import torch

def mse_loss(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """
    Calculates the mean squared error loss between the network's output and the expected output.\n
    Input should be two tensors of shape (num_neurons, batch_size)\n
    Output should be a single number, representing the mean of squared difference between every (expected, actual) pair.\n
    - Returns: float representing the loss
    """
    return torch.mean(torch.square(y - y_hat))/2

def mse_prime(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the derivative of the MSE loss function with respect to the network's output.\n
    Input should be:
    - [y] a tensor of shape (num_neurons, batch_size), representing the target output
    - [ŷ] a tensor of shape (num_neurons, batch_size), representing the network's output.\n
    Returns:
    - tensor of shape (num_neurons, batch_size), representing the gradient of the loss function with respect to the network's output.
    """
    return y_hat - y