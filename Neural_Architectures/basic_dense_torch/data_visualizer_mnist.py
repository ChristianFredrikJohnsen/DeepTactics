"""
File which contains functions for visualizing the pictures in the MNIST dataset, and the network's predictions for them.\n
There is a function for plotting a single sample, and a function for plotting multiple samples at once.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING # Getting around circular imports
if TYPE_CHECKING:
    from basic_net_torch import Network


def visualize_sample(net: 'Network', test_data: torch.Tensor, index: int = 0) -> None:
    """
    Visualizes a single sample and the network's prediction for it.

    Parameters:
    - net: The network which will be used to predict the sample.
    - test_data: The test data which will be used to get the sample.
    - index: The index of the picture you want to visualize.
    """
    _ , (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sample = test_data[0][index].cpu().numpy().reshape(28, 28) # Get the first sample in the test data, and reshape it to a 28x28 tensor.
    predictions = net.feedforward(test_data[0][index].unsqueeze(1)).cpu().numpy() # Get the network's predictions for the sample.

    ax1.imshow(sample, cmap = 'gray')
    ax1.set_title(f'Prediction: {net.feedforward(test_data[0][index].unsqueeze(1)).argmax(dim=0).item()} | Label: {test_data[1][index].item()}')

    ax2.bar(np.arange(10), predictions.flatten())
    ax2.set_title("Neural Network predictions")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(np.arange(10))
    plt.show()

def visualize_samples(net: 'Network', test_data: torch.Tensor, start_index: int = 0, num_samples: int = 10) -> None:
    """
    Method for visualizing multiple samples and corresponding predictions at once.\n
    
    Parameters:
    - net: The network which will be used to predict the samples.
    - test_data: The test data which will be used to get the samples.
    - start_index: The index of the first picture you want to visualize.
    - num_samples: The number of samples to visualize. (Shouldn't be much more than 10)

    Makes a plot with two columns. The first column contains the images, and the second column contains the network's predictions for the images.
    """
    n = test_data[0][start_index : start_index + num_samples].size(0) # Number of samples in the test data.
    if n <= 1:
        raise ValueError(f"Start index is too large. Valid range is [0, {test_data[0].size(0) - 2}]")
    
    samples = test_data[0][start_index : start_index + n].cpu().numpy().reshape(n, 28, 28) # Get the images
    
    predictions = [] # Get the predictions for the images
    for tensor in test_data[0][start_index : start_index + n]:
        predictions.append(net.feedforward(tensor.unsqueeze(1)).cpu().numpy())
    
    fig, axes = plt.subplots(n, 2, figsize = (12, n)) # Start plotting the images and corresponding predictions.
    
    for i in range(n):
        axes[i, 0].imshow(samples[i], cmap = 'gray')
        axes[i, 0].set_title(f'Label: {test_data[1][start_index+i].item()} | Prediction: {np.argmax(predictions[i])}', fontsize = 8)

        axes[i, 1].bar(np.arange(10), predictions[i].flatten())
        axes[i, 1].set_ylabel("Probability", fontsize = 8)
        axes[i, 1].set_xticks(np.arange(10))
    
    fig.suptitle("NN predictions", x=0.78, y=0.98, ha='right', va='top', fontsize=16, color='turquoise', fontweight = 'bold')  # Add title in the top right corner with bold font
    
    plt.tight_layout()
    plt.show()