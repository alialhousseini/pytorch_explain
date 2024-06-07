import torch
import torch.nn as nn
import torch.nn.functional as F


class AVG(nn.Module):
    """Implementation of Global Average Pooling (GAP)"""

    def __init__(self):
        super(AVG, self).__init__()

    def forward(self, x):
        """
        Performs the forward pass of the GAP pooling layer.

        Parameters:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            torch.Tensor: The output tensor after applying the AVG pooling operation.

        Explanation:
            torch.mean(x, dim=[-2, -1]) calculates the mean of the input tensor along the spatial dimensions (height and width).
            x.flatten(1) flattens the pooled tensor starting from the first dimension, converting it into a 1D vector per example in the batch.
            F.normalize(x, p=2, dim=1) normalizes these vectors to have unit norm.

        Scratch is better than AvgPool2d :)
        """
        x = torch.mean(x, dim=[-2, -1])  # Global average pooling
        x = x.flatten(1)  # Flatten to a vector
        return F.normalize(x, p=2, dim=1)  # Normalize the vectors


# Example usage:
# model = nn.Sequential(
#     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     AVG()
# )
# input_tensor = torch.randn(10, 3, 32, 32)  # Example input
# output_tensor = model(input_tensor)
# print(output_tensor.shape)
