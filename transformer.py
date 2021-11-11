import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Represents Multi-Head Attention Module"""

    def __init__(self, num_heads, d_model, ff_dim, p_dropout):
        """Initializes the module."""
        assert (num_heads % d_model) == 0, "Embedding dimension d_model must be divisible by the number of heads!"
        # Represents query, key, value matrices used for mapping the input sequence
        self.qkv_matrices = [nn.Linear(d_model, d_model) for _ in range(3)]

    def forward(self, query, key, value, padd_mask):
        """Performs forward pass of the module"""
        # Map the input into query key and value
        query, key, value = [mapper_net(input_vec) for mapper_net, input_vec in zip(self.qkv_matrices, [query, key, value])]
