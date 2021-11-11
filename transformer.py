import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Represents Multi-Head Attention Module"""

    def __init__(self, num_heads, d_model):
        """Initializes the module.

        Arguments:
            num_heads (int): Number of attention heads
            d_model (int): Embedding dimension of each input token
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % num_heads) == 0, "Embedding dimension d_model must be divisible by the number of heads!"
        self.d_head = int(d_model // num_heads)
        self.num_heads = num_heads
        # Represents query, key, value matrices used for mapping the input sequence
        self.qkv_matrices = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out_projection = nn.Linear(d_model, d_model)

    def self_attention(self, query, key, value, padd_mask=None):
        """Performs scaled dot-product attention from 'Attention is All You Need'.

        Arguments:
            query (torch.Tensor): Query vector. Expected shape: (seq_len, batch_size, embedding_dim)
            key (torch.Tensor): Key vector. Expected shape: (seq_len, batch_size, embedding_dim)
            value (torch.Tensor): Value vector. Expected shape: (seq_len, batch_size, embedding_dim)
            padd_mask (torch.Tensor): Expected shape: (batch_size, seq_len)
                Usage: Specifies if some tokens should be ignored when calculating attention scores
        Returns:
            output (torch.Tensor): Represents attention combination of input tensors.
                Expected shape: (seq_len, batch_size, embedding_dim)
            attn_weights (torch.Tensor): Attention weights for each token
        """
        seq_len, batch_size, d_model = query.shape
        if padd_mask:
            assert padd_mask.shape == (batch_size, seq_len), f"Invalid mask shape! Expected shape of ({batch_size}, {seq_len})"
            padd_mask = padd_mask.view(batch_size, 1, 1, seq_len). \
                expand(-1, self.num_heads, -1, -1). \
                reshape(batch_size * self.num_heads, 1, seq_len)

        # We map the order of dimensions to (bsz * head_dim, seq_len, d_head)
        query = query.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)
        key = key.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)
        value = value.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)

        # Scores shape: (bsz * head_dim, seq_len, seq_len)
        attn_scores = torch.bmm(query, key.transpose(-2, -1))
        if padd_mask:
            attn_scores.masked_fill_(padd_mask == torch.tensor(True), float("-inf"))
        attn_scores /= (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Output shape: (bsz * head_dim, seq_lean, d_head)
        output = torch.bmm(attn_weights, value)
        # Map the output to the original input shape
        output = output.transpose(1, 0).contiguous().view(seq_len, batch_size, d_model)
        return output, attn_weights

    def forward(self, query, key, value, padd_mask=None):
        """Performs forward pass of the module"""
        # Map the input into query key and value
        query, key, value = [mapper_net(input_vec) for mapper_net, input_vec in zip(self.qkv_matrices, [query, key, value])]
        # Perform multi-head self-attention
        attn_output, attn_weights = self.self_attention(query, key, value, padd_mask)
        output = self.out_projection(attn_output)

        return output, attn_weights
