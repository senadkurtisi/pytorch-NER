import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


class ScaleNorm(nn.Module):
    """Represents Scale Norm layer

    reference: https://github.com/tnq177/transformers_without_tears
    """
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        scale = nn.Parameter(torch.tensor(scale))
        self.register_buffer("eps", torch.Tensor([eps]))
        self.register_parameter("scale", scale)

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    """Represents the encoder of the "Attention is All You Need" Transformer"""

    def __init__(self, num_layers, num_heads, d_model, ff_dim, p_dropout):
        """Initializes the module.

        Arguments:
            num_layers (int): Number of Transformer encoder layers
            num_heads (int): Number of self-attention heads per layer
            d_model (int): Embedding dimension of every token
            ff_dim (int): Number of neurons of middle layer in the feedgforward segment
            p_dropout (float): Probability used for dropout layers
        """
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(num_heads, d_model, ff_dim, p_dropout) for _ in range(num_layers)
        ])
        self.scale_norm = ScaleNorm(d_model ** 0.5)

    def forward(self, x, padd_mask=None):
        """Performs forward pass of the module."""
        attn_weights_accumulator = []
        for encoder_layer in self.encoder_blocks:
            x, attn_weights = encoder_layer(x, padd_mask)
            attn_weights_accumulator.append(attn_weights)

        x = self.scale_norm(x)
        return x, attn_weights_accumulator


class TransformerEncoderLayer(nn.Module):
    """Represents a single Transformer Encoder Block."""

    def __init__(self, num_heads, d_model, ff_dim, p_dropout):
        """Initializes the module.

        Arguments:
            num_heads (int): Number of self-attention heads
            d_model (int): Embedding dimension of every token
            ff_dim (int): Number of neurons of middle layer in the feedgforward segment
            p_dropout (float): Probability used for dropout layers
        """
        super(TransformerEncoderLayer, self).__init__()
        self.scale_norm_1 = ScaleNorm(d_model ** 0.5)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, p_dropout)
        self.dropout_1 = nn.Dropout(p_dropout)

        self.scale_norm_2 = ScaleNorm(d_model ** 0.5)
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout_3 = nn.Dropout(p_dropout)

    def forward(self, x, padd_mask=None):
        """Performs forward pass of the module."""
        x = self.scale_norm_1(x)
        skip_connection = x
        attn_output, attn_weights = self.multi_head_attention(query=x, key=x, value=x, padd_mask=padd_mask)
        x = skip_connection + self.dropout_1(attn_output)

        x = self.scale_norm_2(x)
        skip_connection = x
        x = self.ff_net(x)
        x = skip_connection + self.dropout_3(x)

        return x, attn_weights


class MultiHeadAttention(nn.Module):
    """Represents Multi-Head Attention Module"""

    def __init__(self, num_heads, d_model, p_dropout):
        """Initializes the module.

        Arguments:
            num_heads (int): Number of attention heads
            d_model (int): Embedding dimension of each input token
            p_dropout (float): Probability used for dropout layers
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % num_heads) == 0, "Embedding dimension d_model must be divisible by the number of heads!"
        self.d_head = int(d_model // num_heads)
        self.num_heads = num_heads
        # Represents query, key, value matrices used for mapping the input sequence
        self.qkv_matrices = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p_dropout)
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
        if padd_mask is not None:
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
        if padd_mask is not None:
            attn_scores.masked_fill_(padd_mask == torch.tensor(True), float("-inf"))
        attn_scores /= (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

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
