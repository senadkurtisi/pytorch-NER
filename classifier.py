import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import TransformerEncoder, PositionalEncodings


class ResidualBlock(nn.Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x


class NERClassifier(nn.Module):
    """Represents model which classifies named entities in the given body of text."""

    def __init__(self, config):
        """Initializes the module."""
        super(NERClassifier, self).__init__()
        num_classes = len(config["class_mapping"])
        embedding_dim = config["embeddings"]["size"]
        num_of_transformer_layers = config["num_of_transformer_layers"]
        transformer_embedding_dim = config["transformer_embedding_dim"]
        attention_heads = config["attention_heads"]
        ff_dim = config["transformer_ff_dim"]
        dropout = config["dropout"]

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        self.entry_mapping = nn.Linear(embedding_dim, transformer_embedding_dim)
        self.res_block = ResidualBlock(transformer_embedding_dim)

        self.positional_encodings = PositionalEncodings(
            config["max_len"],
            transformer_embedding_dim,
            dropout
        )

        self.transformer_encoder = TransformerEncoder(
            num_of_transformer_layers,
            attention_heads,
            transformer_embedding_dim,
            ff_dim,
            dropout
        )
        self.classifier = nn.Linear(transformer_embedding_dim, num_classes)

    def forward(self, x, padding_mask):
        """Performs forward pass of the module."""
        # Get token embeddings for each word in a sequence
        x = self.embedding_layer(x)

        # Map input tokens to the transformer embedding dim
        x = self.entry_mapping(x)
        x = F.leaky_relu(x)
        x = self.res_block(x)
        x = F.leaky_relu(x)

        # Leverage the self-attention mechanism on the input sequence
        x = self.positional_encodings(x)
        x = x.permute(1, 0, 2)
        x, _ = self.transformer_encoder(x, padding_mask)
        x = x.permute(1, 0, 2)

        y_pred = self.classifier(x)
        return y_pred
