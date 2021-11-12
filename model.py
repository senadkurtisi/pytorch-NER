import torch
import numpy as np

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
        x += self.positional_encodings
        x = self.dropout(x)
        return x


class NERClassifier(nn.Module):
    """Represents model which classifies """

    def __init__(self, config):
        super(NERClassifier, self).__init__()
        num_classes = len(config["class_mapping"])
        embedding_dim = config["embedding_dim"]
        num_of_transformer_layers = config["num_of_transformer_layers"]
        transformer_embedding_dim = config["transformer_embedding_dim"]
        attention_heads = config["attention_heads"]
        ff_dim = config["transformer_ff_dim"]
        dropout = config["dropout"]

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings_path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        self.entry_mapping = nn.Linear(embedding_dim, transformer_embedding_dim)
        self.positional_encodings = PositionalEncodings(
            config["max_len"],
            transformer_embedding_dim,
            dropout
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            transformer_embedding_dim,
            attention_heads,
            ff_dim,
            dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_of_transformer_layers
        )
        self.classifier = nn.Linear(transformer_embedding_dim, num_classes)

    def forward(self, x, padding_mask):
        x = self.embedding_layer(x)
        x = self.entry_mapping(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)

        y_pred = self.classifier(x)
        return y_pred
