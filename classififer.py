import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import PositionalEncodings
from transformer import TransformerEncoder


class NERClassifier(nn.Module):
    """Represents model which classifies named entities in the given body of text."""

    def __init__(self, config):
        """Initializes the module."""
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
        x = self.embedding_layer(x)
        x = self.entry_mapping(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        x = x.permute(1, 0, 2)
        x, _ = self.transformer_encoder(x, padding_mask)
        x = x.permute(1, 0, 2)

        y_pred = self.classifier(x)
        return y_pred
