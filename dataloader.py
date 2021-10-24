import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class CoNLLDataset(Dataset):
    """Implements CoNLL2003 dataset consumption class.

       Data is saved in .txt format.
       Each sample is in a distinct line in the following format:
        - sample_length[TAB]input_tokens[TAB]ner_labels_per_token
        - @input_tokens and @ner_labels_per_token are also separated by [TAB]
    """
    def __init__(self, config, path, separator="\t"):
        with open(path, "r", encoding="utf8") as f:
            self.data = f.readlines()
        self.data = [sample.replace("\n", "") for sample in self.data]

        # Load the vocabulary mappings
        with open(config["word2idx_path"], "r", encoding="utf8") as f:
            self._word2idx = json.load(f)
        self._idx2word = {str(idx): word for word, idx in self._word2idx.items()}

        # Set the default value for the OOV tokens
        self._word2idx = defaultdict(
            lambda: self._word2idx[config["OOV_token"]],
            self._word2idx
        )

        self._separator = separator
        self._PAD_token = config["PAD_token"]
        self._PAD_label = config["PAD_label"]
        self._max_len = config["max_len"]

        self._dataset_size = len(self.data)

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        sample = self.data[index]
        sample_decoupled = sample.split(self._separator)
        sample_size = int(sample_decoupled[0])

        # Extract input tokens and labels
        tokens = sample_decoupled[1:(sample_size + 1)]
        labels = sample_decoupled[(sample_size + 1):]
        # Pad the token and label sequences
        tokens = tokens[:self._max_len]
        labels = labels[:self._max_len]
        padding_size = self._max_len - sample_size
        if padding_size > 0:
            tokens += [self._PAD_token for _ in range(padding_size)]
            labels += [self._PAD_label for _ in range(padding_size)]

        # Apply the vocabulary mapping to the input tokens
        tokens = [token.strip().lower() for token in tokens]
        tokens = [self._word2idx[token] for token in tokens]
        tokens = torch.Tensor(tokens).long()

        # Adapt labels for PyTorch consumption
        labels = [int(label) for label in labels]
        labels = torch.Tensor(labels).long()

        # Define the padding mask
        padding_mask = torch.ones([self._max_len, ])
        padding_mask[:sample_size] = 0.0
        return tokens, labels
