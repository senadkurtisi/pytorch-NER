import torch

from torch.utils.data import DataLoader

from dataloader import CoNLLDataset
from model import NERClassifier


def train_loop(config, writer, device):
    """Implements training of the model.

    Arguments:
        config (dict): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    # Define dataloader hyper-parameters
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "drop_last": True
    }
    valid_hyperparams = {
        "batch_size": config["batch_size"]["validation"],
        "shuffle": False,
        "drop_last": True
    }

    # Create dataloaders
    train_set = CoNLLDataset(config, config["dataset_path"]["train"])
    valid_set = CoNLLDataset(config, config["dataset_path"]["validation"])
    train_loader = DataLoader(train_set, **train_hyperparams)
    valid_loader = DataLoader(valid_set, **valid_hyperparams)

    model = NERClassifier(config)
    model = model.to(device)

    # Load training configuration
    train_config = config["train_config"]

    # Prepare the model optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=config["l2_penalty"]
    )
