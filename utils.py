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
        weight_decay=train_config["l2_penalty"]
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    for epoch in range(train_config["num_of_epochs"]):
        print("Epoch:", epoch)
        model.train()

        for x, y, padding_mask in train_loader:
            x, y = x.to(device), y.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            y_pred = model(x, padding_mask)

            # Extract predictions and only for pre-padding tokens
            unpadded_mask = torch.logical_not(padding_mask)
            y = y[unpadded_mask]
            y_pred = y_pred[unpadded_mask]

            # Calculate focal loss
            ce_loss = criterion(y_pred, y)
            y_pt = torch.exp(-ce_loss)
            loss = (1 - y_pt) ** train_config["focal_gamma"]
            loss *= ce_loss
            loss = train_config["focal_alpha"] * loss.mean()

            # Update model weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config["gradient_clipping"]
            )
            optimizer.step()
