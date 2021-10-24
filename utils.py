import torch
import numpy as np

from torch.utils.data import DataLoader

from dataloader import CoNLLDataset
from model import NERClassifier

from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, dataloader, writer, device, class_mapping=None):
    if class_mapping is None:
        raise ValueError("Argument @class_mapping not provided!")

    y_true_accumulator = []
    y_pred_accumulator = []

    for x, y, padding_mask in dataloader:
        x, y = x.to(device), y.to(device)
        padding_mask = padding_mask.to(device)
        y_pred = model(x, padding_mask)

        # Extract predictions and labels only for pre-padding tokens
        unpadded_mask = torch.logical_not(padding_mask)
        y_pred = y_pred[unpadded_mask]
        y = y[unpadded_mask]

        y_pred = y_pred.argmax(dim=1)
        y_pred = y_pred.view(-1).detach().cpu().tolist()
        y = y.view(-1).detach().cpu().tolist()

        y_true_accumulator += y
        y_pred_accumulator += y_pred

    # Map the integer labels back to NER tags
    y_pred_accumulator = [class_mapping[str(pred)] for pred in y_pred_accumulator]
    y_true_accumulator = [class_mapping[str(pred)] for pred in y_true_accumulator]

    y_pred_accumulator = np.array(y_pred_accumulator)
    y_true_accumulator = np.array(y_true_accumulator)

    # Extract labels and predictions where target label isn't O
    non_O_ind = np.where(y_true_accumulator != "O")
    y_pred_non_0 = y_pred_accumulator[non_O_ind]
    y_true_non_0 = y_true_accumulator[non_O_ind]

    # Calculate accuracy
    accuracy_total = accuracy_score(y_true_accumulator, 
                                    y_pred_accumulator)
    accuracy_non_O = accuracy_score(y_true_non_0,
                                    y_pred_non_0)

    # Calculate F1 score
    f1_total = f1_score(y_true_accumulator,
                        y_pred_accumulator,
                        average="weighted")
    f1_non_O = f1_score(y_true_non_0,
                        y_pred_non_0,
                        average="weighted")


def train_loop(config, writer, device):
    """Implements training of the model.

    Arguments:
        config (dict): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    reverse_class_mapping = {
        str(idx): cls_name for cls_name, idx in config["class_mapping"].items()
    }
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

            # Extract predictions and labels only for pre-padding tokens
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
