import os
import shutil
import time

import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import CoNLLDataset
from classifier import NERClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report


def log_gradient_norm(model, writer, step, mode, norm_type=2):
    """Writes model param's gradients norm to tensorboard"""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    writer.add_scalar(f"Gradient/{mode}", total_norm, step)


def save_checkpoint(model, start_time, epoch):
    """Saves specified model checkpoint."""
    target_dir = f"checkpoints\\{start_time}"
    os.makedirs(target_dir, exist_ok=True)
    # Save model weights
    save_path = f"{target_dir}\\model_{epoch}.pth"
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)

    # Save model configuration
    if not os.path.exists(f"{target_dir}\\config.json"):
        shutil.copy("config.json", os.path.join(target_dir, "config.json"))
        shutil.copy("classifier.py", os.path.join(target_dir, "classifier.py"))
        shutil.copy("transformer.py", os.path.join(target_dir, "transformer.py"))
        shutil.copy("utils.py", os.path.join(target_dir, "utils.py"))


def evaluate_model(model, dataloader, writer, device, mode, step, class_mapping=None):
    """Evaluates the model performance."""
    if mode not in ["Train", "Validation"]:
        raise ValueError(
            f"Invalid value for mode! Expected 'Train' or 'Validation but received {mode}"
        )

    if class_mapping is None:
        raise ValueError("Argument @class_mapping not provided!")

    y_true_accumulator = []
    y_pred_accumulator = []

    print("Started model evaluation.")
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

    # Calculate and log accuracy
    accuracy_total = accuracy_score(y_true_accumulator, 
                                    y_pred_accumulator)
    accuracy_non_O = accuracy_score(y_true_non_0,
                                    y_pred_non_0)
    writer.add_scalar(f"{mode}/Accuracy-Total",
                      accuracy_total, step)
    writer.add_scalar(f"{mode}/Accuracy-Non-O",
                      accuracy_non_O, step)

    # Calculate and log F1 score
    f1_total = f1_score(y_true_accumulator,
                        y_pred_accumulator,
                        average="weighted")
    f1_non_O = f1_score(y_true_non_0,
                        y_pred_non_0,
                        average="weighted")
    writer.add_scalar(f"{mode}/F1-Total",
                      f1_total, step)
    writer.add_scalar(f"{mode}/F1-Non-O",
                      f1_non_O, step)

    print(classification_report(y_true_accumulator, y_pred_accumulator, digits=4))


def train_loop(config, writer, device):
    """Implements training of the model.

    Arguments:
        config (dict): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
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

    # Instantiate the model
    model = NERClassifier(config)
    model = model.to(device)

    # Load training configuration
    train_config = config["train_config"]
    learning_rate = train_config["learning_rate"]

    # Prepare the model optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["l2_penalty"]
    )

    # Weights used for Cross-Entropy loss
    # Calculated as log(1 / (class_count / train_samples))
    # @class_count: Number of tokens in the corpus per each class
    # @train_samples:  Total number of samples in the trains set
    class_w = train_config["class_w"]
    class_w = torch.tensor(class_w).to(device)
    class_w /= class_w.sum()

    train_step = 0
    start_time = time.strftime("%b-%d_%H-%M-%S")
    for epoch in range(train_config["num_of_epochs"]):
        print("Epoch:", epoch)
        model.train()

        for x, y, padding_mask in train_loader:
            train_step += 1
            x, y = x.to(device), y.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            y_pred = model(x, padding_mask)

            # Extract predictions and labels only for pre-padding tokens
            unpadded_mask = torch.logical_not(padding_mask)
            y = y[unpadded_mask]
            y_pred = y_pred[unpadded_mask]

            loss = F.cross_entropy(y_pred, y, weight=class_w)

            # Update model weights
            loss.backward()

            log_gradient_norm(model, writer, train_step, "Before")
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clipping"])
            log_gradient_norm(model, writer, train_step, "Clipped")
            optimizer.step()

            writer.add_scalar("Train/Step-Loss", loss.item(), train_step)
            writer.add_scalar("Train/Learning-Rate", learning_rate, train_step)

        with torch.no_grad():
            model.eval()
            evaluate_model(model, train_loader, writer, device,
                           "Train", epoch, reverse_class_mapping)
            evaluate_model(model, valid_loader, writer, device,
                           "Validation", epoch, reverse_class_mapping)
            model.train()

        save_checkpoint(model, start_time, epoch)
        print()
