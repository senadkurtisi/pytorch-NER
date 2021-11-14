import os
import shutil

import torch


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
