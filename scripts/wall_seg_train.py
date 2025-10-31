"""
train.py

Main training script for the wall segmentation model.
Handles argument parsing, training loop, validation, and checkpointing.
"""

import os
import torch
import torch.nn.functional as F
import argparse
import itertools
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Local imports
from wall_seg_dataset import WallSegDataset
from wall_seg_model import init_model, MultiLoss, affinity_loss
from wall_seg_validate import run_validation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
):
    """
    Runs a single training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        criterion (MultiLoss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run on.
        epoch_index (int): The current epoch number (for logging).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train(True)  # Set model to training mode
    running_loss = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_index}")

    for i, data in enumerate(progress_bar):
        img, mask = data
        img, mask = img.to(device), mask.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output_mask = model(img)

        # Calculate losses
        aff_loss = affinity_loss(output_mask)
        bce_loss = F.binary_cross_entropy_with_logits(output_mask, mask)
        loss = criterion(bce_loss, aff_loss)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update tqdm description
        if i % 100 == 99:
            avg_batch_loss = running_loss / (i + 1)
            progress_bar.set_postfix(loss=f"{avg_batch_loss:.4f}")

    return running_loss / (i + 1)




def main():
    """Main training script."""
    # Parser now only looks for the config file
    parser = argparse.ArgumentParser(description="Wall Segmentation Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wall_seg.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    logging.info(f"Loading configuration from {args.config}...")
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: Config file not found at {args.config}")
        return
    except Exception as e:
        logging.error(f"Error loading YAML: {e}")
        return

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Handle experiment name (use timestamp if not provided)
    exp_name = config["experiment"]["exp_name"]
    if not exp_name:
        exp_name = f"wall_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"No experiment name set, using timestamp: {exp_name}")

    # Create experiment directory
    exp_path = os.path.join(config["experiment"]["exp_dir"], exp_name)
    os.makedirs(exp_path, exist_ok=True)
    logging.info(f"Experiment artifacts will be saved to: {exp_path}")

    # --- Load Data ---
    logging.info("Loading datasets...")
    try:
        train_dataset = WallSegDataset(config["train_dest"], "train_tensors")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
        )

        val_loader = None
        if not config["training"]["no_val"]:
            val_dataset = WallSegDataset(config["val_dest"], "val_tensors")
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"]["num_workers"],
            )
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        logging.error("Please ensure you have run preprocess.py first.")
        return

    # --- Initialize Model, Loss, Optimizer ---
    logging.info("Initializing model...")
    model = init_model(config["hub_cache_dir"], device)
    criterion = MultiLoss().to(device)

    # Optimize both model parameters and loss parameters
    optimizer = torch.optim.SGD(
        itertools.chain(model.parameters(), criterion.parameters()),
        lr=config["training"]["lr"],
        momentum=config["training"]["momentum"],
    )

    # --- Training Loop ---
    best_loss = 1_000_000.0
    epochs = config["training"]["epochs"]

    logging.info(f"--- Starting Training for {epochs} Epochs ---")
    for epoch in range(epochs):

        # Training
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validation
        if not config["training"]["no_val"]:
            avg_vloss = run_validation(model, val_loader, criterion, device)
            logging.info(
                f"EPOCH {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}"
            )
            current_loss = avg_vloss
        else:
            logging.info(f"EPOCH {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")
            current_loss = avg_loss

        # Checkpointing
        if current_loss < best_loss:
            best_loss = current_loss
            model_name = f"model_epoch_{epoch+1}_loss_{best_loss:.4f}.pth"
            model_path = os.path.join(exp_path, model_name)

            # Save the model state dict.
            # If using DataParallel, save module.state_dict() for easier loading
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logging.info(f"Checkpoint saved to {model_path}")

    logging.info("--- Training Complete ---")


if __name__ == "__main__":
    main()
