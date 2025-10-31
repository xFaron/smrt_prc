"""
validate.py

Handles model validation and visualization of predictions.
Can be imported by train.py or run as a standalone script.
"""

import os
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from wall_seg_dataset import WallSegDataset
from wall_seg_model import init_model, MultiLoss, affinity_loss


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiLoss,
    device: torch.device,
):
    """
    Runs a single validation epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (MultiLoss): The loss function.
        device (torch.device): The device to run on.

    Returns:
        float: The average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    running_vloss = 0.0

    i = 0
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(loader, desc="Validation")):
            vimg, vmask = vdata
            vimg, vmask = vimg.to(device), vmask.to(device)

            output_mask = model(vimg)

            aff_loss = affinity_loss(output_mask)
            bce_loss = F.binary_cross_entropy_with_logits(output_mask, vmask)
            vloss = criterion(bce_loss, aff_loss)

            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss


def show_sample_prediction(img: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor):
    """
    Displays the input image, ground truth mask, and predicted mask.

    Args:
        img (torch.Tensor): Input image tensor [1, H, W]
        mask (torch.Tensor): Ground truth mask tensor [1, H, W]
        pred (torch.Tensor): Predicted mask tensor [1, H, W] (logits)
    """
    img = img.cpu().numpy()[0]  # (H, W)
    mask = mask.cpu().numpy()[0]  # (H, W)
    pred = torch.sigmoid(pred).cpu().numpy()[0]  # (H, W)
    pred_binary = (pred > 0.5).astype(float)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].set_title("Input Image")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].set_title("Ground Truth Mask")
    axs[1].imshow(mask, cmap="gray")
    axs[1].axis("off")

    axs[2].set_title("Predicted Mask (Prob)")
    axs[2].imshow(pred, cmap="gray")
    axs[2].axis("off")

    axs[3].set_title("Predicted Mask (Binary)")
    axs[3].imshow(pred_binary, cmap="gray")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main script to run standalone validation and visualization.
    Loads configuration from a YAML file but requires
    the model_path to be specified via command line.
    """
    parser = argparse.ArgumentParser(description="Wall Segmentation Validation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wall_seg.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model .pth file to validate",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    print(f"Loading configuration from {args.config}...")
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print(f"Loading validation data from: {config['val_dest']}")
    try:
        val_dataset = WallSegDataset(config["val_dest"], "val_tensors")
        val_loader = DataLoader(
            val_dataset, batch_size=config["validation"]["batch_size"], shuffle=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run preprocess.py first.")
        return

    # --- Initialize Model ---
    model = init_model(config["hub_cache_dir"], device)

    # --- Load Trained Weights (from args) ---
    print(f"Loading model weights from {args.model_path}")
    try:
        # Handle DataParallel prefix if model was saved from it
        state_dict = torch.load(args.model_path, map_location=device)
        if next(iter(state_dict)).startswith("module."):
            # Model was saved with DataParallel, strip 'module.'
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try loading into module directly (if model is wrapped in DP)
        try:
            model.load_state_dict(state_dict)
        except:
            print("Could not load weights. Model architecture might be different.")
            return

    # Initialize criterion
    criterion = MultiLoss().to(device)

    # --- Run Full Validation ---
    print("Running full validation pass to get loss...")
    avg_vloss = run_validation(model, val_loader, criterion, device)
    print(f"Average Validation Loss: {avg_vloss:.4f}")

    # --- Show Samples ---
    num_samples = config["validation"]["num_samples"]
    print(f"Displaying {num_samples} sample predictions...")
    model.eval()
    with torch.no_grad():
        for i, (img, mask) in enumerate(val_loader):
            if i >= num_samples:
                break

            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            # Use index 0 of the batch
            show_sample_prediction(img[0], mask[0], pred[0])

    print("Validation script finished.")


if __name__ == "__main__":
    main()
