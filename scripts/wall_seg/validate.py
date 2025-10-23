import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from dataset import *
from model import *

data_folder = "/home/xfaron/Desktop/Code/Playground/test_construction/data/external/CubiCasa5k/data/cubicasa5k"
test_list_file = "/home/xfaron/Desktop/Code/Playground/test_construction/data/external/CubiCasa5k/data/cubicasa5k/test.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Validation Metrics Function
# -----------------------------
def compute_metrics(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    intersection = (preds_flat * masks_flat).sum()
    union = preds_flat.sum() + masks_flat.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (preds_flat.sum() + masks_flat.sum() + 1e-7)

    tp = intersection
    fp = preds_flat.sum() - tp
    fn = masks_flat.sum() - tp

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    f1 = (2 * precision * recall + 1e-7) / (precision + recall + 1e-7)

    return {
        "IoU": iou.item(),
        "Dice": dice.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item(),
    }


# -----------------------------
# Validation Loop
# -----------------------------
def validate_model(model, test_dataset, device="cuda", batch_size=1, threshold=0.5):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    iou_list, dice_list, precision_list, recall_list, f1_list = [], [], [], [], []

    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc="Validating"):
            img, mask = img.to(device), mask.to(device).float()
            preds = model(img)

            metrics = compute_metrics(preds, mask, threshold)
            iou_list.append(metrics["IoU"])
            dice_list.append(metrics["Dice"])
            precision_list.append(metrics["Precision"])
            recall_list.append(metrics["Recall"])
            f1_list.append(metrics["F1"])

    print("\n--- Validation Results ---")
    print(f"IoU:       {np.mean(iou_list):.4f}")
    print(f"Dice:      {np.mean(dice_list):.4f}")
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall:    {np.mean(recall_list):.4f}")
    print(f"F1 Score:  {np.mean(f1_list):.4f}")

    return {
        "IoU": np.mean(iou_list),
        "Dice": np.mean(dice_list),
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "F1": np.mean(f1_list),
    }


if __name__ == "__main__":
    from dataset import WallSegDataset
    from model import FPN

    with open(test_list_file) as f:
        test_folders = f.read().split("\n")

    test_dataset = WallSegDataset(data_folder, test_folders)
    model = FPN().to(device)
    model.load_state_dict(torch.load("/path/to/best_model.pth", map_location=device))

    results = validate_model(model, test_dataset, device=device, batch_size=1)
