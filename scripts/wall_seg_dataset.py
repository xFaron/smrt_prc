"""
dataset.py

Defines the PyTorch Dataset for loading pre-processed wall segmentation tensors.
"""

import os
import torch
from torch.utils.data import Dataset


class WallSegDataset(Dataset):
    """
    A PyTorch Dataset to load pre-processed and batched tensor files.
    Each .pt file contains a batch of (images, masks). This dataset
    loads them and provides access to individual items (tiles).
    """

    def __init__(self, tensor_folder: str, tensor_name_split: str):
        """
        Initializes the dataset.

        Args:
            tensor_folder (str): Folder containing the .pt files.
            tensor_name_split (str): The prefix of the tensor files (e.g., 'train_tensors').
        """
        self.tensor_folder = tensor_folder
        self.tensor_files = sorted(
            [
                os.path.join(tensor_folder, f)
                for f in os.listdir(tensor_folder)
                if f.startswith(tensor_name_split) and f.endswith(".pt")
            ]
        )
        if len(self.tensor_files) == 0:
            raise RuntimeError(
                f"No tensor files found for split='{tensor_name_split}' in {tensor_folder}"
            )

        self.file_offsets = []
        self.total_length = 0
        self.loaded_data = {}  # Cache for loaded tensor files

        print(f"Indexing tensor files from {tensor_folder}...")
        # Compute lengths and create a map of indices to files
        for fpath in self.tensor_files:
            # We load the file here to get its length
            try:
                imgs, masks = torch.load(fpath, map_location="cpu")
                count = imgs.shape[0]
                self.file_offsets.append(
                    (fpath, self.total_length, self.total_length + count)
                )
                self.loaded_data[fpath] = (imgs, masks)  # Cache data
                self.total_length += count
            except Exception as e:
                print(f"Warning: Could not load tensor file {fpath}. Error: {e}")

        print(
            f"Found {self.total_length} total samples in {len(self.tensor_files)} files."
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        Retrieves an image and mask tile by index.
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")

        # Find which file this index belongs to
        for fpath, start, end in self.file_offsets:
            if start <= idx < end:
                # Calculate index within the batch
                local_idx = idx - start

                # Data is already cached from __init__
                imgs, masks = self.loaded_data[fpath]
                img, mask = imgs[local_idx], masks[local_idx]

                return img, mask

        raise IndexError(f"Index {idx} not found in any file offset.")


if __name__ == "__main__":
    # Example usage (assumes data is processed)
    print("Running dataset example...")
    try:
        train_dest = "./data/processed/wall_seg/train"
        train_dataset = WallSegDataset(train_dest, "train_tensors")
        print(f"Total training samples: {len(train_dataset)}")

        if len(train_dataset) > 0:
            img, mask = train_dataset[0]
            print(f"Sample 0 Image Shape: {img.shape}")
            print(f"Sample 0 Mask Shape: {mask.shape}")

    except Exception as e:
        print(f"Could not load dataset. Have you run preprocess.py? Error: {e}")
