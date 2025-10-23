from typing import List, Tuple
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes, box_area
from floortrans.loaders.house import House
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings(
    "ignore", message="libpng warning: iCCP: known incorrect sRGB profile"
)

class WallSegDataset(Dataset):
    def __init__(self, tensor_folder: str, tensor_name_split: str):
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

        self.loaded_tensors = []
        self.file_offsets = []
        self.total_length = 0

        # Compute lengths (so __len__ works)
        for fpath in self.tensor_files:
            imgs, masks = torch.load(fpath, map_location="cpu")
            count = imgs.shape[0]
            self.file_offsets.append(
                (fpath, self.total_length, self.total_length + count)
            )
            self.total_length += count

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which file this index belongs to
        for fpath, start, end in self.file_offsets:
            if start <= idx < end:
                local_idx = idx - start
                imgs, masks = torch.load(fpath, map_location="cpu")

                img, mask = imgs[local_idx], masks[local_idx]

                return img, mask

        raise IndexError("Index out of range")
