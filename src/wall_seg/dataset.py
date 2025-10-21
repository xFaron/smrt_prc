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
import warnings

warnings.filterwarnings(
    "ignore", message="libpng warning: iCCP: known incorrect sRGB profile"
)

class WallSegDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        img_folders: List[str],
        crop_size: int = 512,
        train: bool = True,
    ):
        self.data_folder = data_folder
        self.img_folders = img_folders
        self.crop_size = crop_size
        self.train = train

        self.transform = transforms.ToTensor()

        self.img_file_name = "F1_scaled.png"
        self.svg_file_name = "model.svg"

        self.tiles = []
        for img_path in tqdm(img_folders):
            img_path = os.path.join(self.data_folder, img_path[1:], self.img_file_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            # If the whole image is in one tile
            if h <= self.crop_size and w <= self.crop_size:
                self.tiles.append({"img_path": img_path, "x": 0, "y": 0})
                continue

            # Else, If the tile contains some content (We wont be padding stuff, to create blank tiles in summary)
            for y in range(0, h + 1 - crop_size, crop_size):
                for x in range(0, w + 1 - crop_size, crop_size):
                    self.tiles.append({"img_path": img_path, "x": x, "y": y})

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        img_path = tile["img_path"]

        original_img = cv2.imread(img_path)
        height, width, _ = original_img.shape

        # Img preprocessing
        img = self.__pre_process_img(original_img)

        # Getting respective mask
        house = House(
            os.path.join(os.path.dirname(img_path), self.svg_file_name),
            height,
            width,
        )
        mask = self.__get_mask(house)

        # Cutting out the respective tile
        x, y = tile["x"], tile["y"]

        # Pad right if needed
        if x + self.crop_size > width:
            extra_w = x + self.crop_size - width
            img = np.concatenate(
                [img, np.ones((height, extra_w), dtype=img.dtype)], axis=1
            )
            mask = np.concatenate(
                [mask, np.zeros((height, extra_w), dtype=mask.dtype)], axis=1
            )
            width += extra_w

        # Pad bottom if needed
        if y + self.crop_size > height:
            extra_h = y + self.crop_size - height
            img = np.concatenate(
                [img, np.ones((extra_h, width), dtype=img.dtype)], axis=0
            )
            mask = np.concatenate(
                [mask, np.zeros((extra_h, width), dtype=mask.dtype)], axis=0
            )
            height += extra_h

        # Crop (y first, then x)
        img_tile = img[y : y + self.crop_size, x : x + self.crop_size]
        mask_tile = mask[y : y + self.crop_size, x : x + self.crop_size]

        # return original_img, img, mask
        return self.transform(img_tile), self.transform(mask_tile)

    def __pre_process_img(self, original_img, bin_threshold=True):
        # Grayscaling (Color barely matters for wall segmentation)
        if original_img is None:
            raise ValueError("Invalid image")

        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Threshold
        if bin_threshold:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Cleanup with morphological opening and closing
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Edge enhancement
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        img = sharpened

        # Normalize
        img = img.astype(np.float32) / 255.0
        return img

    def __get_mask(self, house: House):
        # Took code from earlier
        if house is None:
            raise ValueError("Invalid house")

        wall_labels = torch.tensor(house.walls)
        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)

        distinct_wall_instance_ids = distinct_wall_instance_ids[1:]

        masks = (wall_instance_ids == distinct_wall_instance_ids[:, None, None]).to(
            dtype=torch.uint8
        )
        boxes = masks_to_boxes(masks)  # Converting the mask to box coordinates

        non_empty_indices = torch.where(box_area(boxes) > 0)
        final_masks = masks[non_empty_indices]
        final_boxes = boxes[non_empty_indices]

        labels = torch.ones((len(final_boxes),), dtype=torch.int64)
        for i in range(len(final_masks)):
            rows, cols = np.where(final_masks[i])
            labels[i] = wall_labels[rows[0], cols[0]]

        joint_mask = torch.sum(final_masks, dim=0)
        return joint_mask.numpy().astype(np.float32)
