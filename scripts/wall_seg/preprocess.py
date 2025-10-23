from typing import List
from tqdm import tqdm
import warnings
import gc
import os
import yaml

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from floortrans.loaders.house import House
from torch.utils.data import Dataset
from torchvision.ops.boxes import masks_to_boxes, box_area
from torchvision import transforms
from PIL import Image

from dataset import *

warnings.filterwarnings(
    "ignore", message="libpng warning: iCCP: known incorrect sRGB profile"
)

class WallSegPreprocess:
    def __init__(
        self,
        data_folder: str,
        img_folders: List[str],
        destination_folder: str,
        tensor_name: str,
        crop_size: int = 512,
        overlap: int = 64,  # Overlap between tiles
        min_content_ratio: float = 0.1,  # Minimum ratio of non-white pixels to keep tile
    ):
        self.data_folder = data_folder
        self.img_folders = img_folders
        self.crop_size = crop_size
        self.overlap = overlap
        self.min_content_ratio = min_content_ratio
        self.destination_folder = destination_folder
        self.tensor_name = tensor_name

        self.transform = transforms.ToTensor()

        self.img_file_name = "F1_scaled.png"
        self.svg_file_name = "model.svg"

        self.processed = False
        self.processed_img_dir = os.path.join(self.destination_folder, "images")
        self.processed_mask_dir = os.path.join(self.destination_folder, "masks")
        self.processed_img_path = os.path.join(self.processed_img_dir, "img_{}.npy")
        self.processed_mask_path = os.path.join(self.processed_mask_dir, "mask_{}.npy")

        # Store list of image paths instead of tiles
        self.img_paths = []
        for img_folder in tqdm(img_folders, desc="Indexing images"):
            img_path = os.path.join(
                self.data_folder, img_folder[1:], self.img_file_name
            )
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def _is_tile_valid(self, img_tile, mask_tile):
        """
        Check if a tile contains enough content to be worth keeping.
        Returns True if the tile should be kept, False otherwise.
        """
        # Check mask first (faster) - if mask has any walls, keep it
        if np.any(mask_tile > 0):
            return True

        # Check image content - count non-white pixels (assuming white is padding/empty)
        # In normalized space, white is close to 1.0
        non_white_pixels = np.sum(img_tile < 0.95)  # Adjust threshold as needed
        total_pixels = img_tile.size
        content_ratio = non_white_pixels / total_pixels

        return content_ratio >= self.min_content_ratio

    def __getitem__(self, index):
        """
        Loads the image and mask
        """
        return np.load(self.processed_img_path.format(index)), np.load(
            self.processed_mask_path.format(index)
        )

    def __pre_process_case(self, index):
        """
        Returns all tiles for a single image as tensors.
        Returns: (img_tiles, mask_tiles) where each is a tensor of shape [N, C, H, W]
        """
        img_path = self.img_paths[index]

        # Read and preprocess the image once
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Invalid image at {img_path}")

        height, width, _ = original_img.shape

        # Preprocess image once
        img = self.__pre_process_img(original_img)

        # Get mask once
        house = House(
            os.path.join(os.path.dirname(img_path), self.svg_file_name),
            height,
            width,
        )
        mask = self.__get_mask(house)
        del house

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        img = sharpened

        # Generate tiles with overlap
        img_tiles = []
        mask_tiles = []

        # Calculate stride (step size between tiles)
        stride = self.crop_size - self.overlap

        # If the whole image fits in one tile
        if height <= self.crop_size and width <= self.crop_size:
            # Pad to crop_size if needed
            img_padded = np.ones((self.crop_size, self.crop_size), dtype=img.dtype)
            mask_padded = np.zeros((self.crop_size, self.crop_size), dtype=mask.dtype)

            img_padded[:height, :width] = img
            mask_padded[:height, :width] = mask

            img_tiles.append(img_padded[np.newaxis, ...])
            mask_tiles.append(mask_padded[np.newaxis, ...])
        else:
            # Generate overlapping tiles
            y_positions = list(range(0, height, stride))
            x_positions = list(range(0, width, stride))

            # Ensure we cover the entire image by adjusting last positions
            if y_positions[-1] + self.crop_size < height:
                y_positions.append(height - self.crop_size)
            if x_positions[-1] + self.crop_size < width:
                x_positions.append(width - self.crop_size)

            for y in y_positions:
                for x in x_positions:
                    # Initialize tile with padding
                    img_tile = np.ones(
                        (self.crop_size, self.crop_size), dtype=img.dtype
                    )
                    mask_tile = np.zeros(
                        (self.crop_size, self.crop_size), dtype=mask.dtype
                    )

                    # Calculate actual crop region
                    y_end = min(y + self.crop_size, height)
                    x_end = min(x + self.crop_size, width)

                    crop_h = y_end - y
                    crop_w = x_end - x

                    # Copy the actual content
                    img_tile[:crop_h, :crop_w] = img[y:y_end, x:x_end]
                    mask_tile[:crop_h, :crop_w] = mask[y:y_end, x:x_end]

                    # Only add tile if it contains sufficient content
                    if self._is_tile_valid(img_tile, mask_tile):
                        img_tiles.append(img_tile[np.newaxis, ...])
                        mask_tiles.append(mask_tile[np.newaxis, ...])

        # Stack all tiles into tensors
        if len(img_tiles) == 0:
            # Return empty tensors if no valid tiles found
            return torch.empty(0, 1, self.crop_size, self.crop_size), torch.empty(
                0, 1, self.crop_size, self.crop_size
            )

        # img_tiles_tensor = torch.stack(img_tiles)  # Shape: [N, C, H, W]
        # mask_tiles_tensor = torch.stack(mask_tiles)  # Shape: [N, C, H, W]

        return img_tiles, mask_tiles

    def __pre_process_img(self, original_img, bin_threshold=True):
        # Grayscaling (Color barely matters for wall segmentation)
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
        return joint_mask.numpy().astype(np.uint8)

    def pre_process_dataset(self):
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)
        if not os.path.exists(self.processed_mask_dir):
            os.makedirs(self.processed_mask_dir)

        for i in tqdm(range(len(self)), desc="Preprocessing images"):
            img, mask = self.__pre_process_case(i)
            np.save(self.processed_img_path.format(i), img)
            np.save(self.processed_mask_path.format(i), mask)
            gc.collect()

        self.processed = True

    def tensorify(self, batch_size: int = 128):
        """
        Loads all pre_processed images, and saves it in batches as tensors
        Result: destination_folder/tensor_name_batch_X.pt files
        """
        all_imgs = []
        all_masks = []
        batch_counter = 0

        if not self.processed:
            self.pre_process_dataset()
        gc.collect()

        for i in tqdm(range(len(self)), desc="Making Batches"):
            # Get all tiles for this image
            img_tiles, mask_tiles = self[i]
            # print("Image Shape: ", img_tiles.shape)
            # print("Mask Shape: ", mask_tiles.shape)

            # Add tiles to batch
            for j in range(len(img_tiles)):
                all_imgs.append(img_tiles[j])
                all_masks.append(mask_tiles[j])

                # Save batch when it reaches batch_size
                if len(all_imgs) >= batch_size:
                    img_tensor = np.stack(all_imgs[:batch_size])
                    mask_tensor = np.stack(all_masks[:batch_size])
                    # print("Batching")
                    # print("Image Shape: ", img_tensor.shape)
                    # print("Mask Shape: ", mask_tensor.shape)

                    save_path = os.path.join(
                        self.destination_folder,
                        f"{self.tensor_name}_batch_{batch_counter}.pt",
                    )
                    torch.save(
                        (
                            torch.from_numpy(img_tensor).float(),
                            torch.from_numpy(mask_tensor).float(),
                        ),
                        save_path,
                    )

                    # Remove saved items and increment counter
                    all_imgs = all_imgs[batch_size:]
                    all_masks = all_masks[batch_size:]
                    batch_counter += 1

        # Save any remaining tiles
        if len(all_imgs) > 0:
            img_tensor = np.stack(all_imgs)
            mask_tensor = np.stack(all_masks)
            # print("Batching")
            # print("Image Shape: ", img_tensor.shape)
            # print("Mask Shape: ", mask_tensor.shape)

            save_path = os.path.join(
                self.destination_folder, f"{self.tensor_name}_batch_{batch_counter}.pt"
            )
            torch.save(
                (
                    torch.from_numpy(img_tensor).float(),
                    torch.from_numpy(mask_tensor).float(),
                ),
                save_path,
            )


def process(dest, file_path, tensor_split):
    


if __name__ == "__main__":
import os
import yaml

def process(dest, file_path, tensor_split, count):
    # Placeholder for actual logic
    print(
        f"Processing:\n  dest={dest}\n  file={file_path}\n  tensor_split={tensor_split}\n  count={count}\n"
    )

if __name__ == "__main__":
    config = {}
    with open("wall_seg.yaml") as file:
        config = yaml.safe_load(file)

    # Check for data_folder
    data_folder = config.get("data_folder")
    if not data_folder or not os.path.exists(data_folder):
        raise FileNotFoundError(f"data_folder not found or invalid: {data_folder}")

    # Define the triplets
    sets = ["train", "val", "test"]

    for s in sets:
        dest = config.get(f"{s}_dest")
        file_path = config.get(f"{s}_file")
        tensor_split = config.get(f"{s}_tensor_split")
        count = config.get(f"{s}_count", None)  # Optional field

        # Validate presence of required values
        if not all([dest, file_path, tensor_split]):
            raise ValueError(
                f"Incomplete configuration for '{s}': "
                f"dest={dest}, file={file_path}, tensor_split={tensor_split}"
            )

        # Run process with count
        process(dest, file_path, tensor_split, count)
