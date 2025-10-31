"""
preprocess.py

Handles data preprocessing for wall segmentation.
Converts raw images and SVG files into processed, tiled, and batched tensors
ready for training.
"""

import os
import cv2
import numpy as np
import torch
import gc
import argparse
import warnings
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops.boxes import masks_to_boxes, box_area
import yaml  

# Assuming 'floortrans' is an installed package or in the PYTHONPATH
# This dependency was identified from the notebook
try:
    from floortrans.loaders.house import House
except ImportError:
    print("Warning: 'floortrans.loaders.house.House' not found.")
    print("Preprocessing will fail. Please ensure this dependency is installed.")
    # Define a placeholder if not found, to allow module import
    House = None

warnings.filterwarnings(
    "ignore", message="libpng warning: iCCP: known incorrect sRGB profile"
)


class WallSegPreprocess:
    """
    Preprocesses raw floor plan images and SVG files into tiled, filtered
    numpy arrays, and then batches them into tensor files (.pt).
    """

    def __init__(
        self,
        data_folder: str,
        img_folders: List[str],
        destination_folder: str,
        tensor_name: str,
        crop_size: int = 512,
        overlap: int = 64,
        min_content_ratio: float = 0.1,
        batch_size: int = 128,
    ):
        """
        Initializes the preprocessor.

        Args:
            data_folder (str): Path to the root data directory (e.g., '.../cubicasa5k').
            img_folders (List[str]): List of folder names (e.g., ['/high_res/1', ...]).
            destination_folder (str): Where to save processed files.
            tensor_name (str): Prefix for saved tensor batch files (e.g., 'train_tensors').
            crop_size (int): Size of the square tiles (H and W).
            overlap (int): Overlap between adjacent tiles.
            min_content_ratio (float): Minimum ratio of non-white pixels to keep a tile.
            batch_size (int): Number of tiles to save in each .pt file.
        """
        if House is None:
            raise ImportError(
                "House dependency from 'floortrans' is required but not found."
            )

        self.data_folder = data_folder
        self.img_folders = img_folders
        self.crop_size = crop_size
        self.overlap = overlap
        self.min_content_ratio = min_content_ratio
        self.destination_folder = destination_folder
        self.tensor_name = tensor_name
        self.batch_size = batch_size

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
            if not img_folder:  # Skip empty lines
                continue
            img_path = os.path.join(
                self.data_folder, img_folder.lstrip("/"), self.img_file_name
            )
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
            else:
                print(f"Warning: Image path not found, skipping: {img_path}")

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
        """Loads the pre-processed numpy array tiles for a given image index."""
        return np.load(self.processed_img_path.format(index)), np.load(
            self.processed_mask_path.format(index)
        )

    def __pre_process_case(self, index):
        """
        Processes a single floor plan image and its SVG.
        Generates and returns all valid tiles.

        Returns: (img_tiles, mask_tiles) where each is a list of [1, H, W] numpy arrays.
        """
        img_path = self.img_paths[index]

        # Read and preprocess the image once
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Invalid image at {img_path}, skipping.")
            return [], []

        height, width, _ = original_img.shape

        # Preprocess image once
        img = self.__pre_process_img(original_img)

        # Get mask once
        svg_path = os.path.join(os.path.dirname(img_path), self.svg_file_name)
        if not os.path.exists(svg_path):
            print(f"Warning: SVG file not found at {svg_path}, skipping.")
            return [], []

        house = House(svg_path, height, width)
        mask = self.__get_mask(house)
        del house

        # Generate tiles with overlap
        img_tiles = []
        mask_tiles = []
        stride = self.crop_size - self.overlap

        # If the whole image fits in one tile
        if height <= self.crop_size and width <= self.crop_size:
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
                    img_tile = np.ones(
                        (self.crop_size, self.crop_size), dtype=img.dtype
                    )
                    mask_tile = np.zeros(
                        (self.crop_size, self.crop_size), dtype=mask.dtype
                    )

                    y_end = min(y + self.crop_size, height)
                    x_end = min(x + self.crop_size, width)
                    crop_h = y_end - y
                    crop_w = x_end - x

                    img_tile[:crop_h, :crop_w] = img[y:y_end, x:x_end]
                    mask_tile[:crop_h, :crop_w] = mask[y:y_end, x:x_end]

                    if self._is_tile_valid(img_tile, mask_tile):
                        img_tiles.append(img_tile[np.newaxis, ...])
                        mask_tiles.append(mask_tile[np.newaxis, ...])

        return img_tiles, mask_tiles

    def __pre_process_img(self, original_img, bin_threshold=True):
        """Applies grayscale, blur, threshold, and normalization to an image."""
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        if bin_threshold:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        img = sharpened

        img = img.astype(np.float32) / 255.0
        return img

    def __get_mask(self, house: House):
        """Extracts the binary wall mask from the House object."""
        if house is None:
            raise ValueError("Invalid house")

        wall_labels = torch.tensor(house.walls)
        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)

        # First ID is usually 0 (background), so skip it
        distinct_wall_instance_ids = distinct_wall_instance_ids[1:]

        if len(distinct_wall_instance_ids) == 0:
            # Return an empty mask if no walls are found
            return np.zeros((house.height, house.width), dtype=np.uint8)

        masks = (wall_instance_ids == distinct_wall_instance_ids[:, None, None]).to(
            dtype=torch.uint8
        )
        boxes = masks_to_boxes(masks)

        non_empty_indices = torch.where(box_area(boxes) > 0)[0]
        if len(non_empty_indices) == 0:
            return np.zeros((house.height, house.width), dtype=np.uint8)

        final_masks = masks[non_empty_indices]

        joint_mask = torch.sum(final_masks, dim=0)
        # Clamp values to 1 (binary mask)
        joint_mask[joint_mask > 1] = 1
        return joint_mask.numpy().astype(np.uint8)

    def pre_process_dataset_to_npy(self):
        """
        Runs the __pre_process_case for all images and saves the
        resulting tiles as .npy files (one file per original image).
        """
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)
        if not os.path.exists(self.processed_mask_dir):
            os.makedirs(self.processed_mask_dir)

        for i in tqdm(range(len(self)), desc="Preprocessing images to .npy"):
            img, mask = self.__pre_process_case(i)
            if len(img) > 0:
                img_stack = np.vstack(img)
                mask_stack = np.vstack(mask)
                np.save(self.processed_img_path.format(i), img_stack)
                np.save(self.processed_mask_path.format(i), mask_stack)
            else:
                # Save empty arrays to maintain index correspondence
                np.save(self.processed_img_path.format(i), np.array([]))
                np.save(self.processed_mask_path.format(i), np.array([]))
            gc.collect()

        self.processed = True

    def tensorify(self):
        """
        Loads all pre-processed .npy files and saves them in
        batched .pt tensor files.
        """
        all_imgs = []
        all_masks = []
        batch_counter = 0

        if not self.processed:
            self.pre_process_dataset_to_npy()
        gc.collect()

        print(f"Starting tensorification for {self.tensor_name}...")
        for i in tqdm(range(len(self)), desc="Making Batches"):
            # Get all tiles for this image
            try:
                img_tiles, mask_tiles = self[i]
                if img_tiles.size == 0:
                    continue  # Skip empty .npy files
            except FileNotFoundError:
                print(f"Warning: File not found for index {i}, skipping.")
                continue

            # Add tiles to batch
            for j in range(len(img_tiles)):
                all_imgs.append(img_tiles[j : j + 1])  # Keep shape [1, H, W]
                all_masks.append(mask_tiles[j : j + 1])

                # Save batch when it reaches batch_size
                if len(all_imgs) >= self.batch_size:
                    img_tensor = np.stack(all_imgs[: self.batch_size])
                    mask_tensor = np.stack(all_masks[: self.batch_size])

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
                    all_imgs = all_imgs[self.batch_size :]
                    all_masks = all_masks[self.batch_size :]
                    batch_counter += 1

        # Save any remaining tiles
        if len(all_imgs) > 0:
            img_tensor = np.stack(all_imgs)
            mask_tensor = np.stack(all_masks)

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
        print(f"Tensorification complete. Saved {batch_counter + 1} batch files.")


def main():
    """Main script to run preprocessing."""
    # Use argparse to get the config file path
    parser = argparse.ArgumentParser(description="Wall Segmentation Preprocessing")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wall_seg.yaml",  # Default path as requested
        help="Path to the configuration YAML file",
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

    # Loading folder names from paths specified in the config
    print("Loading file lists...")
    try:
        with open(config["train_file"]) as file:
            train_folders = file.read().split("\n")
        with open(config["val_file"]) as file:
            val_folders = file.read().split("\n")
    except FileNotFoundError as e:
        print(f"Error: Data list file not found. {e}")
        print("Please check 'train_file' and 'val_file' paths in your config.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in config file {args.config}")
        return

    print(f"Found {len(train_folders)} training files.")
    print(f"Found {len(val_folders)} validation files.")

    # Preprocessing and converting to tensors for Training
    print("\n--- Starting Training Set Preprocessing ---")
    train_pre_processer = WallSegPreprocess(
        config["data_folder"],
        train_folders,
        config["train_dest"],
        "train_tensors",
        crop_size=config["preprocessing"]["crop_size"],
        batch_size=config["preprocessing"]["batch_size"],
    )
    train_pre_processer.tensorify()

    # Preprocessing and converting to tensors for Validation
    print("\n--- Starting Validation Set Preprocessing ---")
    val_pre_processer = WallSegPreprocess(
        config["data_folder"],
        val_folders,
        config["val_dest"],
        "val_tensors",
        crop_size=config["crop_size"],
        batch_size=config["batch_size"],
    )
    val_pre_processer.tensorify()

    print("\n--- Preprocessing Complete ---")


if __name__ == "__main__":
    main()
