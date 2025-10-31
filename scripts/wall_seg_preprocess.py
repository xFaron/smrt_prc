"""
preprocess.py - Memory-Optimized Version

Key optimizations:
1. Streaming tile processing - never loads all tiles into memory
2. Dynamic memory monitoring and adaptive batching
3. Explicit memory cleanup after each image
4. Direct tensor saving without intermediate accumulation
5. Generator-based processing for lazy evaluation
"""

import os
import cv2
import numpy as np
import torch
import gc
import argparse
import warnings
import psutil
from typing import List, Tuple, Generator
from tqdm import tqdm
from torchvision.ops.boxes import masks_to_boxes, box_area
import yaml

try:
    from floortrans.loaders.house import House
except ImportError:
    print("Warning: 'floortrans.loaders.house.House' not found.")
    House = None

warnings.filterwarnings(
    "ignore", message="libpng warning: iCCP: known incorrect sRGB profile"
)


class MemoryMonitor:
    """Monitors system memory and provides adaptive batch sizing."""

    def __init__(self, safety_margin: float = 0.15):
        """
        Args:
            safety_margin: Fraction of memory to keep free (e.g., 0.15 = 15%)
        """
        self.safety_margin = safety_margin
        self.vm = psutil.virtual_memory()

    def get_available_memory_gb(self) -> float:
        """Returns available memory in GB."""
        self.vm = psutil.virtual_memory()
        return self.vm.available / (1024**3)

    def get_memory_usage_percent(self) -> float:
        """Returns current memory usage as percentage."""
        self.vm = psutil.virtual_memory()
        return self.vm.percent

    def should_reduce_batch_size(self, threshold: float = 85.0) -> bool:
        """Check if memory usage is too high."""
        return self.get_memory_usage_percent() > threshold

    def estimate_safe_batch_size(self, tile_size: int, base_batch_size: int) -> int:
        """
        Estimate safe batch size based on available memory.

        Args:
            tile_size: Size of each tile (e.g., 512)
            base_batch_size: Desired batch size

        Returns:
            Adjusted batch size that won't exceed memory
        """
        # Estimate memory per tile (image + mask, float32)
        bytes_per_tile = tile_size * tile_size * 2 * 4  # 2 arrays, 4 bytes per float32
        available_bytes = (
            self.get_available_memory_gb() * (1024**3) * (1 - self.safety_margin)
        )

        max_batch_size = int(available_bytes / bytes_per_tile)
        safe_batch_size = min(base_batch_size, max_batch_size, 256)  # Cap at 256

        return max(1, safe_batch_size)  # At least 1


class WallSegPreprocess:
    """
    Memory-optimized preprocessor for wall segmentation.

    Key changes from original:
    - Uses generators for tile streaming
    - Processes and saves tiles immediately (no accumulation)
    - Adaptive batch sizing based on available memory
    - Explicit memory cleanup after each image
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
        enable_adaptive_batching: bool = True,
    ):
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
        self.base_batch_size = batch_size
        self.enable_adaptive_batching = enable_adaptive_batching

        self.img_file_name = "F1_scaled.png"
        self.svg_file_name = "model.svg"

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()

        # Adaptive batch size
        if self.enable_adaptive_batching:
            self.batch_size = self.memory_monitor.estimate_safe_batch_size(
                self.crop_size, self.base_batch_size
            )
            print(
                f"Adaptive batch size: {self.batch_size} (requested: {self.base_batch_size})"
            )
        else:
            self.batch_size = self.base_batch_size

        # Index image paths without loading anything
        self.img_paths = []
        for img_folder in tqdm(img_folders, desc="Indexing images"):
            if not img_folder:
                continue
            img_path = os.path.join(
                self.data_folder, img_folder.lstrip("/"), self.img_file_name
            )
            if os.path.exists(img_path):
                self.img_paths.append(img_path)

        print(f"Indexed {len(self.img_paths)} valid image paths")
        print(
            f"Available memory: {self.memory_monitor.get_available_memory_gb():.2f} GB"
        )

    def __len__(self):
        return len(self.img_paths)

    def _is_tile_valid(self, img_tile: np.ndarray, mask_tile: np.ndarray) -> bool:
        """Check if tile contains enough content (optimized for speed)."""
        # Check mask first (fastest check)
        if np.any(mask_tile):
            return True

        # Fast content check using mean instead of sum
        # White pixels have value close to 1.0
        mean_value = np.mean(img_tile)
        return mean_value < 0.9  # If mean < 0.9, enough dark content exists

    def _generate_tiles(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator that yields valid tiles one at a time.
        CRITICAL: This prevents loading all tiles into memory at once.

        Yields:
            Tuple of (img_tile, mask_tile), each with shape [1, H, W]
        """
        height, width = img.shape
        stride = self.crop_size - self.overlap

        # Handle small images that fit in one tile
        if height <= self.crop_size and width <= self.crop_size:
            img_padded = np.ones((self.crop_size, self.crop_size), dtype=img.dtype)
            mask_padded = np.zeros((self.crop_size, self.crop_size), dtype=mask.dtype)

            img_padded[:height, :width] = img
            mask_padded[:height, :width] = mask

            if self._is_tile_valid(img_padded, mask_padded):
                yield img_padded[np.newaxis, ...], mask_padded[np.newaxis, ...]
            return

        # Generate overlapping tiles
        y_positions = list(range(0, height, stride))
        x_positions = list(range(0, width, stride))

        # Ensure full coverage
        if y_positions[-1] + self.crop_size < height:
            y_positions.append(height - self.crop_size)
        if x_positions[-1] + self.crop_size < width:
            x_positions.append(width - self.crop_size)

        for y in y_positions:
            for x in x_positions:
                # Create tiles with pre-allocated arrays
                img_tile = np.ones((self.crop_size, self.crop_size), dtype=img.dtype)
                mask_tile = np.zeros((self.crop_size, self.crop_size), dtype=mask.dtype)

                y_end = min(y + self.crop_size, height)
                x_end = min(x + self.crop_size, width)
                crop_h = y_end - y
                crop_w = x_end - x

                # Copy data using slicing (efficient)
                img_tile[:crop_h, :crop_w] = img[y:y_end, x:x_end]
                mask_tile[:crop_h, :crop_w] = mask[y:y_end, x:x_end]

                if self._is_tile_valid(img_tile, mask_tile):
                    yield img_tile[np.newaxis, ...], mask_tile[np.newaxis, ...]

    def _process_single_image(
        self, img_path: str
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Process a single image and yield tiles without storing them.

        MEMORY OPTIMIZATION: Uses generator pattern to yield tiles one at a time,
        allowing them to be written to disk immediately without accumulation.
        """
        # Read image
        original_img = cv2.imread(img_path)
        if original_img is None:
            return

        height, width, _ = original_img.shape

        # Preprocess image
        img = self._preprocess_img(original_img)
        del original_img  # Explicit cleanup
        gc.collect()

        # Get mask
        svg_path = os.path.join(os.path.dirname(img_path), self.svg_file_name)
        if not os.path.exists(svg_path):
            return

        house = House(svg_path, height, width)
        mask = self._get_mask(house)
        del house  # Explicit cleanup
        gc.collect()

        # Generate and yield tiles
        yield from self._generate_tiles(img, mask)

        # Cleanup after processing this image
        del img, mask
        gc.collect()

    def _preprocess_img(
        self, original_img: np.ndarray, bin_threshold: bool = True
    ) -> np.ndarray:
        """Applies grayscale, blur, threshold, and normalization."""
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        if bin_threshold:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        # Convert to float32 and normalize
        return sharpened.astype(np.float32) / 255.0

    def _get_mask(self, house: House) -> np.ndarray:
        """Extract binary wall mask from House object."""
        if house is None:
            raise ValueError("Invalid house")

        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)[
            1:
        ]  # Skip background

        if len(distinct_wall_instance_ids) == 0:
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
        joint_mask[joint_mask > 1] = 1

        return joint_mask.numpy().astype(np.uint8)

    def tensorify_streaming(self):
        """
        STREAMING TENSORIFICATION - Main memory optimization.

        Process images one at a time, accumulate tiles only up to batch_size,
        then immediately save and clear memory. Never stores all data at once.
        """
        os.makedirs(self.destination_folder, exist_ok=True)

        batch_imgs = []
        batch_masks = []
        batch_counter = 0
        total_tiles = 0

        print(f"Starting streaming tensorification for {self.tensor_name}...")
        print(
            f"Initial memory usage: {self.memory_monitor.get_memory_usage_percent():.1f}%"
        )

        # Process images one at a time
        for img_idx, img_path in enumerate(
            tqdm(self.img_paths, desc="Processing images")
        ):
            try:
                # Generate tiles for this image (streaming)
                for img_tile, mask_tile in self._process_single_image(img_path):
                    batch_imgs.append(img_tile)
                    batch_masks.append(mask_tile)
                    total_tiles += 1

                    # Save batch when full
                    if len(batch_imgs) >= self.batch_size:
                        self._save_batch(batch_imgs, batch_masks, batch_counter)
                        batch_counter += 1

                        # Clear batch lists
                        batch_imgs.clear()
                        batch_masks.clear()
                        gc.collect()

                        # Adaptive batching: reduce batch size if memory is high
                        if (
                            self.enable_adaptive_batching
                            and self.memory_monitor.should_reduce_batch_size()
                        ):
                            old_size = self.batch_size
                            self.batch_size = max(16, self.batch_size // 2)
                            print(
                                f"\nMemory pressure detected. Reducing batch size: {old_size} -> {self.batch_size}"
                            )

            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue

            # Periodic memory cleanup (every 10 images)
            if (img_idx + 1) % 10 == 0:
                gc.collect()

        # Save remaining tiles
        if batch_imgs:
            self._save_batch(batch_imgs, batch_masks, batch_counter)
            batch_counter += 1

        print(f"\n✓ Tensorification complete!")
        print(f"  Total tiles processed: {total_tiles}")
        print(f"  Batches saved: {batch_counter}")
        print(
            f"  Final memory usage: {self.memory_monitor.get_memory_usage_percent():.1f}%"
        )

    def _save_batch(
        self,
        batch_imgs: List[np.ndarray],
        batch_masks: List[np.ndarray],
        batch_counter: int,
    ):
        """
        Save a batch of tiles to disk.

        MEMORY OPTIMIZATION: Converts to tensor, saves immediately, then returns.
        The caller is responsible for clearing the input lists.
        """
        if not batch_imgs:
            return

        # Stack and convert to tensors
        img_tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).bool()
        mask_tensor = torch.from_numpy(np.stack(batch_masks, axis=0)).bool()

        # Save to disk
        save_path = os.path.join(
            self.destination_folder, f"{self.tensor_name}_batch_{batch_counter}.pt"
        )
        print(f"Shape of saved batch {batch_counter}: {img_tensor.shape}")
        print(f"Shape of saved mask batch {batch_counter}: {mask_tensor.shape}")
        torch.save((img_tensor, mask_tensor), save_path)

        # Explicit cleanup
        del img_tensor, mask_tensor


def main():
    """Main script with memory-optimized preprocessing."""
    parser = argparse.ArgumentParser(
        description="Memory-Optimized Wall Segmentation Preprocessing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wall_seg.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--disable-adaptive-batching",
        action="store_true",
        help="Disable adaptive batch sizing based on memory",
    )
    args = parser.parse_args()

    # Load configuration
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

    # Load file lists
    print("Loading file lists...")
    try:
        with open(config["train_file"]) as file:
            train_folders = [line for line in file.read().split("\n") if line.strip()]
        with open(config["val_file"]) as file:
            val_folders = [line for line in file.read().split("\n") if line.strip()]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading data lists: {e}")
        return

    print(f"Found {len(train_folders)} training files")
    print(f"Found {len(val_folders)} validation files")


    if config["preprocessing"]["preprocess_train"]:
        # Process training set
        print("\n" + "=" * 60)
        print("TRAINING SET PREPROCESSING")
        print("=" * 60)

        train_preprocessor = WallSegPreprocess(
            data_folder=config["data_folder"],
            img_folders=train_folders[: config["preprocessing"]["max_train_images"]],
            destination_folder=config["train_dest"],
            tensor_name="train_tensors",
            crop_size=config["preprocessing"]["crop_size"],
            overlap=config["preprocessing"].get("overlap", 64),
            min_content_ratio=config["preprocessing"].get("min_content_ratio", 0.1),
            batch_size=config["preprocessing"]["batch_size"],
            enable_adaptive_batching=not args.disable_adaptive_batching,
        )
        train_preprocessor.tensorify_streaming()

        # Cleanup before validation
        del train_preprocessor
        gc.collect()

    if config["preprocessing"]["preprocess_val"]:
        # Process validation set
        print("\n" + "=" * 60)
        print("VALIDATION SET PREPROCESSING")
        print("=" * 60)

        val_preprocessor = WallSegPreprocess(
            data_folder=config["data_folder"],
            img_folders=val_folders[: config["preprocessing"]["max_val_images"]],
            destination_folder=config["val_dest"],
            tensor_name="val_tensors",
            crop_size=config.get("crop_size", 512),
            overlap=config.get("overlap", 64),
            min_content_ratio=config.get("min_content_ratio", 0.1),
            batch_size=config.get("batch_size", 128),
            enable_adaptive_batching=not args.disable_adaptive_batching,
        )
        val_preprocessor.tensorify_streaming()

    print("\n" + "=" * 60)
    print("✓ ALL PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
