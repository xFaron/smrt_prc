import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes, box_area
from floortrans.loaders.house import House
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class SimpleSVGLoader(Dataset):
    def __init__(self, data_folder, data_file):
        self.image_file_name = "/F1_scaled.png"
        self.org_image_file_name = "/F1_original.png"
        self.svg_file_name = "/model.svg"

        self.data_folder = data_folder
        # Loading txt file
        text_file_path = os.path.join(data_folder, data_file)
        self.folders = np.genfromtxt(text_file_path, dtype="str")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        target = self.get_data(index)
        return target

    def get_data(self, index): 
        folder = self.folders[index]

        with Image.open(
            self.data_folder + folder + self.image_file_name
        ) as img:
            width, height = img.size

        house = House(
            self.data_folder + folder + self.svg_file_name, height, width
        )

        wall_instance_ids = torch.tensor(house.wall_ids)
        distinct_wall_instance_ids = torch.unique(wall_instance_ids)
        distinct_wall_instance_ids = distinct_wall_instance_ids[1:]

        masks = (wall_instance_ids == distinct_wall_instance_ids[:, None, None]).to(
            dtype=torch.uint8
        )
        boxes = masks_to_boxes(masks)  # Converting the mask to box coordinates

        non_empty_indices = torch.where(box_area(boxes) > 0)
        final_boxes = boxes[non_empty_indices]

        target = {}
        target["boxes"] = final_boxes
        target["meta"] = {
            "folder_name": conv_folder_name(folder),
            "path": self.data_folder + folder + self.image_file_name,
            "height": height,
            "width": width,
        }

        return target


def conv_folder_name(folder):
    return (os.path.dirname(folder)).replace("/", "_")


def __are_groupable_boxes(box1, box2, padding):
    coords = [
        (box1[0], box1[1]),
        (box1[2], box1[3]),
        (box1[0], box1[3]),
        (box1[2], box1[1]),
    ]
    for coord in coords:
        x, y = coord
        if (x <= box2[2] + padding[0] and x >= box2[0] - padding[0]) and (
            y <= box2[3] + padding[1] and y >= box2[1] - padding[1]
        ):
            return True


def are_groupable_boxes(box1, box2, padding=(10, 10)):
    result = __are_groupable_boxes(box1, box2, padding)
    return result or __are_groupable_boxes(box2, box1, padding)


def group_boxes(box1, box2):
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    ]


# TODO: Right now, this function does not do anything. Future
def include_doors(boundary_boxes, doors):
    return boundary_boxes


def preprocess_data(dataset : SimpleSVGLoader, img_folder : str, label_folder : str, count : int = None) -> None:
    # Make required folders
    req_folders = [img_folder, label_folder]
    for folder in req_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for i in tqdm(range(len(dataset)), total = len(dataset) if count is None else count):
        folder_name = conv_folder_name(dataset.folders[i])
        if os.path.exists(label_folder + f"/{folder_name}.txt"):
            continue

        if (not count is None) and (i > count):
            break

        target = dataset.get_data(i)

        target_boxes = target["boxes"]
        width = target['meta']['width']
        height = target["meta"]["height"]
        boundary_boxes = []

        last_len = 0
        while True:
            for box in target_boxes:
                for j, boundary_box in enumerate(boundary_boxes):
                    if are_groupable_boxes(boundary_box, box):
                        boundary_boxes[j] = group_boxes(boundary_box, box)
                        break
                else:
                    boundary_boxes.append(box)
            target_boxes = boundary_boxes
            if last_len == len(boundary_boxes):
                break

            last_len = len(target_boxes)
            boundary_boxes = []

        # TODO
        boundary_boxes_with_doors = include_doors(boundary_boxes, [])

        with open(label_folder + f"/{target['meta']['folder_name']}.txt", 'w') as f:
            for box in boundary_boxes_with_doors:
                x_center = ((box[0] + box[2]) / 2) / width
                y_center = ((box[1] + box[3]) / 2) / height
                box_width = (box[2] - box[0]) / width
                box_height = (box[3] - box[1]) / height

                f.write(f"0 {x_center:.3f} {y_center:.3f} {box_width:.3f} {box_height:.3f}\n")

        os.system(
            f"cp {target["meta"]["path"]} {img_folder + f"/{target['meta']['folder_name']}.png"}"
        )


train_file = "train.txt"
val_file = "val.txt"
data_folder = "./data/external/CubiCasa5k/data/cubicasa5k"

img_train = "./data/processed/roi/images/train"
img_val = "./data/processed/roi/images/val"
label_train = "./data/processed/roi/labels/train"
label_val = "./data/processed/roi/labels/val"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=int,
        help="Enter the amount of training images to process (default = full)",
        default=None,
    )
    parser.add_argument(
        "--val",
        type=int,
        help="Enter the amount of validation images to process (default = full)",
        default=None,
    )
    parser.add_argument(
        "--remove",
        type=int,
        help="All images will be removed which were preprocessed earlier",
        default=False,
    )

    args = parser.parse_args()

    if (args.remove):
        os.remove()

    train_dataset = SimpleSVGLoader(data_folder, train_file)
    val_dataset = SimpleSVGLoader(data_folder, val_file)

    print("Starting preprocessing of Training data")
    preprocess_data(train_dataset, img_train, label_train, args.train)

    print("Starting preprocessing of Validation data")
    preprocess_data(val_dataset, img_val, label_val, args.val)

    print("Done!")
