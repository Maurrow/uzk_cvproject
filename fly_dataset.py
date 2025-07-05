import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize
from torchvision.models import ResNet50_Weights

class FLY_Dataset(Dataset):
    def __init__(self, path_to_data, mode="training", cam=0, backbone="resnet"):
        self.cam = cam
        self.H = 480
        self.W = 960
        self.backbone = backbone

        self.img_paths = []
        self.annotations = []

        if mode not in {"training", "test"}:
            raise ValueError("mode must be 'training' or 'test'")

        full_path = os.path.join(path_to_data, mode, f"cam{cam}")
        if not os.path.isdir(full_path):
            raise FileExistsError(f"Wrong path {full_path}")

        annotation_path = os.path.join(full_path, "annotations", "annotations.npz")
        image_path = os.path.join(full_path, "images")

        if not os.path.isfile(annotation_path):
            raise FileExistsError(f"Wrong annotation path {annotation_path}")
        if not os.path.isdir(image_path):
            raise FileExistsError(f"Wrong image path {image_path}")

        self.annotations = np.load(annotation_path)["points2d"]

        for image_name in sorted(os.listdir(image_path)):
            if image_name.endswith(".jpg"):
                self.img_paths.append(os.path.join(image_path, image_name))

        if len(self.img_paths) != len(self.annotations):
            raise IndexError("Number of images and annotations must be the same")

    def __getitem__(self, idx):

        if idx >= len(self):
            raise LookupError("Invalid index")

        # Load image and process to be usable by models
        img_tensor = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [H, W] -> [1, H, W] needed for training
        if self.backbone == "resnet":
            img_tensor = img_tensor.expand(3, -1, -1)  # 3 channels needed for ResNet

        # Load keypoints and create a visibility mask, [32 (T/F)], of which keypoints are visible in ground_truth
        # invisible keypoints are located at (0,0)
        keypts = torch.tensor(self.annotations[idx], dtype=torch.float32)
        visible = ~torch.all(keypts == 0.0, dim=1)
        keypts[~visible] = -1.0

        return img_tensor, keypts, visible

    
    def __len__(self):
        # returning whole length of the dataset / number of images
        return len(self.img_paths)