import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FLYDataset(Dataset):
    def __init__(self, path_to_data, mode="training", cam=0, transform=None):
        self.cam = cam
        self.H = 480
        self.W = 960
        self.transform = transform

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

    def __getitem__(self, idx, transform=None):
        if idx >= len(self):
            raise LookupError("Invalid index")

        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]

        keypoints = torch.tensor(self.annotations[idx], dtype=torch.float32)  # [N, 2], (y,x) ∈ [0,1]

        # Vor-Filter (optional)
        pre_mask = (
            (keypoints[:, 0] > 0) & (keypoints[:, 0] < 1) &
            (keypoints[:, 1] > 0) & (keypoints[:, 1] < 1)
        )
        keypoints[~pre_mask] = -1.0

        sample = {
            "image": img_tensor,
            "keypoints": keypoints
        }

        transform_to_use = transform or self.transform
        if transform_to_use:
            sample = transform_to_use(sample)

        # Nach-Filter: Maskiere alles was out-of-bounds ist
        kps = sample["keypoints"]
        mask = (
            (kps[:, 0] >= 0) & (kps[:, 0] <= 1) &
            (kps[:, 1] >= 0) & (kps[:, 1] <= 1)
        )
        kps[~mask] = -1.0
        sample["keypoints"] = kps

        return sample["image"], sample["keypoints"]

    
    def __len__(self):
        # returning whole length of the dataset / number of images
        return len(self.img_paths)
    
    def __getvisual__(self, idx=0, transform=None):
        """
        Gibt das Bild und die gültigen Keypoints in Pixelkoordinaten zurück.
        Optional: wendet eine andere Transform an als im Dataset.
        Kein Plot!
        """
        # Hole das transformierte Sample (nutzt __getitem__)
        img, keypoints = self.__getitem__(idx, transform=transform)

        # Nur gültige Keypoints
        valid = (keypoints != -1).all(dim=-1)
        keypoints = keypoints[valid]

        # Normierte Koordinaten → Pixel
        x = keypoints[:, 1] * self.W  # x = Spalte
        y = keypoints[:, 0] * self.H  # y = Zeile

        return img.squeeze().numpy(), (x.numpy(), y.numpy())
