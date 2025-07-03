import math
import random
import torch
from torchvision.transforms import v2

class FlyTransformer:
    def __init__(self, degrees=(-15, 15)):
        self.degrees = degrees  # tuple oder einzelwert

    def __call__(self, sample, angle_deg=None):
        img = sample["image"]
        kps = sample["keypoints"]
        _, H, W = img.shape

        # Winkel bestimmen
        if angle_deg is None:
            if isinstance(self.degrees, (tuple, list)):
                angle_deg = random.uniform(*self.degrees)
            else:
                angle_deg = float(self.degrees)

        # Bild rotieren
        rot = v2.RandomRotation(degrees=(angle_deg, angle_deg), expand=False)
        img_rot = rot(img)

        # Keypoints transformieren
        center = torch.tensor([W / 2, H / 2])
        x = kps[:, 1] * W
        y = kps[:, 0] * H

        valid = (kps != -1).all(dim=1)
        x_valid = x[valid]
        y_valid = y[valid]

        coords = torch.stack([x_valid, y_valid], dim=1)
        rel_coords = coords - center

        angle_rad = math.radians(-angle_deg)
        R = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ])
        rotated = rel_coords @ R.T + center

        x[valid] = rotated[:, 0]
        y[valid] = rotated[:, 1]

        # Zurück zu normierten Koordinaten
        x = x / W
        y = y / H
        kps_rot = torch.stack([y, x], dim=1)

        sample["image"] = img_rot
        sample["keypoints"] = kps_rot
        sample["angle"] = angle_deg  # optional für Debug

        return sample