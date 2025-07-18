import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FLY_Dataset(Dataset):
    """
    A Dataset for fly pose estimation using 2D keypoints.

    This dataset handles loading grayscale images and corresponding 
    normalized 2D keypoint annotations. It supports preprocessing to 
    fit common backbones like ResNet (converting to 3-channel fake RGB image).

    Parameters:
    -----------
    path_to_data : str
        Root path to the dataset directory.

    mode : str
        Either "training" or "test" to select the data split.

    cam : int
        Camera ID used to select the subdirectory `cam{cam}`.

    backbone : str
        Backbone type, e.g. "resnet". If "resnet", converts grayscale images to 3-channel fake RGB.

    Returns:
    --------
    img_tensor : torch.Tensor
        The processed image tensor of shape [3, H, W] (if ResNet) or [1, H, W] (default).

    keypts : torch.Tensor
        Tensor of keypoint coordinates, shape [38, 2], values in [0, 1], or set to -1 for invisible points.

    visible : torch.Tensor
        Boolean mask of shape [38] indicating whether each keypoint is visible.
    """
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

        img_tensor = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # 'backbone' is legacy: we started with custom CNNs, then switched to
        # pretrained ResNet50s. This condition lets us support both modes, 
        # so we keep it for 'backward' compatibility.
        if self.backbone == "resnet":
            # 3 Channels to create a fake RGB Image, due to ResNet only 
            # accepting RGB Images / Images with 3 channels.
            img_tensor = img_tensor.expand(3, -1, -1)

        # Load your 38×2 array of keypoints
        keypts = torch.tensor(self.annotations[idx], dtype=torch.float32)

        # Mask joints exactly at (0,0)
        mask_zero = torch.all(keypts == 0.0, dim=1) 

        # Mask joints exactly at (0,1)
        mask_outside = (keypts[:, 0] == 0.0) & (keypts[:, 1] == 1.0)

        # Visible = everything that is neither (0,0) nor (0,1)
        # These are the locations where DeepFly3D puts keypoints which are 
        # currently not on the image / not detected.
        visible = ~(mask_zero | mask_outside)

        keypts[~visible] = -1.0

        return img_tensor, keypts, visible

    
    def __len__(self):
        return len(self.img_paths)