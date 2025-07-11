import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FLY_Resnet(nn.Module):
    """
    Fly keypoint regressor using a pretrained ResNet50 backbone.

    This module loads a ResNet50 pretrained on ImageNet (IMAGENET1K_V2 weights)
    and replaces its final fully-connected layer with a two-stage head that
    maps the 2048-dim feature vector to `num_joints * 2` outputs, corresponding
    to the (x, y) coordinates of each keypoint.

    Args:
        num_joints (int): Number of keypoints to predict per image. The network
            will output a tensor of shape (batch_size, num_joints, 2).

    Architecture:
        - Backbone: torchvision.models.resnet50(pretrained on IMAGENET1K_V2)
        - Head:
            - Linear(2048 → 512)
            - ReLU
            - Linear(512 → num_joints * 2)

    Forward:
        x (Tensor): Input image batch of shape (B, 3, H, W).
        returns (Tensor): Predicted keypoint coordinates of shape
        (B, num_joints, 2).
    """
    def __init__(self, num_joints=38):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 2)
        )
        self.num_joints = num_joints

    def forward(self, x):
        out = self.backbone(x)
        return out.view(-1, self.num_joints, 2)