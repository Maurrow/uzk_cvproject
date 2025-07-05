import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FLY_Resnet(nn.Module):
    def __init__(self, num_joints=32):
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