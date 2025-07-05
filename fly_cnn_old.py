import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CNN_Fly(nn.Module):
    def __init__(self, input_size, embedding_size, num_joints=38):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_joints = num_joints
        self.input_size = input_size  # expected: int or (H, W)

        """Encoder"""
        self.e1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.e2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.e3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.e4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

        """Bottleneck"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size) if isinstance(input_size, tuple) else torch.zeros(1, 1, input_size, input_size)
            dummy_out = self.encoder(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]
            
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.flatten_dim, 2 * embedding_size)

        """Decoder"""
        self.d1 = nn.Linear(embedding_size, self.flatten_dim)
        self.d2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.d3 = nn.ConvTranspose2d(64, 64, 2, stride=2, output_padding=0)
        self.d4 = nn.ConvTranspose2d(64, 32, 2, stride=2, output_padding=0)
        self.d5 = nn.ConvTranspose2d(32, 1, 3, padding=1)
        self.last_activation = nn.Sigmoid()

        """Keypoint regression head"""
        self.kp_head = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_joints * 2)
        )

    def encoder(self, x):
        x = self.activation(self.e1(x))
        x = self.activation(self.e2(x))
        x = self.activation(self.e3(x))
        x = self.activation(self.e4(x))
        return x

    def decode_image(self, z):
        x = self.d1(z)
        x = x.view(-1, 64, int(self.input_size[0] / 4), int(self.input_size[1] / 4))  # adjust based on encoder strides
        x = self.activation(self.d2(x))
        x = self.activation(self.d3(x))
        x = self.activation(self.d4(x))
        x = self.d5(x)
        return self.last_activation(x)


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # Encode
        enc = self.encoder(x)
        flat = self.flatten(enc)
        z_params = self.linear(flat)
        mu, logvar = torch.chunk(z_params, 2, dim=1)
        z = self.reparametrize(mu, logvar)
        kp_out = self.kp_head(z)
        #img_out = self.decode_image(z)

        return kp_out.view(-1, self.num_joints, 2), z, mu, logvar
        #return img_out, z, mu, logvar