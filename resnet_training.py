import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as F

class FLYDataset(Dataset):
    def __init__(self, path_to_data, mode="training", cam=0, transform=None):

        # save selected camera and init lists
        self.cam = cam
        self.img_paths = []
        self.annotations = []
        self.H = 480
        self.W = 960
        self.transform = transform

        # e.g path to the data, different classes, number of images per class, or image IDs per class
        if(mode != "test" and mode != "training"):
            raise ValueError("No such kind of data available")

        #Create path from path and parameter training/test
        full_path = os.path.join(path_to_data, mode, f"cam{cam}")
        # check if path exists. Path to data and kind are user inputs.
        if not os.path.isdir(full_path):
            raise FileExistsError(f"Wrong path {full_path}")

        # get the path to the annotation file and the path to the images
        annotation_file_path = os.path.join(full_path, "annotations", "annotations.npz")
        image_path = os.path.join(full_path, "images")
        # check if path exists. 
        if not os.path.isfile(annotation_file_path):
            raise FileExistsError(f"Wrong path to annotation file {annotation_file_path}")
        if not os.path.isdir(image_path):
            raise FileExistsError(f"Wrong path to image {image_path}")
        
        # load annotations
        annotations = np.load(annotation_file_path)
        self.annotations = annotations['points2d']

        for image_name in sorted(os.listdir(image_path)):
                # get the label as an int and the path of the image
                if(image_name.endswith(".jpg")):
                    self.img_paths.append(os.path.join(image_path, image_name))

        if len(self.img_paths) != len(self.annotations):
            raise IndexError("Number of images and annotations must be the same")


    def __getitem__(self, idx):
        # returning a single image per given index idx and its corresponding label

        #Check if the index exists and read in the image. Return both the image and the label as torch tensors
        if(idx >= len(self.img_paths)):
            raise LookupError("Invalid index for image")
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = np.stack([img]*3, axis=0)  # [3, H, W]
        
        #creating torch tensor and normalizing,a lso adding front channel to match requirement from torch
        t_img = torch.tensor(img, dtype=torch.float32) / 255
        # When training with ResNet50, this is not required
        #t_img = t_img.unsqueeze(0)

        # prepare annotations as tensor
        annotation = self.annotations[idx]
        t_anno = torch.tensor(annotation, dtype=torch.float32)
        if self.transform:
            sample = {
                "image": t_img,
                "keypoints": t_anno,
                "keypoints_format": "xy"
            }
            sample = self.transform(sample)
            t_img = sample["image"]
            t_anno = sample["keypoints"]

        return t_img, t_anno
    
    def __len__(self):
        # returning whole length of the dataset / number of images
        return len(self.img_paths)
    
    def __getvisual__(self, idx = 0):
        if(idx >= len(self.img_paths)):
            raise LookupError("Invalid index for image")
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        annotation = self.annotations[idx]
        print(f"class {len(self.annotations)}")
        annotation = (annotation[:,1] * self.W, annotation[:,0] * self.H)
        
        return img, annotation

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

class ResNet50Keypoints(nn.Module):
    def __init__(self, num_joints=38):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace final classification layer with regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 2)
        )

        self.num_joints = num_joints

    def forward(self, x):
        out = self.backbone(x)               # shape: [B, num_joints * 2]
        out = out.view(-1, self.num_joints, 2)  # shape: [B, J, 2]
        return out

# Train
def train_keypoint_model(model, dataset, num_epochs=10, batch_size=16, lr=1e-4, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for img, keypoints in loop:
            img       = img.to(device)           # [B, 1, H, W]
            keypoints = keypoints.to(device)     # [B, J, 2]

            preds = model(img)       # [B, J, 2]
            keypoints_px = keypoints.clone()
            keypoints_px[:, :, 0] *= dataset.W
            keypoints_px[:, :, 1] *= dataset.H

            preds_px = preds.clone()
            preds_px[:, :, 0] *= dataset.W
            preds_px[:, :, 1] *= dataset.H
            loss = loss_fn(preds_px, keypoints_px)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), os.path.join('.', f'fly-test-resnet50.pt'))

    return model


# Visualize model
def visualize_predictions(model, dataset, device="cuda", num_samples=5):
    model.eval()

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    axes = axes if num_samples > 1 else [axes]

    for i in range(num_samples):
        img, true_kp = dataset[i]
        # Since the model expects a batch, we unsqueeze to create a "batch" 
        # consisting of a single image.
        input_img = img.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Permute to get from [C, H, W] to [H, W, 3]
        vis_img = img.permute(1, 2, 0).cpu().numpy() 

        # Run model
        with torch.no_grad():
            # Model returns a batch, meaning [B, J, 2], with squeezing 0 we 
            # end up with [J, 2].
            pred_kp = model(input_img).squeeze(0).cpu() 

        # Get H, W from the visualize image shape. We permuted earlier, 
        # meaning now we just cut of the last dim to end up with [H, W].
        H, W = vis_img.shape[:2]
        pred_px = pred_kp.clone()
        pred_px[:, 0] *= W
        pred_px[:, 1] *= H

        true_px = true_kp.clone()
        true_px[:, 0] *= W
        true_px[:, 1] *= H

        ax = axes[i]
        ax.imshow(vis_img)
        ax.scatter(pred_px[:, 0], pred_px[:, 1], c='r', label='Predicted', s=10)
        ax.scatter(true_px[:, 0], true_px[:, 1], c='g', label='GT', s=10, alpha=0.6)
        ax.set_title(f"Sample {i}")
        ax.axis("off")

    plt.tight_layout()
    axes[0].legend()
    plt.show()


# Pre-Trained ResNet50 Model (ImageNet) to have a better starting point 
model = ResNet50Keypoints()

transforms = v2.Compose([
    v2.RandomRotation(degrees=15),                  # Rotate image randomly between -15 and +15 degrees
    v2.ToDtype(torch.float32, scale=True),          # Convert image to float32 and rescale pixel values from [0,255] → [0,1]
    v2.Normalize(mean=(0.5,), std=(0.5,))           # Normalize image: (x - 0.5) / 0.5 → values now in [-1, 1]
])

train_dataset = FLYDataset("/scratch/cv-course2025/group2/data", transform=transforms)
test_dataset = FLYDataset("/scratch/cv-course2025/group2/data", mode="test")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use CPU for debug so you do not use too much GPU time
print(f"Used Device: {device}")
print("\nStarting training loop...\n")

m = train_keypoint_model(model, train_dataset, num_epochs=5)
#model.load_state_dict(torch.load("./fly-test.pt"))
#model.to(device)
#visualize_predictions(m, test_dataset, device=device)
