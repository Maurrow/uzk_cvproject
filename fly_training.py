import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2

from fly_dataset import FLY_Dataset
from fly_resnet import FLY_Resnet 

# Training Parameter
path_to_data   = "/scratch/cv-course2025/group2/data"
batch_size     = 16
num_epochs     = 100
lr             = 1e-4
cam            = 0
loss_threshold = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset + DataLoader
transforms = v2.Compose([
    v2.RandomRotation(degrees=15),                  # Rotate image randomly between -15 and +15 degrees
    v2.ToDtype(torch.float32, scale=True),          # Convert image to float32 and rescale pixel values from [0,255] → [0,1]
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225) # ResNet50
    )])
train_ds = FLY_Dataset(path_to_data=path_to_data, mode="training", cam=cam, backbone="resnet")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Model & optimizer
model = FLY_Resnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

prev_avg_loss = None
real_epochs   = None
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for images, keypts, visible in loop:
        # Move to device
        images  = images.to(device)      # [B,3,H,W]
        keypts  = keypts.to(device)      # [B,J,2]
        visible = visible.to(device)     # [B,J]

        # Apply random rotation per sample
        for i in range(images.size(0)):
            sample = {"image": images[i], "keypoints": keypts[i]}
            sample = transforms(sample)
            images[i] = sample["image"]
            keypts[i] = sample["keypoints"]

        # Forward
        preds = model(images)            # [B,J,2]

        # ---- Un-normalized, fully explicit MSE over visible points ----
        diff      = preds - keypts                       # [B,J,2]
        sq_err    = diff.pow(2).sum(dim=2)               # [B,J]  (x² + y² per joint)
        mask      = visible.float()                      # [B,J]
        masked_se = sq_err * mask                        # zero out invisible
        loss      = masked_se.sum()                      # raw sum over batch and joints

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(batch_loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)  # just for logging
    print(f"Epoch {epoch:02d} — total loss: {epoch_loss:.4f}, avg batch loss: {avg_loss:.4f}")

    # —— early exit condition ——  
    if prev_avg_loss is not None and (prev_avg_loss - avg_loss) < loss_threshold:
        print(f"Stopping training: avg_loss worsened from ({prev_avg_loss:.4f} → {avg_loss:.4f})")
        real_epochs = epoch
        break

    prev_avg_loss = avg_loss

# Get current timestamp to have a unique identifier for the model
timestr = time.strftime("%Y%m%d-%H%M%S")
# Save the trained model
torch.save(model.state_dict(), os.path.join('.', f'deep-fly-model-resnet50_{timestr}_{real_epochs:02d}epochs.pt'))

# Pre-Trained ResNet50 Model (ImageNet) to have a better starting point 


# model.load_state_dict(torch.load('/scratch/cv-course2025/group2/uzk_cvproject/fly-test-resnet50_nice.pt'))
# model.to(device)

#model.load_state_dict(torch.load("./fly-test.pt"))
#model.to(device)
#visualize_predictions(m, test_dataset, device=device)
