import argparse
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import copy
import time
from tqdm import tqdm
from torchvision.transforms import v2

from fly_dataset import FLY_Dataset
from fly_resnet import FLY_Resnet 

parser = argparse.ArgumentParser(description="Train ResNet-based fly pose model")
parser.add_argument("--data", type=str, required=True, help="Path to the root data directory")
parser.add_argument("--cam", type=int, default=0, help="Camera ID to use (default: 0)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--lt", type=float, default=0.001, help="Loss threshold for early stopping")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args = parser.parse_args()

# Training Parameters
path_to_data = args.data
cam          = args.cam
num_epochs   = args.epochs
batch_size   = args.batch_size
lr           = args.lr
# Parameters for early Stopping
patience       = args.patience
loss_threshold = args.lt

# list for counting occurrences of keypoints
n_keypoints_used = np.zeros(38)
n_keyp_used_temp = np.zeros(38)
n_keypoints_epoch = np.zeros(38)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transforms, Dataset and Dataloader
transforms = v2.Compose([
    v2.RandomRotation(degrees=15),                  # Rotate image randomly between -15 and +15 degrees
    v2.ToDtype(torch.float32, scale=True),          # Convert image to float32 and rescale pixel values from [0,255] → [0,1]
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)                   # ResNet50 Normalizations (parameters from the ResNet50 IMAGENET1K_V2)
    )])
train_ds = FLY_Dataset(path_to_data=path_to_data, mode="training", cam=cam, backbone="resnet")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Model & optimizer
model = FLY_Resnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

prev_avg_loss = None
real_epochs   = 0
consec_worse  = 0
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for images, keypts, visible in loop:
        images  = images.to(device)
        keypts  = keypts.to(device)
        visible = visible.to(device)

        # Apply above defined transforms to both image and keypoints
        for i in range(images.size(0)):
            sample    = {"image": images[i], "keypoints": keypts[i]}
            sample    = transforms(sample)
            images[i] = sample["image"]
            keypts[i] = sample["keypoints"]

        # Forward
        preds = model(images)

        # MSE over visible points
        diff      = preds - keypts
        sq_err    = diff.pow(2).sum(dim=2)
        mask      = visible.float()
        masked_se = sq_err * mask   # zero out invisible
        loss      = masked_se.sum() # raw sum over batch and joints

        # For each image, find the keypoint with the highest squared error
        # and increment its counter. Over all epochs this highlights
        # which keypoints are most frequently the worst predicted
        for single_image in masked_se:
            n_keypoints_epoch[np.argmax(single_image.cpu().detach().numpy())] += 1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(batch_loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch:02d} — total loss: {epoch_loss:.4f}, avg batch loss: {avg_loss:.4f}")

    print(f"Keypoints with most square error per image this epoch: {n_keypoints_epoch}")
    # reset epoch keypoint count, add to temp
    n_keyp_used_temp  += n_keypoints_epoch
    n_keypoints_epoch *= 0

    # Check the performance of the current epoch against the last one
    is_worse = prev_avg_loss is not None and (prev_avg_loss - avg_loss) < loss_threshold

    if is_worse:
        if consec_worse == 0:
            # Save the model state if this is the first epoch which is 
            # performing worse than previous - Rollback state
            saved_state = copy.deepcopy(model.state_dict())

            # increase keypoint counter
            n_keypoints_used += n_keyp_used_temp
            n_keyp_used_temp *= 0
        # Increase the bad epoch counter, dont increase keypoint count yet
        consec_worse += 1
    else:
        # reset bad epoch counter, remove saved state
        consec_worse = 0
        saved_state  = None

        # increase keypoint count
        n_keypoints_used += n_keyp_used_temp
        n_keyp_used_temp *= 0

    prev_avg_loss = avg_loss

    if consec_worse >= patience:
        print(
            f"No improvement for {patience} consecutive epochs. "
            f"Rolling back to state from {patience} epochs ago."
        )
        model.load_state_dict(saved_state)
        epoch = epoch - patience
        break
print(f"Keypoints with most square error per image in entire training: {n_keypoints_used}")

# Get current timestamp to have a unique identifier for the model
timestr = time.strftime("%Y%m%d-%H%M%S")
# Save the trained model
torch.save(model.state_dict(), os.path.join('.', f'cam{cam}_deep-fly-model-resnet50_{timestr}_{epoch:02d}epochs.pt'))
