import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torchvision.transforms import v2

from fly_dataset import FLYDataset
from fly_cnn import ResNet50Keypoints, CNN_Fly

# Train
def train_keypoint_model(model, dataset, num_epochs=10, batch_size=16, lr=1e-4, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        iter = 0
        for img, keypoints in loop:
            if iter > 100:
                break
            img       = img.to(device)           # [B, 1, H, W]
            keypoints = keypoints.to(device)     # [B, J, 2]

            preds, _, _, _ = model(img)       # [B, J, 2]
            keypoints_px = keypoints.clone()

            mask = (keypoints_px != -1).all(dim=-1)  # [B, J]
            keypoints_px[:, 0] *= dataset.H
            keypoints_px[:, 1] *= dataset.W

            preds_px = preds.clone()
            preds_px[:, 0] *= dataset.H
            preds_px[:, 1] *= dataset.W
            
            loss = ((preds_px - keypoints_px) ** 2).sum(dim=-1)  # [B, J]
            print(loss)
            loss = loss[mask].mean()
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            iter +=1

        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), os.path.join('.', f'fly-test-resnet50.pt'))

    return model


# Pre-Trained ResNet50 Model (ImageNet) to have a better starting point 
model = CNN_Fly(input_size=(480, 960), embedding_size=32)
#ResNet50Keypoints()

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

# model.load_state_dict(torch.load('/scratch/cv-course2025/group2/uzk_cvproject/fly-test-resnet50_nice.pt'))
# model.to(device)

m = train_keypoint_model(model, train_dataset, num_epochs=1)
#model.load_state_dict(torch.load("./fly-test.pt"))
#model.to(device)
#visualize_predictions(m, test_dataset, device=device)
