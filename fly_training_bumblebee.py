import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.transforms.functional import normalize
from tqdm import tqdm

def train_keypoint_model(model, dataset, num_epochs=10, batch_size=16, lr=1e-4, device="cuda"):
    # ImageNet-Preprocessing aus den offiziellen Weights laden
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for img, keypoints, visible in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # apply ImageNet-normalization to each sample in the batch
            img = torch.stack([preprocess(i.cpu()) for i in img])  # apply transform per image
            img = img.to(device)
            keypoints = keypoints.to(device)
            visible = visible.to(device)

            preds = model(img)  # [B, J, 2]

            # Sichtbare Punkte maskieren
            mask = visible.unsqueeze(-1)
            loss = ((preds - keypoints) ** 2) * mask
            loss = loss.sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    return model