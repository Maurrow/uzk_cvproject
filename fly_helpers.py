import matplotlib.pyplot as plt
import torch

def visualize_fly(image, keypoints_gt, keypoints_pred=None, visible=None, title="Fly Visualization"):
    """
    Visualisiert Fly-Bild mit Ground-Truth-Keypoints und optional Predicted Keypoints.
    Geht davon aus, dass das Bild [3, H, W] Fake-RGB ist (alle Kanäle gleich).
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Extrahiere einfach den ersten Kanal → echtes Graubild
    image = image[0]  # [H, W]

    H, W = image.shape

    keypoints_gt = keypoints_gt.detach().cpu().numpy() * [H, W]
    if keypoints_pred is not None:
        keypoints_pred = keypoints_pred.detach().cpu().numpy() * [H, W]

    if visible is not None:
        visible = visible.detach().cpu().numpy()
    else:
        visible = (keypoints_gt[:, 0] >= 0) & (keypoints_gt[:, 1] >= 0)

    plt.figure(figsize=(8, 4))
    plt.imshow(image, cmap="gray")
    plt.title(title)

    # Ground Truth
    for i, (x, y) in enumerate(keypoints_gt):
        if visible[i]:
            plt.scatter(y, x, c="lime", s=20, label="GT" if i == 0 else "", marker="o")

    # Prediction
    if keypoints_pred is not None:
        for i, (x, y) in enumerate(keypoints_pred):
            if visible[i]:
                plt.scatter(y, x, c="red", s=20, label="Pred" if i == 0 else "", marker="x")

    # Legende
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.axis("off")
    plt.tight_layout()
    plt.show()