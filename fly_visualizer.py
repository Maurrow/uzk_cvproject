import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import numpy as np
from df3d.config import config

def visualize_fly(image, keypoints_gt, keypoints_pred=None, visible=None, title="Fly Visualization"):
    """
    Visualizes a single fly image with ground truth keypoints and optionally predicted keypoints.

    Parameters:
    -----------
    image : torch.Tensor or numpy.ndarray
        The fly image of shape [3, H, W] or [1, H, W]; assumed to be fake-RGB (all channels equal).

    keypoints_gt : torch.Tensor
        Ground truth keypoints (normalized).

    keypoints_pred : torch.Tensor, optional
        Predicted keypoints (normalized). If provided, predictions are also plotted.

    visible : torch.Tensor or list, optional
        Boolean array marking which keypoints are visible. If None, inferred from GT.

    title : str
        Title for the plot.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

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

    # Scatter Ground Truth
    for i, (x, y) in enumerate(keypoints_gt):
        if visible[i]:
            plt.scatter(y, x, c="lime", s=20, label="GT" if i == 0 else "", marker="o")

    # Scatter Predictions
    if keypoints_pred is not None:
        for i, (x, y) in enumerate(keypoints_pred):
            if visible[i]:
                plt.scatter(y, x, c="red", s=20, label="Pred" if i == 0 else "", marker="x")

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_fly_with_limbs(image, keypoints_gt, keypoints_pred=None, visible=None, title="Fly w/ Limbs"):
    """
    Visualizes a fly image with both ground truth and predicted keypoints, as well as limb connections (bones).

    Parameters:
    -----------
    image : torch.Tensor or numpy.ndarray
        The fly image of shape [3, H, W] or [1, H, W]; assumed to be fake-RGB (all channels equal).

    keypoints_gt : torch.Tensor
        Ground truth keypoints (normalized).

    keypoints_pred : torch.Tensor, optional
        Predicted keypoints (normalized). If provided, predictions are also plotted.

    visible : torch.Tensor or list, optional
        Boolean array marking which keypoints are visible. If None, inferred from GT.

    title : str
        Title for the plot.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = image[0]
    H, W = image.shape

    kp_gt = keypoints_gt.detach().cpu().numpy() * [H, W]
    
    kp_pred = (keypoints_pred.detach().cpu().numpy() * [H, W]
               if keypoints_pred is not None else None)
    
    vis   = (visible.detach().cpu().numpy()
             if visible is not None else (kp_gt[:,0]>=0)&(kp_gt[:,1]>=0))

    plt.figure(figsize=(8,4))
    plt.imshow(image, cmap="gray")
    plt.title(title)

    # Fetch the “true” 38-point skeleton from DeepFly3D config
    skeleton = config["bones"]

    # Draw GT joints & bones
    for i,(x,y) in enumerate(kp_gt):
        if vis[i]:
            plt.scatter(y, x, c="lime", s=20, marker="o",
                        label="GT" if i==0 else "")
    for a,b in skeleton:
        if vis[a] and vis[b]:
            xa,ya = kp_gt[a]
            xb,yb = kp_gt[b]
            plt.plot([ya,yb],[xa,xb], c="lime", lw=2)

    # Draw Pred joints & bones
    if kp_pred is not None:
        for i,(x,y) in enumerate(kp_pred):
            if vis[i]:
                plt.scatter(y, x, c="red", s=20, marker="x",
                            label="Pred" if i==0 else "")
        for a,b in skeleton:
            if vis[a] and vis[b]:
                xa,ya = kp_pred[a]
                xb,yb = kp_pred[b]
                plt.plot([ya,yb],[xa,xb], c="red", lw=2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_fly_batch(images, gts, preds, visibles, titles):
    """
    Visualizes a batch of fly images with their keypoints and limbs in a single row of subplots.

    Parameters:
    -----------
    images : list of torch.Tensor
        List of fly images of shape [1, H, W] (grayscale or fake-RGB).

    gts : list of torch.Tensor
        List of ground truth keypoints (normalized) per image.

    preds : list of torch.Tensor
        List of predicted keypoints (normalized) per image.

    visibles : list of torch.Tensor
        List of boolean visibility masks per image.

    titles : list of str
        List of titles (one per subplot/image).
    """
    num = len(images)
    clear_output(wait=True)
    fig, axes = plt.subplots(1, num, figsize=(num * 6, 4))

    if num == 1:
        axes = [axes]

    for ax, image, gt, pred, visible, title in zip(axes, images, gts, preds, visibles, titles):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        # Prepare Plot Legend
        gt_handle   = ax.scatter([], [], c="lime", marker="o", s=30, label="GT")
        pred_handle = ax.scatter([], [], c="red",  marker="x", s=30, label="Pred")

        image = image[0]
        H, W = image.shape
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

        gt           = gt.detach().cpu().numpy() * [H, W]
        pred         = pred.detach().cpu().numpy() * [H, W]
        visible      = visible.detach().cpu().numpy()
        # We create a pred_mask that differs from the gt_mask to be able to
        # perform evaluations on other cameras even though the model was 
        # trained on a specific camera.
        # This is done by taking the sum of the visible points in the visible 
        # tensor and creating a mask which sets the first visible_sum points 
        # to visible in the prediction.
        visible_sum  = sum(1 for v in visible if v)
        pred_visible = [True] * visible_sum + [False] * (len(visible) - visible_sum)

        # Scatter GT
        for i, (x, y) in enumerate(gt):
            if visible[i]:
                ax.scatter(y, x, c="lime", s=10, marker="o", label="GT" if i == 0 else "")

        # Scatter Predictions
        if pred is not None:
            for i, (x, y) in enumerate(pred):
                if pred_visible[i]:
                    ax.scatter(y, x, c="red", s=10, marker="x", label="Pred" if i == 0 else "")

        # DeepFly3D Config    
        skeleton = config["bones"]
        
        # GT Bones / Limbs
        for a,b in skeleton:
            if visible[a] and visible[b]:
                xa,ya = gt[a]
                xb,yb = gt[b]
                ax.plot([ya,yb],[xa,xb], c="lime", lw=2)

        # Pred Bones / Limbs
        for a,b in skeleton:
            if pred_visible[a] and pred_visible[b]:
                xa,ya = pred[a]
                xb,yb = pred[b]
                ax.plot([ya,yb],[xa,xb], c="red", lw=2)
    
    fig.legend(
        handles=[gt_handle, pred_handle],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=True,
        fontsize=18
    )
    plt.tight_layout()
    plt.show()
