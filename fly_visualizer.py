import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import numpy as np
from df3d.config import config

def visualize_fly(image, keypoints_gt, keypoints_pred=None, visible=None, title="Fly Visualization"):
    """
    Visualisiert Fly-Bild mit Ground-Truth-Keypoints und optional Predicted Keypoints.
    Geht davon aus, dass das Bild [3, H, W] Fake-RGB ist (alle Kanäle gleich).
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
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = image[0]                     # fake-RGB → [H, W]
    H, W = image.shape

    kp_gt = keypoints_gt.detach().cpu().numpy() * [H, W]
    
    kp_pred = (keypoints_pred.detach().cpu().numpy() * [H, W]
               if keypoints_pred is not None else None)
    
    vis   = (visible.detach().cpu().numpy()
             if visible is not None else (kp_gt[:,0]>=0)&(kp_gt[:,1]>=0))

    plt.figure(figsize=(8,4))
    plt.imshow(image, cmap="gray")
    plt.title(title)

    # fetch the “true” 38-point skeleton from DeepFly3D
    skeleton = config["bones"]  # a list of [parent,child] pairs 

    # draw GT joints & bones
    for i,(x,y) in enumerate(kp_gt):
        if vis[i]:
            plt.scatter(y, x, c="lime", s=20, marker="o",
                        label="GT" if i==0 else "")
    for a,b in skeleton:
        if vis[a] and vis[b]:
            xa,ya = kp_gt[a]
            xb,yb = kp_gt[b]
            plt.plot([ya,yb],[xa,xb], c="lime", lw=2)

    # draw Pred joints & bones
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

    # legend cleanup
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_fly_batch(images, gts, preds, visibles, titles):
    num = len(images)
    clear_output(wait=True)
    fig, axes = plt.subplots(1, num, figsize=(num * 6, 6))

    if num == 1:
        axes = [axes]

    for ax, image, gt, pred, visible, title in zip(axes, images, gts, preds, visibles, titles):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = image[0]
        H, W = image.shape
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

        gt = gt.detach().cpu().numpy() * [H, W]
        pred = pred.detach().cpu().numpy() * [H, W]
        visible = visible.detach().cpu().numpy()


        # GT Punkte
        for i, (x, y) in enumerate(gt):
            if visible[i]:
                ax.scatter(y, x, c="lime", s=10, marker="o", label="GT" if i == 0 else "")

        # Pred Punkte
        if pred is not None:
            for i, (x, y) in enumerate(pred):
                if visible[i]:
                    ax.scatter(y, x, c="red", s=10, marker="x", label="Pred" if i == 0 else "")
            
        skeleton = config["bones"]
        

        # draw bones

        for a,b in skeleton:
            if visible[a] and visible[b]:
                xa,ya = gt[a]
                xb,yb = gt[b]
                ax.plot([ya,yb],[xa,xb], c="lime", lw=2)

        for a,b in skeleton:
            if visible[a] and visible[b]:
                xa,ya = pred[a]
                xb,yb = pred[b]
                ax.plot([ya,yb],[xa,xb], c="red", lw=2)


    plt.tight_layout()
    plt.show()
