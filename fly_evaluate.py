from tqdm import tqdm
import torch
from torchvision.transforms.functional import normalize
from IPython.display import clear_output
from fly_helpers import visualize_fly_batch

def fly_evaluate_and_visualize(model, dataset, device="cuda", pck_thresh=5, visualize_every=500):
    model = model.to(device)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    total_mse = 0.0
    total_rmse = 0.0
    total_pck = 0.0
    total_visible = 0

    last_images = []
    last_gts = []
    last_preds = []
    last_visibles = []
    last_titles = []

    for i in tqdm(range(len(dataset)), desc="Evaluating on test set"):
        img_tensor, keypoints, visible = dataset[i]

        vis_image = img_tensor[0].clone().cpu()

        img_tensor = img_tensor.normalize(img_tensor, mean=mean, std=std).unsqueeze(0).to(device)
        gt_px = keypoints.to(device)
        visible = visible.to(device)

        with torch.no_grad():
            pred = model(img_tensor)
        pred_px = pred.squeeze(0)

        mask = visible.bool()
        num_visible = mask.sum().item()
        if num_visible == 0:
            continue

        diff = pred_px[mask] - gt_px[mask]
        mse = (diff ** 2).mean().item()
        rmse = torch.norm(diff, dim=1).mean().item()
        pck = (torch.norm(diff, dim=1) < pck_thresh).float().mean().item()

        total_mse += mse * num_visible
        total_rmse += rmse * num_visible
        total_pck += pck * num_visible
        total_visible += num_visible

        # Visualisierung alle X Bilder
        if i < 5:
            title = f"#{i} | RMSE={rmse:.1f}px | PCK={pck*100:.1f}%"
            last_images.append(vis_image)
            last_gts.append(gt_px.cpu())
            last_preds.append(pred_px.cpu())
            last_visibles.append(visible.cpu())
            last_titles.append(title)

            if len(last_images) > 3:
                last_images.pop(0)
                last_gts.pop(0)
                last_preds.pop(0)
                last_visibles.pop(0)
                last_titles.pop(0)

            visualize_fly_batch(
                last_images,
                last_gts,
                last_preds,
                last_visibles,
                last_titles
            )

    mean_mse = total_mse / total_visible
    mean_rmse = total_rmse / total_visible
    mean_pck = total_pck / total_visible

    print("\nTotals:")
    print(f"  MSE  = {mean_mse:.2f} px²")
    print(f"  RMSE = {mean_rmse:.2f} px")
    print(f"  PCK  = {mean_pck*100:.2f}% (bei {pck_thresh}px Toleranz)")

    return mean_mse, mean_rmse, mean_pck

def fly_eval(
    model,
    dataset,
    device="cuda",
    pck_thresh=5,          # now interpreted in pixels
    visualize_every=500
):
    model.eval().to(device)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    H, W = dataset.H, dataset.W
    scale = torch.tensor([H, W], device=device)
    print(scale)

    total_mse     = 0.0
    total_rmse    = 0.0
    total_correct = 0
    total_visible = 0

    last_images, last_gts, last_preds, last_vis, last_titles = [], [], [], [], []

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        img, keypts, visible = dataset[i]
        vis_img = img.clone()                                   # for viz

        # normalize & batch→device
        img = normalize(img, mean=mean, std=std).unsqueeze(0).to(device)
        gt  = keypts.to(device)                                 # [J,2] normalized
        mask= visible.to(device)

        with torch.no_grad():
            pred = model(img).squeeze(0)                        # [J,2] normalized

        # --- convert to pixel coords ---
        pred_px = pred * scale
        gt_px   = gt   * scale

        diff    = pred_px[mask] - gt_px[mask]                   # [N_vis,2] in px
        sq_err  = diff.pow(2).sum(dim=1)                        # [N_vis] px squared
        euc     = sq_err.sqrt()                                # [N_vis] px

        n_vis   = mask.sum().item()
        if n_vis == 0:
            continue

        # accumulate
        total_mse     += sq_err.mean().item()  * n_vis
        total_rmse    += euc.mean().item()     * n_vis
        total_correct += (euc < pck_thresh).sum().item()
        total_visible += n_vis

        # visualize occasionally
        if i < 5 or (visualize_every and i % visualize_every == 0):
            title = f"#{i} RMSE={euc.mean().item():.1f}px PCK={100*(euc< pck_thresh).float().mean().item():.1f}%"
            last_images .append(vis_img)
            last_gts    .append(keypts)
            last_preds  .append(pred.cpu())
            last_vis    .append(visible)
            last_titles .append(title)
            if len(last_images)>4:
                for buf in (last_images,last_gts,last_preds,last_vis,last_titles):
                    buf.pop(0)
            clear_output(wait=True)
            visualize_fly_batch(last_images, last_gts, last_preds, last_vis, last_titles)

    # average back out
    mean_mse  = total_mse  / total_visible
    mean_rmse = total_rmse / total_visible
    mean_pck  = total_correct / total_visible

    print(f"\nResults:  MSE={mean_mse:.2f}px²,  RMSE={mean_rmse:.2f}px,  PCK@{pck_thresh}px={mean_pck*100:.2f}%")
    return mean_mse, mean_rmse, mean_pck
