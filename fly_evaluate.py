from tqdm import tqdm
import torch
from torchvision.transforms.functional import normalize
from IPython.display import clear_output
from fly_visualizer import visualize_fly_batch

def fly_evaluate_visualize(
    model,
    dataset,
    device="cuda:0",
    pck_thresh=10,          # now interpreted in pixels
    visualize_every=500
):
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    H, W = dataset.H, dataset.W
    scale = torch.tensor([H, W], dtype=torch.float32, device=device)
    
    total_mse     = 0.0
    total_rmse    = 0.0
    total_correct = 0
    total_visible = 0

    min_euc = 999
    max_euc = 0
    
    extr_images, extr_gts, extr_preds, extr_vis, extr_titles = [None] * 3, [None] * 3, [None] * 3, [None] * 3, [None] * 3

    for i in tqdm(range(len(dataset)), desc="Evaluating", ncols=100):
        img, keypts, visible = dataset[i]
        #normalize & batch→device

        gt  = keypts.to(device)                                 # [J,2] normalized
        mask= visible.to(device)
        test_img = img.to(device).unsqueeze(0)
        test_img = normalize(test_img, mean=mean, std=std)      # [1, 3, H, W] on CUDA
        with torch.no_grad():
            pred_b = model(test_img)                # [J,2] normalized
        pred = pred_b.squeeze(0)
        # # --- convert to pixel coords ---
        pred_px = pred * scale
        gt_px   = gt   * scale

        diff    = pred_px[mask] - gt_px[mask]
        sq_err  = diff.pow(2).sum(dim=1)                  
        euc     = sq_err.sqrt()                              

        n_vis   = mask.sum().item()
        if n_vis == 0:
            continue

        # accumulate
        total_mse     += sq_err.mean().item()  * n_vis
        total_rmse    += euc.mean().item()     * n_vis
        total_correct += (euc < pck_thresh).sum().item()
        total_visible += n_vis

        if min_euc > euc.mean().item() or extr_images[0] is None:
            title = f"#{i} Lowest RMSE={euc.mean().item():.1f}px PCK={100*(euc< pck_thresh).float().mean().item():.1f}%"
            extr_images[0] = test_img.cpu().squeeze(0)
            extr_gts[0] = keypts
            extr_preds[0] = pred
            extr_vis[0] = visible
            extr_titles[0] = title
            min_euc = euc.mean().item()

        if max_euc < euc.mean().item() or extr_images[2] is None:
            title = f"#{i} Highest RMSE={euc.mean().item():.1f}px PCK={100*(euc< pck_thresh).float().mean().item():.1f}%"
            extr_images[2] = test_img.cpu().squeeze(0)
            extr_gts[2] = keypts
            extr_preds[2] = pred
            extr_vis[2] = visible
            extr_titles[2] = title
            max_euc = euc.mean().item()


        # visualize
        if i % visualize_every == 0:
            title = f"#{i} RMSE={euc.mean().item():.1f}px PCK={100*(euc< pck_thresh).float().mean().item():.1f}%"
            extr_images[1] = test_img.cpu().squeeze(0)
            extr_gts[1] = keypts
            extr_preds[1] = pred
            extr_vis[1] = visible
            extr_titles[1] = title

            clear_output(wait=True)
            #print(extr_images)
            visualize_fly_batch(extr_images, extr_gts, extr_preds, extr_vis, extr_titles)

    # averages
    mean_mse  = total_mse  / total_visible
    mean_rmse = total_rmse / total_visible
    mean_pck  = total_correct / total_visible

    print(f"\nResults:  MSE={mean_mse:.2f}px²,  RMSE={mean_rmse:.2f}px,  PCK@{pck_thresh}px={mean_pck*100:.2f}%")
    return mean_mse, mean_rmse, mean_pck
