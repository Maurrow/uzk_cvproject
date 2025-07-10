import numpy as np
import shutil
import pickle
from pathlib import Path
import re

source_dir = Path("unsorted_data/training")
target_dir = Path("data/training")

for cam_id in range(7):
    (target_dir / f"cam{cam_id}" / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / f"cam{cam_id}" / "annotations").mkdir(parents=True, exist_ok=True)

keypoints_by_cam = {i: [] for i in range(7)}

for folder in sorted(source_dir.glob("aDN_*_*")):
    if not folder.is_dir():
        continue

    match = re.match(r"aDN_(\d+)_(\d+)", folder.name)
    if not match:
        print(f"Überspringe ungültigen Ordnernamen: {folder.name}")
        continue

    fly_id = int(match.group(1))
    zip_id = match.group(2)

    image_dir = folder / "images"
    ann_file = list((image_dir / "df3d").glob("df3d_*.pkl"))[0]

    if not image_dir.is_dir():
        print(f"Bilderordner fehlt: {image_dir}")
        continue

    if not ann_file.is_file():
        print(f"Keine Annotation gefunden: {ann_file}")
        continue

    with open(ann_file, "rb") as f:
        data = pickle.load(f)

    points2d = data["points2d"] 

    for img_path in sorted(image_dir.glob("camera_*_img_*.jpg")):
        match_img = re.match(r"camera_(\d+)_img_(\d+).jpg", img_path.name)
        if not match_img:
            continue

        cam_id = int(match_img.group(1))
        frame_id = int(match_img.group(2))

        new_name = f"fly{fly_id}_zip{zip_id}_{frame_id:06d}.jpg"
        new_path = target_dir / f"cam{cam_id}" / "images" / new_name
        shutil.copy(img_path, new_path)

        try:
            kp = points2d[cam_id, frame_id, :, :] 
            keypoints_by_cam[cam_id].append(kp)
        except IndexError:
            print(f"Keine Keypoints für cam {cam_id}, frame {frame_id}")

for cam_id, kplist in keypoints_by_cam.items():
    if kplist:
        arr = np.stack(kplist, axis=0)
        out_path = target_dir / f"cam{cam_id}" / "annotations" / "annotations.npz"
        np.savez_compressed(out_path, points2d=arr)

