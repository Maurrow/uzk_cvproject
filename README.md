# Animal Pose Esitmation: 2D Limb Pose Estimation for Tethered Fruit Flies

## Project Description

This project implements a 2D keypoint detection pipeline for predicting limb poses in tethered flies using deep learning. The model is trained on one camera fly recordings and annotations provided by DeepFly3D. It uses a ResNet50 backbone adapted for regression of 38 keypoints per frame.

## Quick Start

### 0. Prerequisites
Make sure to setup a conda environment with the provided requirements.txt. 

In case you want to create annotations on your own dataset, you can use the DeepFly3D CLI. The installation documentation can be found [here](https://github.com/NeLy-EPFL/DeepFly3D?tab=readme-ov-file#installing).

### 1. Organize dataset:
> ⚠️ **Please note:** The Data Structure described [here](#data-structure) is a requirement for this script to work correctly. 
```bash
python3 create_data_structure.py --source <path_to_unsorted_data>/<training or test> --target <path_to_data_store>/<training or test>
```

#### Data Structure

##### `unsorted_data/` (expected input)

This directory should contain individual fly recordings named as:

```
unsorted_data/
└── training/
    ├── aDN_<fly_id>_<zip_id>/
    │   └── images/
    │       ├── camera_0_img_000000.jpg
    │       ├── camera_1_img_000000.jpg
    │       └── df3d/
    │           └── df3d_result.pkl
    └── ...
```

Each `aDN_*_*` folder must contain:
- an `images/` subfolder with `.jpg` images named as `camera_<cam_id>_img_<frame_id>.jpg`
- a `df3d/` folder inside `images/`, containing one `df3d_result.pkl` with 2D keypoint annotations under `data["points2d"]` (these annotations are created with df3d-cli).


##### `data/` (recommended structure)
Before running the script you should ideally have a structure like this:
```
data/
└── training/
└── ...
```

---
##### `data/` (processed output)

After running the script, the structure will be:

```
data/
└── training/
    ├── cam0/
    │   ├── images/
    │   │   └── fly<id>_zip<id>_<frame>.jpg
    │   └── annotations/
    │       └── annotations.npz
    ├── cam1/
    └── ...
```

Each `camX/` directory contains:
- all fly images from that camera
- a compressed `.npz` file with extracted 2D keypoints for that camera

Repeat the process for any other data splits you may need.

### 2. Train the model:
```bash
python3 fly_training.py --data <path-to-data>
```
optional parameters:<br>
| Argument               | Description                                                                                         | Default   |
|------------------------|-----------------------------------------------------------------------------------------------------|-----------|
| `--cam <cam_id>`       | ID of the camera to train the model on                                                             | `0`       |
| `--epochs <num_epochs>`| Number of training epochs                                                                           | `100`     |
| `--batch_size <size>`  | Batch size used during training                                                                     | `16`      |
| `--lr <learning_rate>` | Learning rate for the optimizer                                                                     | `1e-4`    |
| `--patience <num_epochs>`       | Epochs with no improvement before early stopping is triggered                                       | `5`       |

### 3. Evaluate and visualize predictions:

```bash
python3 fly_evaluate_cli.py --data <path-to-data> --model_path <path_to_model>
```
optional parameters:<br>
| Argument                    | Description                                                                 | Default   |
|-----------------------------|-----------------------------------------------------------------------------|-----------|
| `--cam <cam_id>`            | ID of the camera used for evaluation                                       | `0`       |
| `--pck_thresh <threshold>`  | Pixel threshold for computing PCK (Percentage of Correct Keypoints)        | `10.0`    |
| `--visualize_every <int>`   | Frequency (in samples) to trigger batch visualization                      | `500`     |

In case you want to visualize on a remote you can also use the Jupyter Notebook (fly_evaluate_visualize.ipynb) and either adjust or see the saved visualizations and evaluations. 

## File Overview
| File/Dir                       | Description                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------|
| `requirements.txt`                       | Conda environment list export                                              |
| `create_data_structure.py`               | Extracts and reorganizes raw annotated fly data by camera                  |
| `fly_dataset.py`                         | Dataset class for loading images, keypoints, and visibility masks          |
| `fly_training.py`                        | Training loop with rotation augmentation and early stopping                |
| `fly_resnet.py`                          | ResNet50-based keypoint regression model definition                        |
| `fly_evaluate.py`                        | Evaluation and visualization: RMSE, PCK, and image plotting                |
| `fly_visualizer.py`                      | Functions to visualize single or batch predictions with skeleton structure |
| `fly_evaluate_visualize.ipynb`           | Jupyter notebook version for interactive experimentation                   |
| `models/*`                               | Directory where the trained models are stored. The best performing one being: `cam0_deep-fly-model-resnet50_20250711-020806_27epochs`                                                                 |

## References
- ResNet50 backbone sourced from torchvision.models, pretrained on ImageNet
- Bone visualization logic / limb detection is using config["bones"] from the DeepFly3D structure.