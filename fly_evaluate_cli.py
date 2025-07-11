import argparse
import torch

from fly_dataset import FLY_Dataset
from fly_resnet import FLY_Resnet
from fly_evaluate import fly_evaluate_visualize

parser = argparse.ArgumentParser(description="Evaluate and visualize a fly pose model.")
parser.add_argument('--data', type=str, required=True, help='Path to dataset root. Test dataset will be used for evaluation')
parser.add_argument('--model_path', type=str, required=True, help='Path to model')
parser.add_argument('--cam', type=int, default=0, help='Camera ID')
parser.add_argument('--pck_thresh', type=float, default=10.0, help='PCK threshold in pixels')
parser.add_argument('--visualize_every', type=int, default=500, help='Visualization interval')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = FLY_Dataset(path_to_data=args.data, mode="test", cam=args.cam, backbone="resnet")

model = FLY_Resnet().to(device)
if args.model_path:
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

fly_evaluate_visualize(
    model=model,
    dataset=dataset,
    device=device,
    pck_thresh=args.pck_thresh,
    visualize_every=args.visualize_every
)
