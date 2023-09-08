import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import math
import os
from dataset import * 
from models import ResNet
from utils import run_trial, color_transforms, mask_transforms, device

# python trial.py --input_dir ../data/ --model_path ../logs/checkpoints/best_model.pt --output ../output/ --debug

def main(args):

    # process command ling args
    data_dir = Path(args.input_dir)
    model_path = Path(args.model_path)
    output_path = Path(args.output)
    debug = args.debug
    batch_size = 1 # enforced for trial phase
    # path checking
    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not model_path.exists():
        raise Exception("model directory does not exist")
    if not output_path.exists():
        print(f'Output path does not exist, so creating one at {str(output_path)}')
        os.mkdir(str(output_path))
    if debug and not os.path.exists('./test'):
        print(f'test directory does not exist so creating one at ./test')
        os.mkdir('./test')
    if debug:
        print(f'Batch size: {batch_size}')
    
    # define dataset + dataloader
    dataset = Endovis23Dataset(data_dir, train=True, debug=debug, color_transforms=color_transforms, mask_transforms=mask_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # define model and load our pretrained network
    model = ResNet()
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception:
        print("Invalid model path.")
    model.to(device)
    model.eval()

    # define our loss function and optimizer
    loss = nn.BCEWithLogitsLoss()
    results = run_trial(model, dataloader, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)