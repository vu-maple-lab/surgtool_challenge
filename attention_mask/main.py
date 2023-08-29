import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
from dataset import * 
from models import ResNet, ResidualBlock

def main(args):
    # process command ling args
    data_dir = Path(args.input_dir)
    logs_dir = Path(args.logs) 
    debug = args.debug

    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # path checking
    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not logs_dir.exists():
        raise Exception("logs directory does not exist")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')

    # define dataset + dataloader
    endovis_dataset = Endovis23Dataset(data_dir, debug)
    endovis_dataloader = DataLoader(endovis_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # model is ResNet-101
    model = ResNet(ResidualBlock, [3, 4, 23, 3]).to(device)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)