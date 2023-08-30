import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
from dataset import * 
from models import ResNet

def main(args):
    # process command ling args
    data_dir = Path(args.input_dir)
    logs_dir = Path(args.logs) 
    debug = args.debug

    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"cuda?: {device}")

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

    # define model and freeze parameters
    model = ResNet()
    for name, param in model.named_parameters():
        if "last_layer" in name or "resnet.conv1" in name or "sigmoid" in name:
            continue
        param.requires_grad = False 
    
    for name, param in model.named_parameters():
        if param.requires_grad == True: 
            print(name)
       

    # for i, (x, y_hat) in enumerate(endovis_dataloader):
    #     x = x.to(device)
    #     y_hat = y_hat.to(device)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)