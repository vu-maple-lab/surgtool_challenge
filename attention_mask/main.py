import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import math
import matplotlib.pyplot as plt
import os
from dataset import * 
from models import ResNet
from utils import train

def main(args):
    # process command ling args
    data_dir = Path(args.input_dir)
    logs_dir = Path(args.logs) 
    epochs = args.num_epochs
    batch_size = args.batch_size
    debug = args.debug


    # path checking
    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not logs_dir.exists():
        raise Exception("logs directory does not exist")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')
    if debug:
        print(f'Num Epochs: {epochs}')
        print(f'Batch size: {batch_size}')

    # augmentations to apply
    transforms = T.Compose([T.ToPILImage(), 
                            T.Resize(256), 
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # define dataset
    endovis_dataset = Endovis23Dataset(data_dir, debug, transforms=transforms)

    # split train-test + define dataloader
    train_length = int(0.7 * len(endovis_dataset))
    test_length = len(endovis_dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(endovis_dataset, (train_length, test_length))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    if debug:
        _, _ = next(iter(train_dataloader))
    breakpoint()

    # define model and freeze parameters
    model = ResNet()
    for name, param in model.named_parameters():
        if "last_layer" in name or "resnet.conv1" in name:
            continue
        param.requires_grad = False   

    # define our loss function and optimizer
    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1.)
    steps = math.floor(len(train_dataset) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)

    # train + test
    logs = train(model, (train_dataloader, test_dataloader), loss, optim, scheduler, epochs, debug)
    # results = test(model, test_dataloader, loss, debug)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--num_epochs', default=50, type=int, required=True)
    parser.add_argument('--batch_size', default=4, type=int, required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)