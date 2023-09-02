import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
from pathlib import Path
import math
import os
from dataset import * 
from models import ResNet
from utils import trial

def main(args):

    # process command ling args
    data_dir = Path(args.input_dir)
    logs_dir = Path(args.logs) 
    model_dir = Path(args.model)
    batch_size = args.batch_size
    debug = args.debug

    # path checking
    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not logs_dir.exists():
        raise Exception("logs directory does not exist")
    if not model_dir.exists():
        raise Exception("pretrained model path does not exist")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')
    if debug:
        print(f'Batch size: {batch_size}')

    # augmentations to apply
    color_transforms = T.Compose([T.ToPILImage(), 
                                  T.Resize(512), 
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    mask_transforms = T.Compose([T.ToPILImage(), 
                                 T.Resize(512), 
                                 T.ToTensor()])
    # define dataset
    dataset = Endovis23Dataset(data_dir, train=True, debug=debug, color_transforms=color_transforms, mask_transforms=mask_transforms)

    # split train-test + define dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    if debug:
        _, _ = next(iter(dataloader))

    # load pretrained model
    model = ResNet()
    model.load_state_dict(torch.load(str(model_dir)))

    # define our loss function
    loss = nn.BCEWithLogitsLoss()

    # train + test
    logs = trial(model, dataloader, loss, debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--logs', required=True) 
    parser.add_argument('--model', required=True)
    parser.add_argument('--num_epochs', default=20, type=int, required=True)
    parser.add_argument('--batch_size', default=8, type=int, required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)