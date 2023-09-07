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
from utils import train, prepare_logs, save_vis, color_transforms, mask_transforms

# python train.py --input_dir ../data/ --logs ../logs/ --num_epochs 25 --batch_size 8

def main(args):

    # process command ling args
    data_dir = Path(args.input_dir)
    logs_dir = Path(args.logs) 
    epochs = args.num_epochs
    batch_size = args.batch_size
    debug = args.debug
    pretrained = args.pretrained

    # path checking
    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not logs_dir.exists():
        raise Exception("logs directory does not exist")
    if pretrained:
        pretrained = Path(pretrained)
        if not pretrained.exists():
            raise Exception("pretrained path does not exist")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')
    if debug:
        print(f'Num Epochs: {epochs}')
        print(f'Batch size: {batch_size}')

    # define dataset
    train_dataset = Endovis23Dataset(data_dir, train=True, debug=debug, color_transforms=color_transforms, mask_transforms=mask_transforms)
    test_dataset = Endovis23Dataset(data_dir, train=False, debug=debug, color_transforms=color_transforms, mask_transforms=mask_transforms)

    # split train-test + define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    if debug:
        _, _ = next(iter(train_dataloader))
    prepare_logs(logs_dir)

    # define model and freeze parameters
    model = ResNet()
    # define our loss function and optimizer
    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1.)
    steps = math.floor(len(train_dataset) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)

    if pretrained:
        try:
            print('Loading in pretrained model, optimizer, and scheduler...')
            
            model_path = str(pretrained / 'checkpoints' / 'best_model.pt')
            model.load_state_dict(torch.load(model_path))

            # load optimizer and scheduler
            optim_path = str(pretrained / 'checkpoints' / 'optimizer.pt')
            optim.load_state_dict(torch.load(optim_path))

            sched_path = str(pretrained / 'checkpoints' / 'scheduler.pt')
            scheduler.load_state_dict(torch.load(sched_path))

        except Exception:
            print('ERROR: one of the pretrained paths does not exist')
            return 

    for name, param in model.named_parameters():
        if "last_layer" in name or "resnet.conv1" in name:
            continue
        param.requires_grad = False   

    # train + test
    logs = train(model, (train_dataloader, test_dataloader), loss, optim, scheduler, epochs, logs_dir, debug)
    save_vis(logs_dir, logs)
    print(f'Finished Training! Logs saved to {str(logs_dir)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--pretrained')
    parser.add_argument('--num_epochs', default=50, type=int, required=True)
    parser.add_argument('--batch_size', default=4, type=int, required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)