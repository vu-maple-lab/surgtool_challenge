# main script to call for challenge 
# given a directory of videos and the image labels of the tools,
# classify and localize the bounding boxes around each tool. 

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
    pass

if __name__ == '__main__':
    main()