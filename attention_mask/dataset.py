import os
import torch 
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import glob as glob
from natsort import natsorted
from torchvision import transforms, utils
from pathlib import Path
import cv2 as cv
from utils import TOOLS_ONE_HOT_ENCODING, process_binary

class Endovis23Dataset(Dataset):

    def __init__(self, root_dir, debug=False, transform=None):
        self.transforms = transform
        self.debug = debug

        color_dir = root_dir / 'raw' / 'color'
        mask_dir = root_dir / 'raw' / 'processed_mask'
        labels_path = root_dir / 'labels.csv'

        if not ((color_dir).exists() and (mask_dir).exists() and (labels_path).exists()):
            raise Exception("Your input_dir must include a labels.csv and a raw/ file with color/ and mask/ inside")
        
        # extract the paths of the color and mask imgs
        self.path_color_imgs = color_dir
        self.path_mask_imgs = natsorted(glob.glob(str(mask_dir / '*.jpg')))

        # extract csv file
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        if self.debug:
            print(f'The length of our data is {len(self.path_mask_imgs)}')
        return len(self.path_mask_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # image = np.array(Image.open(self.path_color_imgs[idx]))
        mask = cv.imread(self.path_mask_imgs[idx])
        image = self.find_corresponding_img(mask)

        # first apply transformations if they exists
        if self.transforms:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # apply smoothed attention mask 
        attn_mask = self.get_attention_mask(mask)
        attentioned_image = self.apply_attention(image, attn_mask)

        # extract label
        img_name = 'clip_' + str(Path(self.path_color_imgs[idx]).stem)[:6]
        if self.debug:
            print(f'The clip name is: {img_name}')
        try:
            clip_index = list(self.labels['clip_name']).index(img_name)
        except ValueError: 
            print('can not find clip name from labels.csv')
        tool_label = list(self.labels['tools_present'])[clip_index]
        tool_label = tool_label[1:-1].split(', ') # get rid of [] chars and put into list of strings

        # some had leading or ending spaces which we need to get rid of 
        for i, label in enumerate(tool_label):
            if label[-1] == ' ':
                tool_label[i] = tool_label[i][:-1]
            if label[0] == ' ':
                tool_label[i] = tool_label[i][1:]
        if 'nan' in tool_label:
            tool_label.remove('nan')
        
        # one-hot encoding of label
        tool_label_onehot = self.get_one_hot(tool_label)
        return np.array(attentioned_image), tool_label_onehot

    def find_corresponding_img(self, mask_path):
        mask_name = Path(mask_path).name
        return cv.imread(str(self.path_color_imgs / mask_name))

    def get_one_hot(self, labels):
        result = np.zeros(14)
        for label in labels:
            if not label in TOOLS_ONE_HOT_ENCODING:
                raise Exception(f'The label {label} is not in our dict.')
            result[TOOLS_ONE_HOT_ENCODING[label]] = 1
        if self.debug:
            print(f'Our labels are:\n {labels}')
            print(f'Our one hot encoding is:\n {result}')
        return result

    def get_attention_mask(self, mask):
        assert len(mask.shape) == 2, 'your mask should be some binary or grayscale image not color'
        kernel = np.ones((7,7),np.float32) / 49
        blurred_img = cv.filter2D(mask, -1, kernel)
        if self.debug:
            blurred_img.save('./test/attention_mask_debug.jpg')
        return blurred_img
    
    def apply_attention(self, image, mask):
        assert image.size[0] == mask.size[0] and image.size[1] == mask.size[1], 'dimensions of the image and mask should match'
        result = image * mask
        if self.debug:
            image.save('./test/original_rgb_debug.jpg')
            result.save('./test/rgb_applied_attention_debug.jpg')
        return result.astype('uint8')