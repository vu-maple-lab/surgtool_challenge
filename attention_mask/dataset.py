import torch 
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import glob as glob
from natsort import natsorted
from pathlib import Path
import cv2 as cv
from utils import TOOLS_ONE_HOT_ENCODING

class Endovis23Dataset(Dataset):

    def __init__(self, root_dir, train=True, debug=False, color_transforms=None, mask_transforms=None):
        self.color_transform = color_transforms
        self.mask_transform = mask_transforms
        self.debug = debug

        if train:
            color_dir = root_dir / 'actual' / 'train' / 'color'
            mask_dir = root_dir / 'actual' / 'train' / 'mask'
        else:
            color_dir = root_dir / 'actual' / 'test' / 'color'
            mask_dir = root_dir / 'actual' / 'test' / 'mask'

        labels_path = root_dir / 'labels.csv'

        if not (color_dir.exists() and mask_dir.exists() and labels_path.exists()):
            raise Exception("Your input_dir must include a labels.csv and a raw/ file with color/ and mask/ inside")
        
        # extract the paths of the color and mask imgs
        self.path_color_imgs = color_dir
        self.path_mask_imgs = natsorted(glob.glob(str(mask_dir / '*.jpg')))

        # extract csv file
        csv_labels = pd.read_csv(labels_path)
        self.clip_names = list(csv_labels['clip_name'])
        self.tools_present = list(csv_labels['tools_present'])

    def __len__(self):
        if self.debug:
            print(f'The length of our data is {len(self.path_mask_imgs)}')
        return len(self.path_mask_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mask = cv.imread(self.path_mask_imgs[idx])
        mask = np.uint8(np.dot(mask[...,:3], [0.2989, 0.5870, 0.1140])) # ensure that we have a 1 channel, uint8 mask 
        image = self.find_corresponding_img(self.path_mask_imgs[idx])
        
        # apply smoothed attention mask 
        r = 15
        attn_mask = self.get_attention_mask(mask, r)
        attentioned_image = self.apply_attention(image, attn_mask)

        # extract label
        img_name = 'clip_' + str(Path(self.path_mask_imgs[idx]).stem)[:6]
        if self.debug:
            print(f'The clip name is: {img_name}')
        try:
            clip_index = self.clip_names.index(img_name)
        except ValueError: 
            print(f'can not find clip name "{img_name}" from labels.csv')
        tool_label = self.tools_present[clip_index]
        tool_label = tool_label[1:-1].split(', ') # get rid of [] chars and put into list of strings

        # some had leading or ending spaces which we need to get rid of 
        for i, label in enumerate(tool_label):
            if label[-1] == ' ':
                tool_label[i] = tool_label[i][:-1]
            if label[0] == ' ':
                tool_label[i] = tool_label[i][1:]
        if 'nan' in tool_label:
            tool_label.remove('nan')
        
        # "one-hot encoding" of label
        y_hat = self.get_one_hot(tool_label)

        # apply transformations if they exists
        if self.color_transform:
            # image = self.color_transform(image)
            attentioned_image = self.color_transform(attentioned_image)
            mask = self.mask_transform(mask)
            if self.debug: 
                print(f'Confirming if we are transforming correctly...')
                img1 = np.moveaxis(torch.Tensor.numpy(image), 0, -1)
                a1 = np.moveaxis(torch.Tensor.numpy(attentioned_image), 0, -1)
                mask1 = np.moveaxis(torch.Tensor.numpy(mask), 0, -1).squeeze()
                cv.imwrite('./test/transformed_original_img_debug.jpg', img1)
                cv.imwrite('./test/transformed_attentioned_img_debug.jpg', a1)
                cv.imwrite('./test/transformed_mask_img_debug.jpg', mask1)

        # our input to the image should be the rgb, attentioned image, and segmentation mask, ie H x W X 7
        # x = torch.cat((image, attentioned_image, mask))
        x = torch.cat((attentioned_image, mask))
        return x, y_hat

    def find_corresponding_img(self, mask_path):
        mask_name = Path(mask_path).name
        if self.debug:
            cv.imwrite('./test/found_corresponding_color_img_debug.jpg', cv.imread(str(self.path_color_imgs / mask_name)))
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

    def get_attention_mask(self, mask, r):
        assert len(mask.shape) == 2, 'your mask should be some binary or grayscale image not color'
        kernel = np.ones((r,r),np.float32) / (r ** 2)
        blurred_img = cv.filter2D(mask, -1, kernel)
        if self.debug:
            cv.imwrite('./test/attention_mask_debug.jpg', blurred_img)
        return blurred_img
    
    def apply_attention(self, image, mask):
        assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1], 'dimensions of the image and mask should match'
        # an attention map should be [0, 1]
        mask = np.stack((mask / 255.0,)*3, axis=-1)
        result = image * mask
        if self.debug:
            cv.imwrite('./test/original_rgb_debug.jpg', image)
            cv.imwrite('./test/rgb_applied_attention_debug.jpg', result)
        return result.astype('uint8')