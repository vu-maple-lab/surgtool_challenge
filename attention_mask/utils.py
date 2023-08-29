import cv2 as cv
import numpy as np
import os 
from tqdm import tqdm
from pathlib import Path
import glob
from natsort import natsorted

TOOLS_ONE_HOT_ENCODING = {
    'needle driver': 0,
    'monopolar curved scissors': 1, 
    'force bipolar': 2,
    'clip applier': 3, 
    'tip-up fenestrated grasper': 4,
    'cadiere forceps': 5, 
    'bipolar forceps': 6,
    'vessel sealer': 7,
    'suction irrigator': 8,
    'bipolar dissector': 9,
    'prograsp forceps': 10,
    'stapler': 11,
    'permanent cautery hook/spatula': 12,
    'grasping retractor': 13
}

# process_binary reads in mask image to denoise and encourage connectivity
# mask: grayscale image that represents the mask 
# debug: boolean option to save intermediate images
# returns: np array with size mask, denoised binary image
def process_binary(mask, debug):

    # first first, convert our mask to an actual binary image
    _, bin_img = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

    # next we want to do morphological dilation because some of the tools 
    # are disconnected in the mask
    SE_dilation = np.ones((10,10), np.uint8)
    dilated_img = cv.dilate(bin_img, SE_dilation, iterations=1)

    # then we want to perform some sort of denoising to get rid of small speckles. 
    # they might be large though because of the dilation we performed
    SE_opening = np.ones((20,20), np.uint8)
    opened_img = cv.morphologyEx(dilated_img, cv.MORPH_OPEN, SE_opening)

    # finally, let's do a closing to make sure that there aren't any holes
    # in our connected components
    SE_closing = np.ones((20,20), np.uint8)
    closed_img = cv.morphologyEx(opened_img, cv.MORPH_CLOSE, SE_closing)

    # finally convert it to grayscale so that we have 1 channel
    grayscale = np.uint8(np.dot(closed_img[...,:3], [0.2989, 0.5870, 0.1140]))
    _, result = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY)

    # save intermediate images to a file for testing
    if debug:
        print('Testing! Saving files to ./test')
        if not os.path.exists('./test'):
            os.mkdir('./test')
        cv.imwrite('./test/1original_img.jpg', bin_img)
        cv.imwrite('./test/2dilated_img.jpg', dilated_img)
        cv.imwrite('./test/3opened_img.jpg', opened_img)
        cv.imwrite('./test/4closed_img.jpg', closed_img)
        cv.imwrite('./test/5result.jpg', grayscale)
    
    return result

def filter_segmentations(root_dir, debug):
    color_dir = root_dir / 'raw' / 'color'
    mask_dir = root_dir / 'raw' / 'mask'
    save_dir = root_dir / 'raw' / 'processed_mask'
    
    if not ((color_dir).exists() and (mask_dir).exists()):
        raise Exception("Your input_dir must include a labels.csv and a raw/ file with color/ and mask/ inside")
        
    path_color_imgs = natsorted(glob.glob(str(color_dir / '*.jpg')))
    path_mask_imgs = natsorted(glob.glob(str(mask_dir / '*.jpg')))

    assert len(path_color_imgs) == len(path_mask_imgs), 'num of mask imgs and color imgs should match'

    # create a folder to move all our bad segmentations into
    trash_dir = root_dir / 'bad'
    if not trash_dir.exists():
        os.mkdir(str(trash_dir))
    
    # the name of the game is to filter out bad segmentations. the definition of a "bad segmentation" 
    # is pretty arbitrary but i've sort of defined a meaningful segmentation as the following:
    # 1) a "blob" or a connected component should be at least 2% of the entire image in terms of pixel area
    # 2) if the total number of positive pixels are more than 40% of the entire image, then 
    #    we can assume that the segmentation didn't do a great job. 

    for i, mask_path in enumerate(tqdm(path_mask_imgs)):

        if debug:
            print(f'Our mask path: {mask_path}')
            mask_path = '../data/raw/mask/011124_1559.jpg'
        img = cv.imread(mask_path)
        denoised_img = process_binary(img, debug)

        # check how large it is
        total_area = img.shape[0] * img.shape[1]
        white_pixel_area = np.sum(denoised_img == 255)
        if white_pixel_area > 0.4 * total_area:
            # mv it to our trash folder
            mv_command = 'mv ' + mask_path + ' ' + str(trash_dir) + '/'
            if debug:
                print(f'{mask_path} sucks so we are skipping it')
                print(f'our move command is: {mv_command}')
                os.system(mv_command)
                break
            os.system(mv_command)
            continue  
        
        # connected component analysis
        num_labels, labeled_img, stats, _ = cv.connectedComponentsWithStats(denoised_img)

        # filter out the labels 
        useful_labels = []

        for label in range(num_labels):
            # skip background
            if label == 0:
                continue
            
            # get the area of connected component
            component_area = stats[label, 4]

            # assert that a useful component should be at least 2% of the area of 
            # the entire image. that being said
            if component_area > 0.02 * total_area:
                useful_labels.append(label)

        # create a new mask with only useful labels
        new_mask = np.zeros_like(denoised_img)
        for label in useful_labels:
            new_mask[labeled_img == label] = 255 
        
        # save our new mask
        save_name = str(save_dir / Path(mask_path).name)
        cv.imwrite(save_name, new_mask)
        
        if debug:
            print(f'The useful labels are: {useful_labels}')
            cv.imwrite('./test/new_mask_debug.jpg', new_mask)
            break 
            
        