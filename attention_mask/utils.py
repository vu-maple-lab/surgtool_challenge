import cv2 as cv
import numpy as np
import os 
import torch 
from tqdm import tqdm
import random
from pathlib import Path
import glob
from natsort import natsorted
import matplotlib.pyplot as plt 

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

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def train_test_split(root_dir, split_val, debug):
    assert split_val < 1 and split_val > 0
    color_dir = root_dir / 'raw' / 'color'
    mask_dir = root_dir / 'raw' / 'processed_mask'
    save_dir = root_dir / 'actual'

    train_dir = save_dir / 'train'
    test_dir = save_dir / 'test'

    path_mask_imgs = glob.glob(str(mask_dir / '*.jpg'))
    random.shuffle(path_mask_imgs)
    train_set_cutoff = int(split_val * len(path_mask_imgs))

    for i, mask_path in enumerate(tqdm(path_mask_imgs)):
        # save it in train 
        if i < train_set_cutoff:
            # get the mask image and save it to train 
            save_path_mask = train_dir / 'mask' 
            command = 'cp ' + mask_path + ' ' + str(save_path_mask)
            os.system(command)
            if debug:
                print(f"command: {command}") 

            # get the color image and save it to train
            save_path_color = train_dir / 'color'
            color_img_path = color_dir / Path(mask_path).name 
            command = 'cp ' + str(color_img_path) + ' ' + str(save_path_color)
            os.system(command)

            if debug:
                print(f"command: {command}") 

        # save it in test
        else:
            save_path_mask = test_dir / 'mask' 
            command = 'cp ' + mask_path + ' ' + str(save_path_mask)
            os.system(command)
            if debug:
                print(f"command: {command}") 

            # get the color image and save it to train
            save_path_color = test_dir / 'color'
            color_img_path = color_dir / Path(mask_path).name 
            command = 'cp ' + str(color_img_path) + ' ' + str(save_path_color)
            os.system(command)

            if debug:
                print(f"command: {command}") 
                
    print('All done!')

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
            
def calc_accuracy(y, y_hat):
    batch_size, num_classes = y_hat.shape
    running_avg = 0.
    for i in range(batch_size):
        predictions = torch.round(torch.sigmoid(y[i,:]))
        running_avg += (torch.sum(predictions == y_hat[i,:])) / num_classes
    return running_avg / batch_size

def train_one_epoch(model, dataloader, loss, optim, scheduler, debug=False):
    
    # set model to train
    model.train()

    # get metrics of our current performance during training
    running_loss = 0.
    running_acc = 0.
    
    print("Training")
    for i, (x, y_hat) in enumerate(tqdm(dataloader)):
        x = x.to(device).float()
        y_hat = y_hat.to(device).float()

        optim.zero_grad()
        y = model(x)
        loss_val = loss(y, y_hat)
        loss_val.backward()
        optim.step()
        scheduler.step()
        running_loss += loss_val.item()
        running_acc += calc_accuracy(y, y_hat)
    
    return running_loss / (i+1), running_acc / (i+1)


def test_one_epoch(model, dataloader, loss, debug=False):
    # set model to evaluation
    model.eval()
    # get metrics of our current performance during training
    running_loss = 0.
    running_acc = 0.
    
    print("Testing")
    for i, (x, y_hat) in enumerate(tqdm(dataloader)):
        x = x.to(device).float()
        y_hat = y_hat.to(device).float()
        y = model(x)
        loss_val = loss(y, y_hat)
        running_loss += loss_val.item()
        running_acc += calc_accuracy(y, y_hat)
    
    return running_loss / (i+1), running_acc / (i+1)

def train(model, dataloaders, loss, optim, scheduler, epochs, logs_dir, debug=False):
    
    # initialize log vectors 
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # set model to cuda and initialize best model
    model = model.to(device)
    best_model = model 

    # start training loop
    for i in range(epochs):
        print(f"EPOCH: {i+1}/{epochs}")
        train_results = train_one_epoch(model, dataloaders[0], loss, optim, scheduler, debug)
        test_results = test_one_epoch(model, dataloaders[1], loss, debug)
        log_results(train_loss, train_acc, test_loss, test_acc, train_results, test_results, debug)
        if test_results[0] == min(test_loss):
            best_model = model 
    
    # save dicts
    torch.save(best_model.state_dict(), str(logs_dir / 'checkpoints' / 'best_model.pt'))
    torch.save(optim.state_dict(), str(logs_dir / 'checkpoints' / 'optimizer.pt'))
    torch.save(scheduler.state_dict(), str(logs_dir / 'checkpoints' / 'scheduler.pt'))
    return train_loss, train_acc, test_loss, test_acc


def log_results(train_loss, train_acc, test_loss, test_acc, train_results, test_results, debug=False):
    print(f'Training Loss: {train_results[0]} Testing Loss: {test_results[0]}')
    print(f'Training Accuracy: {train_results[1]} Testing Accuracy: {test_results[1]}')
    train_loss.append(train_results[0])
    train_acc.append(train_results[1])
    test_loss.append(test_results[0])
    test_acc.append(test_results[1])

def prepare_logs(logs_dir):
    if not (logs_dir / 'checkpoints').exists():
        os.mkdir(str(logs_dir / 'checkpoints'))
    if not (logs_dir / 'visualizations').exists():
        os.mkdir(str(logs_dir / 'visualizations'))
    if not (logs_dir / 'metrics').exists():
        os.mkdir(str(logs_dir / 'metrics'))

def save_vis(logs_dir, logs):

    # save numpy arrays
    train_loss, train_acc, test_loss, test_acc = logs 
    np.save(str(logs_dir / 'metrics' / 'training_loss.npy'), np.array(train_loss))
    np.save(str(logs_dir / 'metrics' / 'training_acc.npy'), np.array(train_acc))
    np.save(str(logs_dir / 'metrics' / 'testing_loss.npy'), np.array(test_loss))
    np.save(str(logs_dir / 'metrics' / 'testing_acc.npy'), np.array(test_acc))

    # training loss 
    plt.plot(np.array(logs[0]))
    plt.title('Training loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(str(logs_dir / 'visualizations' / 'training_loss.png'))

    # training accuracy 
    plt.plot(np.array(logs[1]))
    plt.title('Training Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.savefig(str(logs_dir / 'visualizations' / 'training_acc.png'))

    # testing loss 
    plt.plot(np.array(logs[2]))
    plt.title('Testing loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Loss')
    plt.savefig(str(logs_dir / 'visualizations' / 'testing_loss.png'))

    # testing accuracy
    plt.plot(np.array(logs[3]))
    plt.title('Testing Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy')
    plt.savefig(str(logs_dir / 'visualizations' / 'testing_acc.png'))