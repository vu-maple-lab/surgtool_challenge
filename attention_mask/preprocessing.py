import argparse 
from utils import preprocess, filter_segmentations, train_test_split
from pathlib import Path
import os

def main(args):
    # should be the endovis23 directory 
    data_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    debug = args.debug

    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if not model_dir.exists():
        raise Exception("model path does not exist.")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')
        
    print(f'Preprocessing: generating binary masks and saving color images...')
    preprocess(data_dir, model_dir, debug)
    print(f'Generating Train-Test split for Training purposes...')
    train_test_split(data_dir, 0.7, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)