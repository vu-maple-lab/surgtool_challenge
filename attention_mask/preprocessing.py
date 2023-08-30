import argparse 
from utils import filter_segmentations, train_test_split
from pathlib import Path
import os

def main(args):
    data_dir = Path(args.input_dir)
    debug = args.debug

    if not data_dir.exists():
        raise Exception("data directory does not exist")
    if debug and not os.path.exists('./test'):
        os.mkdir('./test')

    # filter_segmentations(data_dir, debug)
    train_test_split(data_dir, 0.7, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)