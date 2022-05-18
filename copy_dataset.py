"""
# Author: ruben 
# Date: 4/5/22
# Project: DRPrediction
# File: copy_dataset.py

Description: Copy original dataset to local workspace and resize img to 224x224
"""
import os
import shutil
import sys

TARGET_FOLDER = 'input/diabetic_retinopathy_detection/'
from PIL import Image

# bytes pretty-printing
UNITS_MAPPING = [
    (1<<50, ' PB'),
    (1<<40, ' TB'),
    (1<<30, ' GB'),
    (1<<20, ' MB'),
    (1<<10, ' KB'),
    (1, (' byte', ' bytes')),
]


def pretty_size(bytes, units=UNITS_MAPPING):
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def main(args):
    source = args[1]
    target = TARGET_FOLDER
    print(f'from {source} to {target}')
    assert os.path.exists(os.path.join(source, 'train_unsplitted'))
    assert os.path.exists(os.path.join(source, 'test'))
    assert os.path.exists(os.path.join(source, 'trainLabels.csv'))

    print(f'** Source original size = {pretty_size(get_size(source))} ')

    print("# copy train_unsplitted folder")
    total = len(os.listdir(os.path.join(source, 'train_unsplitted')))
    processed = 1
    for img_file in os.listdir(os.path.join(source, 'train_unsplitted')):
        print(f'Processing train_unsplitted image ({processed}/{total})')
        processed += 1
        with Image.open(os.path.join(source, 'train_unsplitted', img_file)) as im:
            width, height = im.size
            center = (width / 2, height / 2)
            im = im.crop((center[0] - (height / 2), 0, center[0] + (height / 2), height))
            im = im.resize((224, 224))
            im.save(os.path.join(target, 'train_unsplitted', img_file), "jpeg")

    print("# copy test folder")
    total = len(os.listdir(os.path.join(source, 'test')))
    processed = 1
    for img_file in os.listdir(os.path.join(source, 'test')):
        print(f'Processing test image ({processed}/{total})')
        processed += 1
        with Image.open(os.path.join(source, 'test', img_file)) as im:
            width, height = im.size
            center = (width / 2, height / 2)
            im = im.crop((center[0] - (height / 2), 0, center[0] + (height / 2), height))
            im = im.resize((224, 224))
            im.save(os.path.join(target, 'test', img_file), "jpeg")

    # copy CSV labels
    print("# copy CSV labels")
    shutil.copyfile(os.path.join(source, 'trainLabels.csv'), os.path.join(target, 'trainLabels.csv'))

    print(f'** Source original size = {pretty_size(get_size(target))} ')

if __name__ == '__main__':
    args = sys.argv
    main(args)

