"""
# Author: ruben 
# Date: 10/5/22
# Project: DRPrediction
# File: make_dataset.py

Description: Create dataset based on folds
"""
import os.path
import pprint
from random import randrange
import shutil
from distutils.dir_util import copy_tree
import sys
import pandas as pd
from csv import reader, writer


def init_test():
    print("Creating test folder")
    train_folder = 'input/diabetic_retinopathy_detection/train_unsplitted/'
    csv_labels = 'input/diabetic_retinopathy_detection/trainLabels.csv'
    assert os.path.exists((train_folder))
    assert os.path.exists((csv_labels))

    # read csv file as a list of lists
    with open(csv_labels, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        images = list(map(tuple, csv_reader))
        # display all rows of csv
        # print(images)
        # print(len(images))

        # Check existance
        image_list = [elem[0] for elem in images]
        # print(image_list)
        not_present = []
        # for img_name in os.listdir(train_folder):
        #     # assert img_name in image_list, img_name
        #
        #     if img_name.split('.')[0] in image_list:
        #         continue
        #     not_present.append(img_name)
        # print(f'Incoherences between labels and images: {len(not_present)}')

        dataset_len = len(image_list)
        test_len = int(dataset_len * 0.20)

        # Create test folder and move the last 20% images
        test_folder = os.path.join(train_folder.split('train_unsplitted')[0], "test")
        print(f'Dataset length: ({dataset_len}). Taking 20% last images({test_len}) for test folder at {test_folder}')

        # # Create folder
        # if not os.path.exists(test_folder):
        #     os.makedirs(test_folder)
        #
        # csv_test_labels = 'input/diabetic_retinopathy_detection/testLabels.csv'
        # with open(csv_test_labels, 'w') as out:
        #     csv_out = writer(out)
        #
        #     for image in images[-test_len:]:
        #         # Move files
        #         source = os.path.join(train_folder, f'{image[0]}.jpeg')
        #         target = os.path.join(test_folder, f'{image[0]}.jpeg')
        #         shutil.move(source, target)
        #
        #         # Create test csv
        #         csv_out.writerow(image)
        #         print(f'Moving image from {source} to {target} and writing {image} at csv file')
        #
        #         # Remove

        print(f'Train dataset: {len(os.listdir(train_folder))}')
        print(f'Test dataset: {len(os.listdir(test_folder))}')

    # with open(csv_labels, 'w') as out:
    #     csv_out = writer(out)
    #     for image in images[0:-test_len-1]:
    #         # Remove
    #         csv_out.writerow(image)

    with open(csv_labels, 'r') as read_obj:
        csv_reader = reader(read_obj)
        images = list(map(tuple, csv_reader))
        image_list = [elem[0] for elem in images]
        for img_name in os.listdir(train_folder):
            assert img_name.split('.')[0] in image_list, (img_name.split('.')[0], train_folder)

    with open('input/diabetic_retinopathy_detection/testLabels.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        images = list(map(tuple, csv_reader))
        image_list = [elem[0] for elem in images]
        for img_name in os.listdir(test_folder):
            assert img_name.split('.')[0] in image_list, (img_name.split('.')[0], test_folder)


def arrange_train():
    train_unsplitted_folder = 'input/diabetic_retinopathy_detection/train_unsplitted/'
    train_splitted_folder = 'input/diabetic_retinopathy_detection/train_splitted/'
    csv_labels = 'input/diabetic_retinopathy_detection/trainLabels.csv'

    # # Copy images from usplited to unsplited trani folder (original)
    # for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
    #     for img in os.listdir(os.path.join(train_splitted_folder, fold)):
    #         shutil.copy(os.path.join(train_splitted_folder, fold, img), train_unsplitted_folder)

    # # Create classes folder
    # for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
    #     fold_folder = os.path.join(train_splitted_folder, fold)
    #     os.makedirs(os.path.join(fold_folder, 'class_0'))
    #     os.makedirs(os.path.join(fold_folder, 'class_1'))
    #     os.makedirs(os.path.join(fold_folder, 'class_2'))
    #     os.makedirs(os.path.join(fold_folder, 'class_3'))
    #     os.makedirs(os.path.join(fold_folder, 'class_4'))

    # # Create classes folder
    # csv_labels = 'input/diabetic_retinopathy_detection/trainLabels.csv'
    # labels = get_csv_dict(csv_labels)
    # for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
    #     fold_folder = os.path.join(train_splitted_folder, fold)
    #     for image in os.listdir(fold_folder):
    #         if not image.endswith('.jpeg'):
    #             continue
    #
    #         src = os.path.join(fold_folder, image)
    #         if labels[image.split('.')[0]] == '0':
    #             dst = os.path.join(fold_folder, 'class_0', image)
    #         elif labels[image.split('.')[0]] == '1':
    #             dst = os.path.join(fold_folder, 'class_1', image)
    #         elif labels[image.split('.')[0]] == '2':
    #             dst = os.path.join(fold_folder, 'class_2', image)
    #         elif labels[image.split('.')[0]] == '3':
    #             dst = os.path.join(fold_folder, 'class_3', image)
    #         elif labels[image.split('.')[0]] == '4':
    #             dst = os.path.join(fold_folder, 'class_4', image)
    #         else:
    #             print("Wrong type")
    #             continue
    #         shutil.move(src, dst)



    # Create option A: duplicate minoritray classes
    train_A = 'input/diabetic_retinopathy_detection/train_A/'

    n_samples_fold = {}
    n_target = {}
    m_class = {}
    n_target_less = {}
    m_class_less = {}
    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
        most = (-1, None)
        n_samples_fold[fold] = {}
        n_target[fold] = None
        m_class[fold] = None

        less = (sys.maxsize, None)
        n_target_less[fold] = None
        m_class_less[fold] = None
        fold_folder = os.path.join(train_splitted_folder, fold)
        for cls in ['class_0', 'class_1','class_2','class_3','class_4']:
            n_samples = len(os.listdir(os.path.join(fold_folder, cls)))
            n_samples_fold[fold][cls] = n_samples

            if n_samples > most[0]:
                most = n_samples, cls

            if n_samples < less[0]:
                less = n_samples, cls

            n_target[fold] = most[0]
            m_class[fold] = most[1]
            n_target_less[fold] = less[0]
            m_class_less[fold] = less[1]


    # for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
    #
    #     # Copy mayor class as is in train_A
    #     src = os.path.join(train_splitted_folder,fold,m_class[fold])
    #     dst = os.path.join(train_A,fold,m_class[fold])
    #     print(f'{src} --> {dst}')
    #     copy_tree(src, dst)
    #
    #     # For the rest of the clases, copy as is and then randomly copy images till n_target is reached
    #
    #     for cls in ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']:
    #         if cls == m_class[fold]:
    #             continue
    #
    #         # Copy as is
    #         src = os.path.join(train_splitted_folder, fold, cls)
    #         dst = os.path.join(train_A, fold, cls)
    #         copy_tree(src, dst)
    #
    #         # Generate random copies
    #         image_list = os.listdir(os.path.join(train_splitted_folder, fold, cls))
    #         count = len(image_list)
    #         while count < n_target[fold]:
    #             pos = randrange(0, len(image_list) - 1)
    #             selected_image = image_list[pos]
    #             new_image_name = f'{selected_image.split(".")[0]}_{count}.jpeg'
    #             src = os.path.join(train_splitted_folder, fold, cls, selected_image)
    #             dst = os.path.join(train_A, fold, cls, new_image_name)
    #             shutil.copy(src, dst)
    #             count += 1
    #
    # # Print result
    # for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
    #     for cls in ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']:
    #         print(f'[{fold}] Samples in class {cls}: {len(os.listdir(os.path.join(train_A, fold, cls)))} samples')
    #     print("-----")

    # pprint.pprint(n_samples_fold)
    # pprint.pprint(n_target_less)
    # pprint.pprint(m_class_less)


    # Create option B: remove samples from mayor cases
    train_B = 'input/diabetic_retinopathy_detection/train_B/'

    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:

        # Copy minor class as is in train_A
        src = os.path.join(train_splitted_folder, fold, m_class_less[fold])
        dst = os.path.join(train_B, fold, m_class_less[fold])
        print(f'{src} --> {dst}')
        copy_tree(src, dst)

        # For the rest of the clases, copy as is and then randomly remove images till n_target is reached

        for cls in ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']:
            if cls == m_class_less[fold]:
                continue

            # Copy as is
            src = os.path.join(train_splitted_folder, fold, cls)
            dst = os.path.join(train_B, fold, cls)
            copy_tree(src, dst)

            # Generate random copies
            image_list = os.listdir(os.path.join(train_B, fold, cls))
            count = len(image_list)
            while count > n_target_less[fold]:
                pos = randrange(0, len(image_list) - 1)
                selected_image = image_list[pos]
                os.remove(os.path.join(train_B, fold, cls, selected_image))
                image_list = os.listdir(os.path.join(train_B, fold, cls))
                count -= 1

        # Print result
    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
        for cls in ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']:
            print(f'[{fold}] Samples in class {cls}: {len(os.listdir(os.path.join(train_B, fold, cls)))} samples')
        print("-----")

def split_test_in_class():
    test_folder = 'input/diabetic_retinopathy_detection/test/'
    test_splitted_folder = 'input/diabetic_retinopathy_detection/test_splitted/'

    test_csv = get_csv_dict('input/diabetic_retinopathy_detection/testLabels.csv')
    print(test_csv)

    for image in os.listdir(test_folder):

        src = os.path.join(test_folder, image)
        if test_csv[image.split('.')[0]] == '0':
            dst = os.path.join(test_splitted_folder, 'class_0', image)
        elif test_csv[image.split('.')[0]] == '1':
            dst = os.path.join(test_splitted_folder, 'class_1', image)
        elif test_csv[image.split('.')[0]] == '2':
            dst = os.path.join(test_splitted_folder, 'class_2', image)
        elif test_csv[image.split('.')[0]] == '3':
            dst = os.path.join(test_splitted_folder, 'class_3', image)
        elif test_csv[image.split('.')[0]] == '4':
            dst = os.path.join(test_splitted_folder, 'class_4', image)
        else:
            print("Wrong type")
            continue
        shutil.copy(src, dst)


def clean_folder():
    folder = 'input/diabetic_retinopathy_detection/train_B/'
    for root, dirs, files in os.walk(folder):
        path = root.split(os.sep)
        for file in files:
            if file.endswith('.jpeg'):
                os.remove(os.path.join(root, file))




def main(args):
    print(args)
    if args[1] == '-t':
        init_test()

    if args[1] == '-o':
        arrange_train()

    if args[1] == '-c':
        clean_folder()

    if args[1] == '-v':
        split_test_in_class()


def get_csv_dict(csv_file):
    with open(csv_file, mode='r') as infile:
        rd = reader(infile)
        mydict = {rows[0]: rows[1] for rows in rd}

    return mydict
if __name__ == '__main__':
    args = sys.argv
    args.append("-v")
    main(args)

