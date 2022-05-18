"""
# Author: ruben 
# Date: 22/4/22
# Project: DRPrediction
# File: dr_main.py

Description: Main script to run project options
"""
import os
import shutil
from distutils.dir_util import copy_tree
import os
from metrics import PerformanceMetrics
from PIL import Image, ImageStat
import numpy as np
import logging
import torch
from torchvision import datasets, transforms
from constants.path_constants import *
from constants.train_constants import *
import logging
import torch
import time
from copy import copy, deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from string import Template

import logging
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim

import ssl

from constants.path_constants import *
from constants.train_constants import *


def generate_dynamic_run():
    # TODO: temp. fix according parametrization
    return
    # Copy images
    src_folder = DATASET_ARRANGE_A if DATASET_APPROACH == 'A' else DATASET_ARRANGE_B
    # n_samples_required = 0
    # for fold in os.listdir(src_folder):
    #     for cls in os.listdir(os.path.join(src_folder, fold)):
    #         n_samples_required+= len(os.listdir(os.path.join(src_folder, fold, cls)))
    #
    # n_samples_deployed = 0
    # deployed_folder = DYNAMIC_RUN
    # for cls in os.listdir(os.path.join(DYNAMIC_RUN, 'train')):
    #         n_samples_deployed += len(os.listdir(os.path.join(DYNAMIC_RUN, 'train', cls)))
    #
    # print(f'Number of samples required : {n_samples_required}')
    # print(f'Number of samples deployed : {n_samples_deployed}')
    # print('validation' in os.listdir(DYNAMIC_RUN) and VALIDATION_SET)
    # print(not 'validation' in os.listdir(DYNAMIC_RUN) and not VALIDATION_SET)
    # print(n_samples_deployed == n_samples_required)
    # if ('validation' in os.listdir(DYNAMIC_RUN) and VALIDATION_SET) or (
    #         not 'validation' in os.listdir(DYNAMIC_RUN) and not VALIDATION_SET) and (
    #         n_samples_deployed == n_samples_required):
    #     print('dataset already generated')
    #     return
    # Clean dynamic folder
    shutil.rmtree(DYNAMIC_RUN)

    # Create class folders
    cls_list = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    os.makedirs(os.path.join(DYNAMIC_RUN, 'train'))
    for cls in cls_list:
        os.makedirs(os.path.join(DYNAMIC_RUN, 'train', cls))

    if VALIDATION_SET:
        os.makedirs(os.path.join(DYNAMIC_RUN, 'validation'))
        for cls in cls_list:
            os.makedirs(os.path.join(DYNAMIC_RUN, 'validation', cls))



    fold_list = ['fold_1', 'fold_2', 'fold_3'] if VALIDATION_SET else ['fold_1', 'fold_2', 'fold_3', 'fold_4']

    for fold in fold_list:
        for cls in cls_list:
            src = os.path.join(src_folder, fold, cls)
            dst = os.path.join(DYNAMIC_RUN, 'train', cls)
            copy_tree(src, dst)

    if VALIDATION_SET:
        for cls in cls_list:
            src = os.path.join(src_folder, 'fold_4', cls)
            dst = os.path.join(DYNAMIC_RUN, 'validation', cls)
            copy_tree(src, dst)

    print("Train dataset generated.")


def get_model():
    model = models.vgg16(pretrained=True)

    torch.manual_seed(3)

    # Freeze trained weights
    for param in model.features.parameters():
        param.requires_grad = True

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    linear = nn.Linear(num_features, 5)

    features.extend([linear])  # Add our layer with 2 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    return model


def save_loss_plot(measures, plot_type='loss'):
    print(measures)
    fig, ax = plt.subplots()

    tx = f'Model = {"vgg16"}\nEpochs = {EPOCHS}\nBatch size = {BATCH_SIZE}\nLearning rate = {LEARNING_RATE}'

    ax.plot(list(range(1, len(measures) + 1)), measures)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.72, 0.95, tx, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel(plot_type)
    plt.xlabel('epochs')
    plt.title(f'{plot_type} evolution')
    plot_path = os.path.join(OUTPUTS, f'loss.png')
    print(f'Saving plot to {plot_path}')
    plt.savefig(plot_path)


def save_accuracies_plot(train, validation, test):
    assert len(train) == len(validation)
    print(train)
    print(validation)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    plt.title(f'Model accuracy')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    x_epochs = list(range(1, len(validation) + 1))
    ax1.plot(x_epochs, validation, label='validation')
    ax1.plot(x_epochs, train, label='train')
    ax1.plot(x_epochs, test, label='test')
    ax1.legend()

    plot_path = os.path.join(OUTPUTS, f'accuracy_plots.png')
    print(f'Saving accuracy plot to {plot_path}')
    plt.savefig(plot_path)


def load_data(dataset, test=False):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(dataset, transform=data_transforms)
    print(f'Classes in {dataset}: {image_datasets.class_to_idx}')
    bs = 1 if test else BATCH_SIZE
    print(dataset, bs)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=bs, shuffle=True, num_workers=4)

    return data_loader


def train_model(model, device, train_loader, test_loader, validation_loader=None):
    n_train = len(train_loader.dataset)
    print(f'''Starting training:
            Epochs:          {EPOCHS}
            Batch size:      {BATCH_SIZE}
            Learning rate:   {LEARNING_RATE}
            Training size:   {n_train}
            Device:          {device.type}
        ''')

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train(True)
        running_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                if i > DEBUG_BATCHES:
                    break
                    
                sample, ground = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                current_batch_size = sample.size(0)
                optimizer.zero_grad()
                prediction = model(sample)

                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * current_batch_size

                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])
        epoch_loss = running_loss / n_train
        losses.append(epoch_loss)

        if SAVE_PLOT:
            print('Evaluate train/validation dataset accuracy')
            tr_a = evaluate_model(model, train_loader, device)
            train_accuracies.append(tr_a)
            t_a = evaluate_model(model, test_loader, device)
            test_accuracies.append(t_a)
            if VALIDATION_SET:
                v_a = evaluate_model(model, validation_loader, device)
                val_accuracies.append(v_a)


            torch.save(model.state_dict(), 'partials/partial_model.pt')

            with open('partials/partial_model.accuracy', 'w') as f_out:
                f_out.write(f'Train process at epoch: {epoch}')
                f_out.write(f'  Train accuracy: {tr_a}')
                f_out.write(f'  Validation accuracy: {v_a}')
                f_out.write(f'  Test accuracy: {t_a}')

        print(f'EPOCH time: {time.time() - t0}')

    return model, losses, train_accuracies, val_accuracies,test_accuracies


def evaluate_model(model, test_loader, device, metrics=False):
    n_test = len(test_loader.dataset)
    print(f'''Starting tesing:
            Test size:  {n_test}
            Device:     {device.type}
        ''')

    correct = 0
    total = 0
    ground_array = []
    prediction_array = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i > DEBUG_BATCHES:
                break
            sample, ground = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            ground_array += [elem.item() for elem in ground]
            prediction_array += [elem.item() for elem in predicted]

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

    if metrics:
        pm = PerformanceMetrics(ground_array,
                                prediction_array,
                                output_folder='outputs/',
                                classes=['class_0', 'class_1', 'class_2', 'class_3', 'class_4'],
                                percent=True,
                                formatted=2
                                )

    return (100 * correct) / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    generate_dynamic_run()

    t0 = time.time()

    print('Get model')
    model = get_model()
    model.to(device=device)

    print('Train data loader')
    train_dataset = TRAIN_DATASET_A if DATASET_APPROACH == 'A' else TRAIN_DATASET_B
    validation_dataset = VALIDATION_DATASET_A if DATASET_APPROACH == 'A' else VALIDATION_DATASET_B
    train_dataloader = load_data(train_dataset)
    test_dataloader = load_data(TEST_DATASET)
    validation_dataloader = None if not VALIDATION_SET else load_data(validation_dataset)

    print("TRAIN MODEL")
    print("-------------")

    model, losses, train_accuracy, val_accuracy, test_accuracy = train_model(model=model,
                                                                             device=device,
                                                                             train_loader=train_dataloader,
                                                                             test_loader=test_dataloader,
                                                                             validation_loader=validation_dataloader)

    print("TEST MODEL")
    print("-------------")
    model_accuracy = evaluate_model(model, test_dataloader, device, metrics=True)
    print(f'Model accuracy: : {model_accuracy}')

    with open(os.path.join(OUTPUTS, 'accuracy'), 'w') as f_out:
        f_out.write(f'Model accuracy: {model_accuracy}')

    if SAVE_LOSS_PLOT:
        print("Plot loss")
        save_loss_plot(losses, plot_type='loss')

    # Generate Loss plot
    if SAVE_ACCURACY_PLOT:
        print("Plot accuracies")
        save_accuracies_plot(train_accuracy, val_accuracy, test_accuracy)


if __name__ == '__main__':
    main()
