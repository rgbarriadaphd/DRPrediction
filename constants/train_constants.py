"""
# Author: ruben 
# Date: 22/4/22
# Project: DRPrediction
# File: train_constants.py

Description: Constants regarding train_unsplitted step and performance
"""
import sys

DATASET_APPROACH = 'B'

VALIDATION_SET= True
SAVE_PLOT = True

BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4e-2

SAVE_LOSS_PLOT = True
SAVE_ACCURACY_PLOT = True

DEBUG_BATCHES = sys.maxsize