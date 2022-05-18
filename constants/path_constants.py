"""
# Author: ruben 
# Date: 22/4/22
# Project: DRPrediction
# File: path_constants.py

Description: Constants regarding project relative addresses
"""
import os.path


DYNAMIC_RUN = 'input/diabetic_retinopathy_detection/train_dynamic_A'

TRAIN_DATASET_A = os.path.join('input/diabetic_retinopathy_detection/train_dynamic_A', 'train')
VALIDATION_DATASET_A = os.path.join('input/diabetic_retinopathy_detection/train_dynamic_A', 'validation')
TRAIN_DATASET_B = os.path.join('input/diabetic_retinopathy_detection/train_dynamic_B', 'train')
VALIDATION_DATASET_B = os.path.join('input/diabetic_retinopathy_detection/train_dynamic_B', 'validation')
TEST_DATASET = 'input/diabetic_retinopathy_detection/test_splitted'

DATASET_ARRANGE_A = 'input/diabetic_retinopathy_detection/train_A'
DATASET_ARRANGE_B = 'input/diabetic_retinopathy_detection/train_B'

OUTPUTS = 'outputs/'