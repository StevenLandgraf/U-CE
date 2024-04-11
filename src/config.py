"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import torch
torch.set_float32_matmul_precision('high')

import argparse

## Parser Arguments
parser = argparse.ArgumentParser(description='U-CE Parser')

parser.add_argument('--project', type=str, default='U-CE (Default)')
parser.add_argument('--run', type=str, default='U-CE (Default)')
parser.add_argument('--loss', type=str, default='UCE')          # Loss function to use ('CE', 'UCE')
parser.add_argument('--uce_exponent', type=float, default=1.0)  # U-CE Exponent
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dataset', type=str, default='Cityscapes')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--mode', type=str, default='MC_Dropout')
parser.add_argument('--model', type=str, default='deeplabv3plus')   
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--pretrained', type=str, default="True")
parser.add_argument('--use_scaling', type=bool, default=True)
parser.add_argument('--use_cropping', type=bool, default=True)
parser.add_argument('--use_flipping', type=bool, default=True)

args = parser.parse_args()

## W&B Logging
ENTITY = '---'
PROJECT = args.project
RUN_NAME = args.run

## U-CE Specific
UCE_EXPONENT = args.uce_exponent

## Model
MODE = args.mode             # 'MC_Dropout', 'Ensemble'
MODEL = args.model               # 'deeplabv3plus', 'unet'
ENCODER_NAME = args.backbone   # Add more encoders later
ENCODER_PRETRAINED = args.pretrained
ENSEMBLE_MODEL_PATH = './ensemble_models/'
ENSEMBLE_MODEL_NUMBER = 10

## Training + Hyperparameters
DEVICES = [0]      
NUM_EPOCHS = args.epochs        # Number of epochs to train for
LOSS = args.loss                # Loss function to use ('CE', 'UCE')
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_UNC_SAMPLES = 10                # Number of samples to draw from MC Dropout for uncertainty estimation
DROPOUT_PROBABILITY = args.dropout  # Dropout probability for MC Dropout
PRECISION = '16-mixed'

## Data Augmentations
USE_SCALING = args.use_scaling
USE_CROPPING = args.use_cropping
USE_FLIPPING = args.use_flipping

## Dataset
DATASET = args.dataset

if DATASET == 'Cityscapes':
    NUMBER_TRAIN_IMAGES = 2975
    NUMBER_VAL_IMAGES = 500
    LEARNING_RATE = 0.01
    BATCH_SIZE = 16
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    NUM_WORKERS = 8          

elif DATASET == 'ACDC':
    NUMBER_TRAIN_IMAGES = 1600
    NUMBER_VAL_IMAGES = 406
    LEARNING_RATE = 0.01
    BATCH_SIZE = 16
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    NUM_WORKERS = 8 