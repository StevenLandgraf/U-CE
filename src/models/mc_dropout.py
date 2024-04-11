"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchmetrics
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
from segmentation_models_pytorch.encoders.resnet import (ResNetEncoder, pretrained_settings)
from torchvision.models.resnet import BasicBlock, Bottleneck

import config
from utils.losses import UCE
from utils.lr_scheduler import PolyLR


class MC_Dropout(L.LightningModule):
    def __init__(self):
        super().__init__()

        # Initialize the model
        self.model = MCDropoutDeepLabV3Plus()

        # Initialize the loss function
        if config.LOSS == 'CE':
            self.criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        elif config.LOSS == 'UCE':
            self.criterion == UCE(ignore_index=config.IGNORE_INDEX)

        # Initialize the optimizer
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.encoder.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decoder.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        # Initialize the metrics
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def training_step(self, batch, batch_index):
        images, labels = batch

        # Forward Pass
        logits = self.model(images)

        # Calculate Loss
        if config.LOSS == 'CE':
            loss = self.criterion(logits, labels.squeeze())

        elif config.LOSS == 'UCE':
            predictive_uncertainty = self._get_predictive_uncertainty(images)
            loss = self.criterion(logits, labels.squeeze(), predictive_uncertainty)

        # Calculate Metrics
        self.train_iou(torch.softmax(logits, dim=1), labels.squeeze())

        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss                   


    def validation_step(self, batch, batch_index):
        images, labels = batch

        # Forward Pass
        logits = self.model(images)

        # Calculate Metrics
        self.val_iou(torch.softmax(logits, dim=1), labels.squeeze())

        # Logging
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    @torch.no_grad()
    def _get_predictive_uncertainty(self, images):
        # Sample config.NUM_UNC_SAMPLES times from the model with dropout
        model_outputs = torch.empty(size=[config.NUM_UNC_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[2], images.shape[3]], device=self.device)        
        for i in range(config.NUM_UNC_SAMPLES):
            model_outputs[i] = self.model(images)

        # Compute the predictive uncertainty
        probability_map = torch.mean(torch.softmax(model_outputs, dim=2), dim=0)
        uncertainty_map = torch.std(torch.softmax(model_outputs, dim=2), dim=0)
        prediction_map = torch.argmax(probability_map, dim=1)

        predictive_uncertainty = torch.empty(size=[images.shape[0], images.shape[2], images.shape[3]], device=self.device)
        for i in range(config.NUM_CLASSES):
            predictive_uncertainty = torch.where(prediction_map == i, uncertainty_map[:, i, :, :], predictive_uncertainty)

        return predictive_uncertainty   


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


class ResNetEncoderMCDropout(ResNetEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=config.DROPOUT_PROBABILITY)

    def forward(self, x):
        features = []   # List of features from each layer
        features.append(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer2(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer3(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer4(x)
        x = self.dropout(x)
        features.append(x)

        return features


class MCDropoutDeepLabV3Plus(DeepLabV3Plus):
    def __init__(self):
        if config.ENCODER_PRETRAINED == 'True':
            weights = 'imagenet'
        else:
            weights = None
        
        super().__init__(
            encoder_name = config.ENCODER_NAME,
            encoder_depth = 5,
            encoder_weights = weights,
            encoder_output_stride = 16,
            decoder_channels = 256,
            decoder_atrous_rates = (12, 24, 36),
            in_channels = 3,
            classes = config.NUM_CLASSES,
            activation = None,
            upsampling = 4,
            aux_params = None,
        )
        
        # ResNet18 Params
        if config.ENCODER_NAME == 'resnet18':
            params = {
                'out_channels': (3, 64, 64, 128, 256, 512),
                'block': BasicBlock,
                'layers': [2, 2, 2, 2],
                'depth': 5,
            }
        
        # ResNet101 Params
        elif config.ENCODER_NAME == 'resnet101':
            params = {
                "out_channels": (3, 64, 256, 512, 1024, 2048),
                "block": Bottleneck,
                "layers": [3, 4, 23, 3],
                'depth': 5,
            }

        # Overwrite encoder with MCDropout version
        self.encoder = ResNetEncoderMCDropout(**params)
        self.encoder.set_in_channels(3)
        self.encoder.make_dilated(16)
        if config.ENCODER_NAME == 'resnet18' and config.ENCODER_PRETRAINED == 'True':
            self.encoder.load_state_dict(model_zoo.load_url(pretrained_settings['resnet18']['imagenet']['url']))
        elif config.ENCODER_NAME == 'resnet101' and config.ENCODER_PRETRAINED == 'True':
            self.encoder.load_state_dict(model_zoo.load_url(pretrained_settings['resnet101']['imagenet']['url']))