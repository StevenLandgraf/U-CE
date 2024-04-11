"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class UCE(nn.Module):
    def __init__(self, ignore_index=255):
        super(UCE, self).__init__()
        self.ignore_index = ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, logits, targets, predictive_uncertainty):
        batch_size, _, height, width = logits.size()

        # compute standard ce-loss without reduction
        ce_loss = self.ce_criterion(logits, targets)    # shape: [batch_size, height, width]

        # get number of ignore_index pixels
        ignore_index_mask = (targets == self.ignore_index)
        ignore_index_count = torch.sum(ignore_index_mask)

        # apply uncertainty weighting
        loss = (1 + predictive_uncertainty)**config.UCE_EXPONENT * ce_loss
            
        return (torch.sum(loss) / (batch_size * height * width - ignore_index_count))