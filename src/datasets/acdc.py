"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import os
import glob
import random
import math

import torch
import numpy as np
import lightning as L
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import config


class ACDCDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = ACDCDataset('train', ['fog', 'night', 'rain', 'snow'])
        self.val_dataset = ACDCDataset('val', ['fog', 'night', 'rain', 'snow'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)
    

class ACDCDataset(Dataset):
    def __init__(self, split, condition_list: list):
        self.root_dir = './data/ACDC'
        self.split = split
        self.condition_list = condition_list

        self._load_image_and_label_paths()

        self.class_conversion_dict = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
        self.ignore_index = 255


    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])

        image = TF.to_tensor(image)
        label = np.array(label)

        # Label encoding
        label_encoded = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.class_conversion_dict.items():
            label_encoded[label == k] = v

        label_encoded = torch.tensor(label_encoded, dtype=torch.long).unsqueeze(0)

        if self.split == 'train':
            # Random Resize
            if config.USE_SCALING:
                random_scaler = RandResize(scale=(0.5, 2.0))
                image, label_encoded = random_scaler(image.unsqueeze(0).float(), label_encoded.unsqueeze(0).float())

                # Pad image if it's too small after the random resize
                if image.shape[1] < 768 or image.shape[2] < 768:
                    height, width = image.shape[1], image.shape[2]
                    pad_height = max(768 - height, 0)
                    pad_width = max(768 - width, 0)
                    pad_height_half = pad_height // 2
                    pad_width_half = pad_width // 2

                    border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                    image = F.pad(image, border, 'constant', 0)
                    label_encoded = F.pad(label_encoded, border, 'constant', self.ignore_index)

            # Random Horizontal Flip
            if config.USE_FLIPPING:
                if random.random() < 0.5:
                    image = TF.hflip(image)
                    label_encoded = TF.hflip(label_encoded)

            # Random Crop
            if config.USE_CROPPING:
                i, j, h, w = transforms.RandomCrop(size=(768, 768)).get_params(image, output_size=(768, 768))
                image = TF.crop(image, i, j, h, w)
                label_encoded = TF.crop(label_encoded, i, j, h, w) 

        elif self.split == 'val':
            # crop image and label to 1072x1920
            image = TF.crop(image, 0, 0, 1072, 1920)
            label_encoded = TF.crop(label_encoded, 0, 0, 1072, 1920)

        return image, label_encoded


    def __len__(self):
        return len(self.images)


    def _load_image_and_label_paths(self):
        self.images = []
        self.labels = []

        for condition in self.condition_list:
            sequences = os.listdir(os.path.join(self.root_dir, 'rgb_anon', condition, self.split))
            for sequence in sequences:
                self.images += sorted(glob.glob(os.path.join(self.root_dir, 'rgb_anon', condition, self.split, sequence, '*.png')))
                self.labels += sorted(glob.glob(os.path.join(self.root_dir, 'gt', condition, self.split, sequence, '*labelIds.png')))


class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (
                self.aspect_ratio[0]
                + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            )
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = F.interpolate(
            image, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")
        return image.squeeze(), label.squeeze(0).to(dtype=torch.int64)