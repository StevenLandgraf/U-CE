"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import os
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


class CityscapesDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = CityscapesDataset('train')
        self.val_dataset = CityscapesDataset('val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.batch_size / 2), num_workers=self.num_workers, drop_last=False)
    

class CityscapesDataset(Dataset):
    def __init__(self, split):
        self.root_dir = './data/cityscapes'
        self.split = split
        
        self.images = []
        self.labels = []

        self._load_image_and_label_paths()

        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]     # classes: 19
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])

        image = TF.to_tensor(image)
        label = np.array(label)

        # Label Encoding
        for void_class in self.void_classes:
            label[label == void_class] = self.ignore_index
        for valid_class in self.valid_classes:
            label[label == valid_class] = self.class_map[valid_class]

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)

        if self.split == 'train':
            # Random Resize
            if config.USE_SCALING:
                random_scaler = RandResize(scale=(0.5, 2.0))
                image, label = random_scaler(image.unsqueeze(0).float(), label.unsqueeze(0).float())

                # Pad image if it's too small after the random resize
                if image.shape[1] < 768 or image.shape[2] < 768:
                    height, width = image.shape[1], image.shape[2]
                    pad_height = max(768 - height, 0)
                    pad_width = max(768 - width, 0)
                    pad_height_half = pad_height // 2
                    pad_width_half = pad_width // 2

                    border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                    image = F.pad(image, border, 'constant', 0)
                    label = F.pad(label, border, 'constant', self.ignore_index)

            # Random Horizontal Flip
            if config.USE_FLIPPING:
                if random.random() < 0.5:
                    image = TF.hflip(image)
                    label = TF.hflip(label)

            # Random Crop
            if config.USE_CROPPING:
                i, j, h, w = transforms.RandomCrop(size=(768, 768)).get_params(image, output_size=(768, 768))
                image = TF.crop(image, i, j, h, w)
                label = TF.crop(label, i, j, h, w) 

        return image, label

    def _load_image_and_label_paths(self):
        if self.split == 'train':
            image_dir = os.path.join(self.root_dir, 'leftImg8bit/train')
            label_dir = os.path.join(self.root_dir, 'gtFine/train')
        elif self.split == 'val':
            image_dir = os.path.join(self.root_dir, 'leftImg8bit/val')
            label_dir = os.path.join(self.root_dir, 'gtFine/val')
        else:
            raise ValueError(f'Invalid split: {self.split}')

        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_label_dir = os.path.join(label_dir, city)

            for image in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, image)
                label_path = os.path.join(city_label_dir, image.replace('leftImg8bit', 'gtFine_labelIds'))
                self.images.append(image_path)
                self.labels.append(label_path)


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


def decode_segmentation_map(segmentation_map):
    """
    It takes a segmentation map, which is a 2D array of integers, and returns a 3D array of RGB values
    Example Usage: 
        image, label = dataset[0]
        output = model(image)
        decoded_output = decode_segmentation_map(output)
        decoded_label = decode_segmentation_map(label)
    """
    segmentation_map = segmentation_map.numpy()
    red_channel = segmentation_map.copy()
    green_channel = segmentation_map.copy()
    blue_channel = segmentation_map.copy()

    colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0],
        [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
    ]
    label_colors = dict(zip(range(20), colors))

    class_indices = [i for i in range(19)]

    for i in class_indices:
        red_channel[segmentation_map == i] = label_colors[i][0]
        green_channel[segmentation_map == i] = label_colors[i][1]
        blue_channel[segmentation_map == i] = label_colors[i][2]
    
    if np.max(segmentation_map) == 255:
        red_channel[segmentation_map == 255] = label_colors[19][0]
        green_channel[segmentation_map == 255] = label_colors[19][1]
        blue_channel[segmentation_map == 255] = label_colors[19][2]

    rgb = np.zeros((segmentation_map.shape[1], segmentation_map.shape[2], 3))
    rgb[:, :, 0] = red_channel / 255.0
    rgb[:, :, 1] = green_channel / 255.0
    rgb[:, :, 2] = blue_channel / 255.0

    return rgb