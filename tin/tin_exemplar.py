from PIL import Image
import os
import numpy as np
import torch
from tiny_imagenet import _Base


class TinyImageNetExemplar(_Base):
    """
    Returns (4, C, H, W) views + target + dataset_index.
    index 0 = normal image, 1-3 = heavily augmented versions.
    Used by student_examplar.py.
    """
    def __init__(self, root, train=True, transform=None, transform_aug=None):
        super().__init__(root, train, transform)
        self.transform_aug = transform_aug if transform_aug is not None else transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        target = self.targets[index]
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        imgs = torch.stack([
            self.transform(img),
            self.transform_aug(img),
            self.transform_aug(img),
            self.transform_aug(img),
        ])
        return imgs, target, index
