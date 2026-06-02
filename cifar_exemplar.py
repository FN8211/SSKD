from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle

import torch
import torch.utils.data as data


class CIFAR10Exemplar(data.Dataset):
    """
    CIFAR10 base with Exemplar augmentation.
    __getitem__ returns:
        imgs   : (4, C, H, W) tensor
                 index 0 = normal image
                 index 1-3 = heavily transformed versions (same transforms
                             as original cifar.py: color drop, rotation,
                             crop+resize, color jitter are applied via
                             the transform pipeline passed in)
        target : int class label
        index  : int dataset index (used as Exemplar class label)
    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', ''],
        ['data_batch_2', ''],
        ['data_batch_3', ''],
        ['data_batch_4', ''],
        ['data_batch_5', ''],
    ]
    test_list = [['test_batch', '']]
    meta = {'filename': 'batches.meta', 'key': 'label_names'}

    def __init__(self, root, train=True, transform=None, transform_aug=None):
        """
        transform     : applied to the normal image (index 0)
        transform_aug : heavy augmentation applied to the 3 transformed views
                        if None, falls back to transform for all views
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_aug = transform_aug if transform_aug is not None else transform
        self.train = train

        downloaded_list = self.train_list if train else self.test_list
        self.data, self.targets = [], []
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(
                    entry['labels'] if 'labels' in entry else entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))   # HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # optional horizontal flip (train only)
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()

        img_pil = Image.fromarray(img)

        # view 0: normal image
        img_nor = self.transform(img_pil)

        # views 1-3: heavily augmented versions
        # transform_aug should include color jitter, random crop, etc.
        # each call to transform_aug produces a different random augmentation
        img_aug1 = self.transform_aug(img_pil)
        img_aug2 = self.transform_aug(img_pil)
        img_aug3 = self.transform_aug(img_pil)

        imgs = torch.stack([img_nor, img_aug1, img_aug2, img_aug3])  # (4, C, H, W)
        return imgs, target, index

    def __len__(self):
        return len(self.data)


class CIFAR100Exemplar(CIFAR10Exemplar):
    base_folder = 'cifar-100-python'
    train_list = [['train', '']]
    test_list  = [['test',  '']]
    meta = {'filename': 'meta', 'key': 'fine_label_names'}