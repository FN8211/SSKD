from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
from itertools import permutations

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


# ── pre-compute all 24 permutations of 4 patches ──
ALL_PERMS = list(permutations(range(4)))   # list of 24 tuples, index = class label


def make_jigsaw(img_tensor, perm_index):
    """
    img_tensor : (C, H, W) tensor, H=W=32
    perm_index : int in [0, 23]
    Returns a (C, H, W) tensor where the 4 patches are rearranged
    according to ALL_PERMS[perm_index].

    Patch layout (original):
        0 | 1
        -----
        2 | 3
    """
    perm = ALL_PERMS[perm_index]
    c, h, w = img_tensor.shape
    ph, pw = h // 2, w // 2      # patch height / width = 16

    # extract 4 patches in reading order
    patches = [
        img_tensor[:, :ph, :pw],    # top-left
        img_tensor[:, :ph, pw:],    # top-right
        img_tensor[:, ph:, :pw],    # bottom-left
        img_tensor[:, ph:, pw:],    # bottom-right
    ]

    # reassemble according to permutation
    shuffled = [patches[perm[i]] for i in range(4)]
    top    = torch.cat([shuffled[0], shuffled[1]], dim=2)   # (C, ph, w)
    bottom = torch.cat([shuffled[2], shuffled[3]], dim=2)   # (C, ph, w)
    return torch.cat([top, bottom], dim=1)                  # (C, h, w)


class CIFAR10Jigsaw(data.Dataset):
    """
    CIFAR10 base with Jigsaw augmentation.
    __getitem__ returns:
        imgs   : (4, C, H, W) tensor — one normal image + 3 jigsaw versions
                 (index 0 is always the normal/identity permutation image)
        target : int class label
        perm_indices : (4,) LongTensor — permutation index for each view
                       index 0 is always 0 (identity permutation)
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

    def __init__(self, root, train=True, transform=None, n_views=4):
        """
        n_views : total number of views per image (1 normal + n_views-1 jigsaw).
                  Must be <= 24. Default 4 to match the original SSKD setup.
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.n_views = n_views

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

        # optional horizontal flip (train only), same as original cifar.py
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()

        img = Image.fromarray(img)
        img_tensor = self.transform(img)    # (C, H, W)

        # view 0: identity permutation (normal image)
        views = [img_tensor]
        perm_indices = [0]

        # views 1..n_views-1: random jigsaw permutations (never identity=0)
        chosen = np.random.choice(range(1, len(ALL_PERMS)),
                                  size=self.n_views - 1, replace=False)
        for p in chosen:
            views.append(make_jigsaw(img_tensor, p))
            perm_indices.append(p)

        imgs = torch.stack(views)                               # (n_views, C, H, W)
        perm_indices = torch.tensor(perm_indices, dtype=torch.long)
        return imgs, target, perm_indices

    def __len__(self):
        return len(self.data)


class CIFAR100Jigsaw(CIFAR10Jigsaw):
    base_folder = 'cifar-100-python'
    train_list = [['train', '']]
    test_list  = [['test',  '']]
    meta = {'filename': 'meta', 'key': 'fine_label_names'}