from PIL import Image
import os
import sys
import numpy as np
import torch
from tiny_imagenet import _Base

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cifar_jigsaw import ALL_PERMS, make_jigsaw  # make_jigsaw is size-agnostic


class TinyImageNetJigsaw(_Base):
    """
    Returns (4, C, H, W) views + target + perm_indices.
    index 0 = identity permutation, 1-3 = random jigsaw shuffles.
    Used by student_jigsaw.py.
    """
    def __init__(self, root, train=True, transform=None, n_views=4):
        super().__init__(root, train, transform)
        self.n_views = n_views

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        target = self.targets[index]
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_tensor = self.transform(img)

        views = [img_tensor]
        perm_indices = [0]
        chosen = np.random.choice(range(1, len(ALL_PERMS)), size=self.n_views - 1, replace=False)
        for p in chosen:
            views.append(make_jigsaw(img_tensor, p))
            perm_indices.append(p)

        return torch.stack(views), target, torch.tensor(perm_indices, dtype=torch.long)
