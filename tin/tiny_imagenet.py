from PIL import Image
import os
import zipfile
import urllib.request
import numpy as np
import torch
import torch.utils.data as data


TIN_MEAN = [0.4802, 0.4481, 0.3975]
TIN_STD  = [0.2770, 0.2691, 0.2821]

_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def _download(root):
    """Download and extract Tiny ImageNet if not already present."""
    if os.path.isfile(os.path.join(root, 'wnids.txt')):
        return
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    if not os.path.isfile(zip_path):
        print('Downloading Tiny ImageNet (~240 MB)...')
        urllib.request.urlretrieve(_URL, zip_path,
            reporthook=lambda b, bs, total: print(
                f'\r  {min(b*bs, total)/1e6:.1f}/{total/1e6:.1f} MB', end='', flush=True))
        print()
    print('Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(os.path.dirname(root))
    print('Done.')


class _Base(data.Dataset):
    """
    Tiny ImageNet standard layout:
        {root}/train/{wnid}/images/*.JPEG
        {root}/val/images/*.JPEG  +  val_annotations.txt
        {root}/wnids.txt
    """
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        _download(self.root)
        with open(os.path.join(self.root, 'wnids.txt')) as f:
            wnids = sorted(l.strip() for l in f if l.strip())
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        if train:
            self._load_train()
        else:
            self._load_val()

    def _load_train(self):
        self.samples, self.targets = [], []
        for wnid in sorted(os.listdir(os.path.join(self.root, 'train'))):
            img_dir = os.path.join(self.root, 'train', wnid, 'images')
            if not os.path.isdir(img_dir) or wnid not in self.class_to_idx:
                continue
            label = self.class_to_idx[wnid]
            for fname in sorted(os.listdir(img_dir)):
                self.samples.append(os.path.join(img_dir, fname))
                self.targets.append(label)

    def _load_val(self):
        self.samples, self.targets = [], []
        img_dir = os.path.join(self.root, 'val', 'images')
        with open(os.path.join(self.root, 'val', 'val_annotations.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                self.samples.append(os.path.join(img_dir, parts[0]))
                self.targets.append(self.class_to_idx[parts[1]])

    def __len__(self):
        return len(self.samples)


class TinyImageNet(_Base):
    """
    Returns (4, C, H, W) rotation views + target.
    index 0 = 0°, 1 = 90° CCW, 2 = 180°, 3 = 270° CCW.
    Used by student.py (contrastive) and student_rot.py.
    """
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        target = self.targets[index]
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        imgs = torch.stack([self.transform(img.rotate(k * 90)) for k in range(4)])
        return imgs, target


class TinyImageNetPlain(_Base):
    """
    Returns plain (img, target). Used by teacher.py.
    Horizontal flip is handled by the transform passed in.
    """
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        return self.transform(img), self.targets[index]
