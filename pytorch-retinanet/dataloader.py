'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import pdb
import traceback
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
from encoder import DataEncoder
from sklearn.preprocessing import LabelEncoder
from transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, root, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          label_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.class_le = LabelEncoder()
        self.encoder = DataEncoder()
        npy_path = 'train_npy' if train else 'val_npy'
        self.fnames = np.load(os.path.join(root, npy_path, 'fnames.npy'))
        self.boxes = np.load(os.path.join(root, npy_path, 'boxes.npy'))
        self.labels = np.load(os.path.join(root, npy_path, 'labels.npy'))
        self.num_samples = len(self.fnames)
        # pdb.set_trace()

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        image_dir = 'train' if self.train else 'val'
        img = Image.open(os.path.join(self.root, image_dir, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = torch.Tensor(self.boxes[idx]).clone()
        labels = torch.LongTensor(self.labels[idx])
        size = self.input_size
        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            # img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size, size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def provider(mode='train'):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    root = '../data/'
    dataset = ListDataset(
        root=root,
        train=mode=='train',
        transform=transform,
        input_size=600
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=6,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    return dataloader


if __name__ == '__main__':
    # import torchvision
    dataloader = provider()
    print(dataloader.__len__())
    for images, loc_targets, cls_targets in dataloader:
        print(images.size(), loc_targets.size(), np.unique(cls_targets, return_counts=True))
        # grid = torchvision.utils.make_grid(images, 1)
        # torchvision.utils.save_image(grid, 'a.jpg')
        # break
