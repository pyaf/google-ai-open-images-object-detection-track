import os
import pdb
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class ListDataset(data.Dataset):
    def __init__(self, root, transform, input_size):
        self.root = root
        self.transform = transform
        self.input_size = input_size
        self.fnames = os.listdir(self.root)
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w = h = self.input_size
        img = img.resize((w, h))
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def provider(batch_size=8):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    root = '../data/test/'
    dataset = ListDataset(
        root=root,
        transform=transform,
        input_size=256
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return dataloader


if __name__ == '__main__':
    print('Loading model..')
    import model
    net = model.resnet50(num_classes=80)
    net.load_state_dict(
        torch.load('model/coco_resnet_50_map_0_335_state_dict.pt')
    )
    net.cuda().eval()
    dataloader = provider(batch_size=1)
    cls_file = '../data/class-descriptions-boxable.csv'
    class_data = pd.read_csv(cls_file, header=None)
    class_le = LabelEncoder()
    class_le.fit(class_data[1])
    class_dict = {
        class_data.iloc[i][1]: class_data.iloc[i][0]
        for i in range(len(class_data))
    }  # a dict mapping class name to MID
    print('Total batches:', len(dataloader))
    for images in dataloader:
        image = Variable(images.cuda())
        scores, classification, transformed_anchors = net(image)
        idxs = np.where(scores > 0.5)
        pdb.set_trace()
        break
        # img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        # img[img < 0] = 0
        # img[img > 255] = 255

        # img = np.transpose(img, (1, 2, 0))
