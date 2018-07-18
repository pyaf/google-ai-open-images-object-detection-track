import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
from retinanet import RetinaNet
from encoder import DataEncoder
from parameters import params
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


def provider(batch_size=params['batch_size']):

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
        input_size=params['input_size']
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
    net = RetinaNet()
    # net.load_state_dict(torch.load('./checkpoint/model.pth'))
    net.eval()
    encoder = DataEncoder()
    dataloader = provider()
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
        print(images.size())
        images = Variable(images)
        loc_preds, cls_preds = net(images)
        # import pdb; pdb.set_trace()
        for i in range(len(images)):
            boxes, labels = encoder.decode(
                loc_preds[i].data.squeeze(),
                cls_preds[i].data.squeeze(),
            )
            print(boxes.shape, labels.shape)
