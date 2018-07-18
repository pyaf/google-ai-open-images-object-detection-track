import os
import pdb
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pycocotools.coco import COCO

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


def load_classes(json_path):
    coco = COCO(json_path)

    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x['id'])

    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    # also load the reverse (label -> name)
    labels = {}
    for key, value in classes.items():
        labels[value] = key

    return labels


class ListDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, transform, input_size):
        self.root = root
        self.transform = transform
        self.input_size = input_size
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df.iloc[:, 0])
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname + '.jpg'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        img = img.resize((self.input_size, self.input_size))
        img = self.transform(img)
        return img, fname, (w, h)

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
    sample_submission_path = '../data/sample_submission.csv'
    dataset = ListDataset(
        root,
        sample_submission_path,
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
    coco_classes = load_classes('data/instances_train2017.json')
    common_classes = []
    for c in class_le.classes_:
        if c.lower() in coco_classes.values():
            common_classes.append(c)
    input_size = 256
    print('Total batches:', len(dataloader))
    submission = {
        'ImageId': [],
        'PredictionString': []
    }
    for batch in tqdm(dataloader):
        images, fname, (w, h) = batch
        images = Variable(images.cuda())
        scores, classification, transformed_anchors = net(images)
        idxs = np.where(scores > 0.5)
        submission['ImageId'].append(fname[0].split('.')[0])
        string = []
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :].cpu().detach().numpy()
            x1 = float((bbox[0] / input_size) * w)
            y1 = float((bbox[1] / input_size) * h)
            x2 = float((bbox[2] / input_size) * w)
            y2 = float((bbox[3] / input_size) * h)
            label_name = coco_classes[int(classification[idxs[0][j]])].title()
            if label_name in common_classes:
                text = list(map(str, [class_dict[label_name], scores[idxs[0][j]].item(), x1, y1, x2, y2]))
                string.extend(text)
        submission['PredictionString'].append(' '.join(string))
        # if len(string): break
    pd.DataFrame(submission).to_csv('test_submission.csv', index=None)
