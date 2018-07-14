"""Prepare annotation file for train and val set"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def main(root, cls_file, box_file, npy_path):
    print('Reading all annotations\n')
    class_le = LabelEncoder()
    class_data = pd.read_csv(cls_file, header=None)
    class_le.fit(class_data[1])
    class_dict = {
        class_data.iloc[i][0]: class_data.iloc[i][1]
        for i in range(len(class_data))
    }  # a dict mapping MID to class name
    box_file = pd.read_csv(box_file).values
    box, label, fnames, current_file = [], [], [], ''
    boxes, labels = [], []
    for idx, line in tqdm(enumerate(box_file)):
        xmin = line[4]
        xmax = line[5]
        ymin = line[6]
        ymax = line[7]
        c = class_dict[line[2]]
        box.append([xmin, ymin, xmax, ymax])
        label.append(class_le.transform([c])[0])
        filename = line[0] + '.jpg'
        if idx:  # don't append on first iteration
            if current_file != filename or idx == len(box_file):
                fnames.append(current_file)
                boxes.append(box)
                labels.append(label)
                box, label = [], []
        current_file = filename

    print('\npreparing downloaded_files defaultdict\n')
    downloaded_files = defaultdict(bool)
    for file in tqdm(os.listdir(root)):
        downloaded_files[file] = True

    print('\npreparing idxs_to_be_deleted\n')
    idxs_to_be_deleted = []
    for idx, fname in tqdm(enumerate(fnames)):
        if downloaded_files[fname] is False:
            idxs_to_be_deleted.append(idx)

    print('idx_to_be_deleted', len(idxs_to_be_deleted))
    print('\ndeleting the labels not in the downloaded dataset\n')
    for idx in tqdm(sorted(idxs_to_be_deleted, reverse=True)):
        del fnames[idx]
        del boxes[idx]
        del labels[idx]

    if not os.path.exists(npy_path):
        os.mkdir(npy_path)

    np.save(os.path.join(npy_path, 'fnames.npy'), np.asarray(fnames))
    np.save(os.path.join(npy_path, 'boxes.npy'), np.asarray(boxes))
    np.save(os.path.join(npy_path, 'labels.npy'), np.asarray(labels))


if __name__ == '__main__':
    root = 'train'
    cls_file = 'class-descriptions-boxable.csv'
    box_file = 'train-annotations-bbox.csv'
    npy_path = 'train_npy'
    main(root, cls_file, box_file, npy_path)
    root = 'val'
    box_file = 'validation-annotations-bbox.csv'
    npy_path = 'val_npy'
    # main(root, cls_file, box_file, npy_path)
