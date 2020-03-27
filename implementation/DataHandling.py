import os
from collections import defaultdict
from random import seed
from itertools import product
import pickle

import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from implementation.Utils import train_dev_split

# Implement datasets (train, test, add validation split)
# Later: augmentations (horizontal flips, color tweaks etc.)


def read_image(path):
    img = Image.open(path)
    return img


def read_train_dataset(path):
    dataset = dict()
    for family in os.listdir(path):
        family_path = os.path.join(path, family)
        dataset[family] = defaultdict(list)
        for person in os.listdir(family_path):
            person_name = f"{family}/{person}"
            person_path = os.path.join(family_path, person)
            for img in os.listdir(person_path):
                dataset[family][person_name].append(os.path.join(family, person, img))

    return dataset


def read_test_dataset(path):
    return {name: os.path.join(path, name) for name in os.listdir(path)}


class KinshipDataset(Dataset):
    @staticmethod
    def get_pair_label(pair, connections):
        person1, _ = os.path.split(pair[0])
        family1, person1 = os.path.split(person1)
        person2, _ = os.path.split(pair[1])
        family2, person2 = os.path.split(person2)
        if person1 == person2:
            return 0
        return 1 if f"{family2}/{person2}" in connections[f"{family1}/{person1}"] else 0

    def __init__(self, path, labels_path, data_augmentation=True):
        super(Dataset, self).__init__()
        self.path = path
        self.families = read_train_dataset(path)
        self.data_augmentation = data_augmentation
        labels = pd.read_csv(labels_path)
        connections = defaultdict(list)
        for _, (per1, per2) in labels.iterrows():
            connections[per1].append(per2)
            connections[per2].append(per1)

        self.allpairs = []
        for family, f_members in tqdm(self.families.items(), desc="families", total=len(self.families)):
            for (per1_name, per1_imgs), (per2_name, per2_imgs) in filter(lambda x: x[0][0] != x[1][0],
                                                                         product(f_members.items(), repeat=2)):
                self.allpairs.extend([(pair, self.get_pair_label(pair, connections))
                                      for pair in product(per1_imgs, per2_imgs)])

    def __getitem__(self, item):
        pair, label = self.allpairs[item]
        path1, path2 = pair
        if self.data_augmentation:
            face_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation([-30, 30]),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3)
            ])
        else:
            face_transforms = transforms.ToTensor()

        img1 = face_transforms(Image.open(os.path.join(self.path, path1)))
        img2 = face_transforms(Image.open(os.path.join(self.path, path2)))

        return torch.stack([img1, img2]), label

    def __len__(self):
        return len(self.allpairs)

    def to_relative_paths(self):
        if self.allpairs[0][0].find(self.path) != 0:
            return
        relative_start = len(self.path)
        self.allpairs = list(map(lambda pair, label: ((pair[0][relative_start:], pair[1][relative_start:]), label),
                                 self.allpairs))

    def change_dir(self, new_path):
        self.path = new_path

    @classmethod
    def get_dataset(cls, pickled_path, raw_path=None, labels_path=None, data_augmentation=True):
        if os.path.isfile(pickled_path):
            with open(pickled_path, 'rb') as f:
                dataset = pickle.load(f)
            dataset.to_relative_paths()
            if raw_path != dataset.path:
                dataset.change_dir(raw_path)
        else:
            dataset = cls(raw_path, labels_path)
            with open(pickled_path, 'wb+') as f:
                pickle.dump(dataset, f)
        dataset.data_augmentation = data_augmentation
        return dataset


if __name__ == "__main__":
    RANDOM_SEED = 32
    seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    raw_train_dataset = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle\\data\\raw\\train\\"
    processed_directory = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle\\data\\processed\\"

    my_dataset = KinshipDataset.get_dataset("..\\data\\train_dataset.pkl",
                                            "..\\data\\processed\\train", "..\\data\\raw\\train_relationships.csv")
    print("Dataset length:", len(my_dataset))
    for idx in range(len(my_dataset))[:10]:
        pair, label = my_dataset[idx]
        if label == 0:
            print(pair)
            to_pil = transforms.ToPILImage()
            to_pil(pair[0]).show()
            to_pil(pair[1]).show()

