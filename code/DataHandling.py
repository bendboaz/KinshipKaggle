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

from code.Utils import train_dev_split

# Implement image reading function
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
                dataset[family][person_name].append(os.path.join(person_path, img))

    return dataset


def read_test_dataset(path):
    return {name: os.path.join(path, name) for name in os.listdir(path)}


class KinshipDataset(Dataset):
    @staticmethod
    def get_pair_label(pair, connections):
        person1, _ = os.path.split(pair[0])
        family1, _ = os.path.split(person1)
        person2, _ = os.path.split(pair[1])
        family2, _ = os.path.split(person2)
        if person1 == person2:
            return 0
        return 1 if person2 in connections[person1] else 0

    def __init__(self, path, labels_path):
        super(Dataset, self).__init__()
        self.path = path
        self.families = read_train_dataset(path)
        labels = pd.read_csv(labels_path)
        connections = defaultdict(list)
        for _, (per1, per2) in labels.iterrows():
            connections[per1].append(per2)
            connections[per2].append(per1)

        self.allpairs = []
        for family, f_members in tqdm(self.families.items(), desc="families", total=len(self.families)):
            for (per1_name, per1_imgs), (per2_name, per2_imgs) in product(f_members.items(), repeat=2):
                self.allpairs.extend([(pair, self.get_pair_label(pair, connections))
                                      for pair in product(per1_imgs, per2_imgs)])

        # self.img2person = {img: person for person, imglist in tqdm(self.per2imgs.items(), total=len(self.per2imgs), desc="dict") for img in imglist}
        # self.images = list(self.img2person.keys())

        # pairs = list(filter(lambda x: os.path.split(x[0])[0] != os.path.split(x[1])[0],
        #                     tqdm(product(self.images, repeat=2), desc="product", total=len(self.images)**2)))
        # self.items = [(pair, self.get_pair_label(pair, connections)) for pair in tqdm(pairs, total=len(pairs), desc="pairs")]

    def __getitem__(self, item):
        pair, label = self.allpairs[item]
        path1, path2 = pair
        img1 = transforms.ToTensor()(Image.open(path1))
        img2 = transforms.ToTensor()(Image.open(path2))
        return torch.stack([img1, img2]), label

    def __len__(self):
        return len(self.allpairs)

    @classmethod
    def get_dataset(cls, pickled_path, raw_path=None, labels_path=None):
        if os.path.isfile(pickled_path):
            with open(pickled_path, 'rb') as f:
                return pickle.load(f)
        dataset = cls(raw_path, labels_path)
        with open(pickled_path, 'wb+') as f:
            pickle.dump(dataset, f)
        return dataset


if __name__ == "__main__":
    RANDOM_SEED = 32
    seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    raw_train_dataset = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle\\data\\raw\\train\\"
    processed_directory = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle\\data\\processed\\"

    my_dataset = KinshipDataset.get_dataset("..\\data\\dataset.pkl",
                                            "..\\data\\processed\\train", "..\\data\\raw\\train_relationships.csv")
    (img1, img2), label = my_dataset[0]
    img1 = transforms.ToPILImage()(img1)
    img1.show()


