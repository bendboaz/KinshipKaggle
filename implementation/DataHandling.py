import os
from collections import defaultdict
from random import seed
from itertools import product, chain
import pickle

import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from implementation.Utils import train_dev_split, PROJECT_ROOT

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
                dataset[family][person_name].append([family, person, img])

    return dataset


def read_test_dataset(path):
    return [[name] for name in os.listdir(path)]


class KinshipDataset(Dataset):
    @staticmethod
    def get_pair_label(pair, connections):
        family1, person1, _ = pair[0]
        family2, person2, _ = pair[1]
        if person1 == person2:
            return 0
        return 1 if f"{family2}/{person2}" in connections[f"{family1}/{person1}"] else 0

    def __init__(self, path, labels_path, data_augmentation=True, is_test=False, sample_path=None):
        super(Dataset, self).__init__()
        self.path = path
        self.data_augmentation = data_augmentation
        self.is_test = is_test
        self.allpairs = []

        self.family_pairs_size = -1

        if self.is_test:
            if sample_path is None:
                raise ValueError('Creating test dataset. Expected a path to sample_submission.csv, got None.')
            sample_pairs = pd.read_csv(sample_path)['img_pair'].tolist()

            for pair in sample_pairs:
                img1, img2 = pair.split('-')
                assert (os.path.isfile(os.path.join(self.path, img1))
                        and os.path.isfile(os.path.join(self.path, img2)))
                self.allpairs.append((([img1], [img2]), pair))
        else:
            self.families = read_train_dataset(path)
            connections = self.get_connections(labels_path)

            for family, f_members in tqdm(self.families.items(), desc="families", total=len(self.families)):
                for (per1_name, per1_imgs), (per2_name, per2_imgs) in filter(lambda x: x[0][0] != x[1][0],
                                                                             product(f_members.items(), repeat=2)):
                    self.allpairs.extend([(pair, self.get_pair_label(pair, connections))
                                          for pair in product(per1_imgs, per2_imgs)])
            self.family_pairs_size = len(self.allpairs)

            self.add_strangers(connections)

    @staticmethod
    def get_connections(labels_path):
        labels = pd.read_csv(labels_path)
        connections = defaultdict(list)
        for _, (per1, per2) in labels.iterrows():
            connections[per1].append(per2)
            connections[per2].append(per1)
        return connections

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

        img1 = face_transforms(Image.open(os.path.join(self.path, *path1)))
        img2 = face_transforms(Image.open(os.path.join(self.path, *path2)))

        return torch.stack([img1, img2]), label

    def __len__(self):
        return len(self.allpairs)

    def add_strangers(self, connections):
        persons_in_dataset = []
        for family, f_members in self.families.items():
            persons_in_dataset.extend(f_members.values())
        np.random.shuffle(persons_in_dataset)
        stranger_pairs = []
        middle = int(len(persons_in_dataset) / 2)
        half_a = persons_in_dataset[:middle].copy()
        half_b = persons_in_dataset[middle:].copy()

        for index in range(min(len(half_b), len(half_a))):
            per1 = half_a.pop()
            per2 = half_b.pop()
            if per1[0][0] == per2[0][0]:
                continue
            if len(stranger_pairs) > len(self.allpairs) / 2:
                break
            stranger_pairs.extend([(pair, self.get_pair_label(pair, connections))
                                   for pair in product(per1, per2)])

        self.allpairs.extend(stranger_pairs)

    def remove_strangers(self):
        self.allpairs = self.allpairs[:self.family_pairs_size]

    def to_relative_paths(self, force_change=False):
        sample_item = self.allpairs[0][0][0]
        if (not isinstance(sample_item, str)) and not force_change:
            assert isinstance(sample_item, list)
            return False
        if force_change or self.allpairs[0][0][0].find(self.path) == 0:
            relative_start = len(self.path) + len(os.sep)
            self.allpairs = map(lambda labeled_pair: ((labeled_pair[0][0][relative_start:],
                                                       labeled_pair[0][1][relative_start:]),
                                                      labeled_pair[1]),self.allpairs)

        self.allpairs = list(map(lambda labeled_pair: ((labeled_pair[0][0].split(os.sep),
                                                        labeled_pair[0][1].split(os.sep)),
                                                       labeled_pair[1]),
                                 self.allpairs))
        return True

    def change_dir(self, new_path):
        self.path = new_path

    @classmethod
    def get_dataset(cls, pickled_path, raw_path=None, labels_path=None, data_augmentation=True,
                    force_change=False):
        if os.path.isfile(pickled_path):
            with open(pickled_path, 'rb') as f:
                dataset = pickle.load(f)

            dataset_changed = dataset.to_relative_paths(force_change)
            if raw_path != dataset.path:
                dataset_changed = dataset.change_dir(raw_path) or dataset_changed
            if dataset_changed:
                with open(pickled_path, 'wb+') as f:
                    pickle.dump(dataset, f)
        else:
            dataset = cls(raw_path, labels_path)
            with open(pickled_path, 'wb+') as f:
                pickle.dump(dataset, f)

        dataset.data_augmentation = data_augmentation
        return dataset

    @classmethod
    def get_test_dataset(cls, pickled_path, raw_path=None, sample_path=None):
        to_save = False
        if os.path.isfile(pickled_path):
            with open(pickled_path, 'rb') as f:
                dataset = pickle.load(f)
            if not dataset.is_test:
                raise Warning(f'Path "{pickled_path}" sent to get_test_dataset, '
                              f'but the saved dataset is not a test set!')
            if raw_path != dataset.path:
                dataset.change_dir(raw_path)
                to_save = True
        else:
            to_save = True
            dataset = cls(raw_path, None, data_augmentation=False,
                          is_test=True, sample_path=sample_path)
        if to_save:
            with open(pickled_path, 'wb+') as f:
                pickle.dump(dataset, f)

        return dataset


class KinshipTripletDataset(Dataset):
    def __init__(self, families_folder, labels_path=None,
                 data_augmentation=True):
        """
        :param families_folder: Path to the folder containing
            family directories.
        :param labels_path: Path to .csv file containing the known
            relation pairs.
        :param data_augmentation: Boolean flag specifying whether or
            not to use data augmentations.
        """
        super(KinshipTripletDataset, self).__init__()
        self.data_augmentation = data_augmentation
        self.families_folder = families_folder
        self.labels_path = labels_path

        self.families = read_train_dataset(families_folder)
        self.related = self.get_connections(labels_path)
        # self.all_people is a list containing all of the people we know
        # (as dictionary: fullname->list of images).
        self.all_people = {}
        for _, members in self.families.items():
            self.all_people.update(**members)

        self.strangers = {person: [other for other in self.all_people
                                   if other not in self.related[person]]
                          for person in self.all_people}
        self.triplets = []
        for person in tqdm(self.all_people, total=len(self.all_people), leave=False):
            for positive, negative in tqdm(
                                        product(
                                              self.related[person],
                                              self.strangers[person]
                                             ),
                                        total=(len(self.related[person])
                                               * len(self.strangers[person])),
                                        leave=True):
                # The triplet format is (anchor, positive, negative).
                self.triplets.extend(filter(
                    lambda x: all(map(lambda p: p in self.all_people, x)),
                    product(
                        self.all_people[person],
                        self.all_people[positive],
                        self.all_people[negative]
                    )
                ))

    def __getitem__(self, item):
        anchor, positive, negative = self.triplets[item]
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

        anchor_img = face_transforms(Image.open(
                             os.path.join(self.families_folder, *anchor)
                            ))
        positive_img = face_transforms(Image.open(
                             os.path.join(self.families_folder, *positive)
                            ))
        negative_img = face_transforms(Image.open(
            os.path.join(self.families_folder, *negative)
        ))

        return torch.stack([anchor_img, positive_img, negative_img])

    def __len__(self):
        return len(self.triplets)

    @staticmethod
    def get_connections(labels_path):
        labels = pd.read_csv(labels_path)
        connections = defaultdict(list)
        for _, (per1, per2) in labels.iterrows():
            connections[per1].append(per2)
            connections[per2].append(per1)
        return connections

    @classmethod
    def get_dataset(cls, pickled_path, families_path,
                    labels_path, data_augmentation=True):
        if os.path.isfile(pickled_path):
            with open(pickled_path, 'rb') as pickled_ds:
                dataset = pickle.load(pickled_ds)
            if dataset.families_path != families_path:
                dataset.families_path = families_path
        else:
            dataset = cls(families_path, labels_path=labels_path)
            with open(pickled_path, 'wb+') as pickle_file:
                pickle.dump(dataset, pickle_file)

        dataset.data_augmentation = data_augmentation
        return dataset


if __name__ == "__main__":
    RANDOM_SEED = 32
    seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data_directory = os.path.join(PROJECT_ROOT, "data")
    raw_train_dataset = os.path.join(PROJECT_ROOT, "data", "raw", "train")
    processed_directory = os.path.join(PROJECT_ROOT, "data", "processed")
    csv_file = os.path.join(PROJECT_ROOT, "data", "raw", "train_relationships.csv")

    my_dataset = KinshipTripletDataset.get_dataset(os.path.join(data_directory, "triplet_train.pkl"),
                                            os.path.join(processed_directory, "train"), csv_file)
    print("Dataset length:", len(my_dataset))
    # for idx in range(len(my_dataset))[:10]:
    #     pair, label = my_dataset[idx]
    #     if label == 0:
    #         print(pair)
    #         to_pil = transforms.ToPILImage()
    #         to_pil(pair[0]).show()
    #         to_pil(pair[1]).show()

