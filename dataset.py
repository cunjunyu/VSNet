import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import get_files, get_stem, tf_to_dof, tf_to_quat
import transformations as tr


def generate_set_paths(img_paths, label_paths, aug_factor=None, limits=None, weights=[1/2, 1/2]):
    assert len(img_paths) == len(label_paths), 'Unequal number of image and label paths: {} vs {}'.format(
        len(img_paths), len(label_paths))

    set_paths = []
    for a in range(len(img_paths)):

        if aug_factor is not None:
            assert aug_factor <= 1.0, 'aug_factor must not be larger than 1'
            b_idx_list_len = int(len(img_paths) * aug_factor)
            b_idx_list = np.random.permutation(len(img_paths))
            b_idx_list = b_idx_list[:b_idx_list_len]  # only select part of b_idx_list
        else:
            b_idx_list = np.arange(len(img_paths))

        for b in b_idx_list:
            img_a_path = img_paths[a]
            img_b_path = img_paths[b]
            label_a_path = label_paths[a]
            label_b_path = label_paths[b]

            assert get_stem(img_a_path) == get_stem(label_a_path), \
                '{} and {} should have the same stem!'.format(img_a_path, label_a_path)
            assert get_stem(img_b_path) == get_stem(label_b_path), \
                '{} and {} should have the same stem!'.format(img_b_path, label_b_path)

            if limits is not None:
                tf_a = np.loadtxt(label_a_path, dtype=np.float32)
                tf_b = np.loadtxt(label_b_path, dtype=np.float32)

                tf_ba = np.matmul(np.linalg.inv(tf_b), tf_a)  # take a as reference
                label = np.array(tf_to_quat(tf_ba), dtype=np.float32)

                trans_deviation = np.sqrt(np.mean((label[:3]) ** 2))
                quat_deviation = np.sqrt(np.mean((label[3:] - np.array([1.0, 0.0, 0.0, 0.0])) ** 2))

                if weights[0] * trans_deviation + weights[1] * quat_deviation > limits:
                    continue  # skip this example

            set_paths.append((img_a_path, img_b_path, label_a_path, label_b_path))

    return set_paths


def split_sets(img_dir_list, label_dir_list,
               train_size_list=[600], dev_size_list=[200], test_size_list=[200],
               random_seed=0, img_ext='.png', label_ext='.txt',
               aug_factor=None, limits=None, weights=[1/2, 1/2]):

    assert isinstance(img_dir_list, list), 'img_dir_list must be a list!'
    assert isinstance(label_dir_list, list), 'label_dir_list must be a list!'
    assert isinstance(train_size_list, list), 'train_size_list must be a list!'
    assert isinstance(dev_size_list, list), 'dev_size_list must be a list!'
    assert isinstance(test_size_list, list), 'test_size_list must be a list!'

    assert len(img_dir_list) == len(label_dir_list) == len(train_size_list) == len(dev_size_list) == len(test_size_list), 'lists must have equal lengths!'

    ret_train_paths = []
    ret_dev_paths = []
    ret_test_paths = []
    for i in range(len(img_dir_list)):

        img_dir = img_dir_list[i]
        label_dir = label_dir_list[i]
        train_size = train_size_list[i]
        dev_size = dev_size_list[i]
        test_size = test_size_list[i]

        img_paths = sorted(get_files(img_dir, img_ext))
        label_paths = sorted(get_files(label_dir, label_ext))

        assert len(img_paths) == len(label_paths), 'Unequal number of images and labels found!'
        assert len(img_paths) >= train_size + dev_size + test_size, 'Not enough images and labels are found'

        np.random.seed(random_seed)
        perm_list = np.random.permutation(len(img_paths))
        img_paths = np.array(img_paths)
        label_paths = np.array(label_paths)

        train_img_paths = img_paths[perm_list[0:train_size]].tolist()
        train_label_paths = label_paths[perm_list[0:train_size]].tolist()
        dev_img_paths = img_paths[perm_list[train_size:train_size + dev_size]].tolist()
        dev_label_paths = label_paths[perm_list[train_size:train_size + dev_size]].tolist()
        test_img_paths = img_paths[perm_list[train_size + dev_size:train_size + dev_size + test_size]].tolist()
        test_label_paths = label_paths[perm_list[train_size + dev_size:train_size + dev_size + test_size]].tolist()

        train_paths = generate_set_paths(train_img_paths, train_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)
        print('{} samples for train from dir {}'.format(len(train_paths), i))
        ret_train_paths.extend(train_paths)

        dev_paths = generate_set_paths(dev_img_paths, dev_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)
        print('{} samples for dev from dir {}'.format(len(dev_paths), i))
        ret_dev_paths.extend(dev_paths)

        test_paths = generate_set_paths(test_img_paths, test_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)
        print('{} samples for test from dir {}'.format(len(test_paths), i))
        ret_test_paths.extend(test_paths)

    print('Total {} samples for train'.format(len(ret_train_paths)))
    print('Total {} samples for dev'.format(len(ret_dev_paths)))
    print('Total {} samples for test'.format(len(ret_test_paths)))

    return ret_train_paths, ret_dev_paths, ret_test_paths


class VSDataset(Dataset):

    def __init__(self, set_paths, img_size, return_path=False):
        self.set_paths = set_paths
        self.return_path = return_path

        if img_size == (640, 480):
            # print('No resizing needed.')
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(size=img_size), # TODO,bug: here should be (h,w)!
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.set_paths)

    def __getitem__(self, idx):
        img_a_path, img_b_path, label_a_path, label_b_path = self.set_paths[idx]

        img_a = Image.open(img_a_path)
        img_b = Image.open(img_b_path)
        tf_a = np.loadtxt(label_a_path, dtype=np.float32)
        tf_b = np.loadtxt(label_b_path, dtype=np.float32)

        tf_ab = np.matmul(np.linalg.inv(tf_b), tf_a)  # take a as reference
        label = np.array(tf_to_quat(tf_ab), dtype=np.float32)

        img_a = self.img_transform(img_a)
        img_b = self.img_transform(img_b)

        label = torch.from_numpy(label)
        if self.return_path:
            return img_a, img_b, label, img_a_path, img_b_path, label_a_path, label_b_path
        else:
            return img_a, img_b, label




