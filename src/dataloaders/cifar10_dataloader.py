import os
import numpy as np
from PIL import Image, ImageFilter
import random

import torch
import lightning as pl

from torch.utils.data import (
    DataLoader,
    random_split,
    Dataset,
    WeightedRandomSampler,
    Subset,
)

from PIL import ImageFilter, ImageOps

from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10


class CustomDataset(Dataset):
    """Taken from : https://github.com/fabio-deep/Variational-Capsule-Routing/blob/master/src/utils.py
    Creates a custom pytorch dataset, mainly
    used for creating validation set splits."""

    def __init__(self, data, labels, transform=None):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        return image, self.labels[idx]


class Standardize(object):
    """Standardizes a 'PIL Image' such that each channel
    gets zero mean and unit variance."""

    def __call__(self, img):
        return (img - img.mean(dim=(1, 2), keepdim=True)) / torch.clamp(
            img.std(dim=(1, 2), keepdim=True), min=1e-8
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


def class_random_split(data, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.
    args:
        data (Array / List): Array of data values or list of paths to data.
        labels (Array, int): Array of each data samples semantic label.
        n_classes (int): Number of Classes.
        n_samples_per_class (int): Quantity of data samples to be placed
                                    per class into the validation set.
    return:
        train / valid (dict): New Train and Valid splits of the dataset.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if n_samples_per_class[0] == 0:
        if isinstance(data, torch.Tensor):
            # torch.stack transforms list of tensors to tensor
            return {"train": torch.stack(train_x), "valid": torch.stack(valid_x)}
        # transforms list of np arrays to tensor
        return (
            {"train": torch.from_numpy(np.stack(train_x))},
            {"train": torch.from_numpy(np.stack(train_y))},
        )

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return (
            {"train": torch.stack(train_x), "valid": torch.stack(valid_x)},
            {"train": torch.stack(train_y), "valid": torch.stack(valid_y)},
        )
    # transforms list of np arrays to tensor
    return (
        {
            "train": torch.from_numpy(np.stack(train_x)),
            "valid": torch.from_numpy(np.stack(valid_x)),
        },
        {
            "train": torch.from_numpy(np.stack(train_y)),
            "valid": torch.from_numpy(np.stack(valid_y)),
        },
    )


def sample_weights(labels):
    """Calculates per sample weights."""
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1.0 / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


# dataset specific classes below here
class CIFAR10_DataModule(pl.LightningDataModule):
    name = "cifar10"
    extra_args = {}

    def __init__(
        self,
        data_dir: str = os.environ.get("DATA_DIR", None),
        val_split: int = 5000,
        num_workers: int = 16,
        batch_size: int = 32,
        num_classes: int = 10,
        imgchannels: int = 3,
        imghw: int = 32,
    ):
        super().__init__()

        self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        self.DATASET(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            **self.extra_args
        )
        self.DATASET(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
            **self.extra_args
        )

    def train_dataloader(self):
        transf = self.default_transforms()

        dataset = self.DATASET(
            self.data_dir,
            train=True,
            download=True,
            transform=transf,
            **self.extra_args
        )

        train_length = len(dataset)

        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator(),
        )

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        transf = self.default_transforms()

        dataset = self.DATASET(
            self.data_dir,
            train=True,
            download=True,
            transform=transf,
            **self.extra_args
        )

        train_length = len(dataset)

        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator(),
        )

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return loader

    def test_dataloader(self):
        transf = self.default_transforms()

        dataset = self.DATASET(
            self.data_dir,
            train=False,
            download=True,
            transform=transf,
            **self.extra_args
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                #                      std=[0.24703223, 0.24348513, 0.26158784])
            ]
        )
        return cf10_transforms


class CIFARTrainDataTransform(object):
    def __init__(self, args):
        color_jitter = transforms.ColorJitter(
            0.8 * args.jitter_d,
            0.8 * args.jitter_d,
            0.4 * args.jitter_d,
            0.2 * args.jitter_d,
        )
        # data_transforms1 = transforms.Compose([  # transforms.ToPILImage(),
        #     transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),#, interpolation=Image.BICUBIC),
        #     transforms.RandomApply([color_jitter], p=args.jitter_p),
        #     transforms.RandomGrayscale(p=args.grey_p),
        #     # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
        #                          std=[0.24703223, 0.24348513, 0.26158784])
        # ])
        data_transforms1 = transforms.Compose(
            [  # transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    (32, 32), scale=(0.75, 1.0), interpolation=Image.BICUBIC
                ),
                # transforms.RandomApply([color_jitter], p=args.jitter_p),
                # transforms.RandomGrayscale(p=args.grey_p),
                # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        data_transforms2 = transforms.Compose(
            [  # transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    (32, 32), scale=(0.75, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomApply([color_jitter], p=args.jitter_p),
                transforms.RandomGrayscale(p=args.grey_p),
                # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
                transforms.RandomHorizontalFlip(),
                Solarization(0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        self.train_transform1 = data_transforms1
        self.train_transform2 = data_transforms2

        # transformation for the local small crops
        self.local_crops_number = 2
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (18, 18), scale=(0.25, 0.5), interpolation=Image.BICUBIC
                ),
                transforms.RandomApply([color_jitter], p=args.jitter_p),
                transforms.RandomGrayscale(p=args.grey_p),
                # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(7),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

    def __call__(self, sample):
        # crops = []
        # crops.append(self.train_transform1(sample))
        # crops.append(self.train_transform2(sample))
        # for _ in range(self.local_crops_number):
        #     crops.append(self.local_transform(sample))
        # return crops

        transform1 = self.train_transform1
        transform2 = self.train_transform2

        xi = transform1(sample)
        xj = transform2(sample)
        return xi, xj


class CIFAREvalDataTransform(object):
    def __init__(self, args):
        test_transform = transforms.Compose(
            [  # transforms.ToPILImage(),
                transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class CIFARTrainLinTransform(object):
    def __init__(self, args):
        data_transforms = transforms.Compose(
            [  # transforms.ToPILImage(),
                transforms.RandomResizedCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class CIFAREvalLinTransform(object):
    def __init__(self, args):
        test_transform = transforms.Compose(
            [  # transforms.ToPILImage(),
                transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class CIFARTestLinTransform(object):
    def __init__(self, args):
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x
