import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from .RandAugment.augmentations import RandAugment
from .RandAugment.augmentations import CutoutDefault

# Parameters for data
cifar10_mean = (
    0.4914,
    0.4822,
    0.4465,
)
cifar10_std = (
    0.2471,
    0.2435,
    0.2616,
)

# Augmentations.
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
transform_strong = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
transform_strong.transforms.insert(0, RandAugment(2, 10))
transform_strong.transforms.append(CutoutDefault(16))
transform_val = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)


class TransformRemixMatch:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)

        return out1, out2, out3


def get_cifar10(
    root,
    l_samples,
    u_samples,
    val_samples=500,
    transform_train=transform_train,
    transform_strong=transform_strong,
    transform_val=transform_val,
    download=True,
):
    base_dataset = CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_split(
        base_dataset.targets, l_samples, u_samples, val_samples
    )

    train_labeled_dataset = CIFAR10_labeled(
        root,
        train_labeled_idxs,
        train=True,
        transform=transform_train,
    )
    train_unlabeled_dataset = CIFAR10_unlabeled(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformRemixMatch(transform_train, transform_strong),
    )
    val_dataset = CIFAR10_labeled(
        root, indexs=val_idxs, train=True, transform=transform_val, download=True
    )
    test_dataset = CIFAR10_labeled(
        root, train=False, transform=transform_val, download=True
    )

    print(f"#Train: Labeled: {len(train_labeled_idxs)}, Unlabeled: {len(train_unlabeled_idxs)}")
    print(f"#Val: {len(val_idxs)}")
    print(f"#Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, n_val_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        val_idxs.extend(idxs[:n_val_per_class])
        train_labeled_idxs.extend(idxs[n_val_per_class : n_labeled_per_class[i]+n_val_per_class])
        train_unlabeled_idxs.extend(
            idxs[
                n_labeled_per_class[i]+n_val_per_class : 
                n_labeled_per_class[i]+n_val_per_class+n_unlabeled_per_class[i]
            ]
        )
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class CIFAR10_labeled(CIFAR10):
    def __init__(
        self,
        root,
        indexs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    ):
        super(CIFAR10_labeled, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.orig_indices = np.asarray(indexs)
        else:
            self.orig_indices = np.arange(len(self.targets))
        
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(index)


class CIFAR10_unlabeled(CIFAR10_labeled):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    ):
        super(CIFAR10_unlabeled, self).__init__(
            root,
            indexs,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
