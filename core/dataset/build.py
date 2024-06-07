import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST #, CIFAR10, CIFAR100, ImageFolder, ImageNet


def build_dataset(cfg, train=True):
    # create a function that return a dataset based on the cfg.DATASETS.TRAIN
    # the dataset could be MNIST, CIFAR10, CIFAR100, Imagenette, ImageNet, etc.

    if cfg.DATASETS.TRAIN == "mnist":
        dataset = MNIST(
            root="datasets",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASETS.TRAIN}")

    return dataset