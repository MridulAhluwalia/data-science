#!/usr/bin/env python3
import torchvision.datasets as datasets

# ----------------------------------------
mnist_trainset = datasets.MNIST(
    root="../data/train", train=True, download=True, transform=None
)

mnist_testset = datasets.MNIST(
    root="../data/test", train=False, download=True, transform=None
)
