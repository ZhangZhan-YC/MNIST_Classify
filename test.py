import torch as t
from torchvision import datasets

data_train = datasets.MNIST(root="./data", train=True, download=True)
data_test = datasets.MNIST(root="./data", train=False)

