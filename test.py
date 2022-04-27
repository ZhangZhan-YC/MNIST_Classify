# import torch as t
# from torchvision import datasets
#
# data_train = datasets.MNIST(root="./data", train=True, download=True)
# data_test = datasets.MNIST(root="./data", train=False)

import torch
from torch import nn
#  查看gpu信息
cudaMsg = torch.cuda.is_available()
gpuCount = torch.cuda.device_count()
print("1.是否存在GPU:{}".format(cudaMsg), "如果存在有：{}个".format(gpuCount))