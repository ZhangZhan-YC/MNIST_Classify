import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MnistDataLoader
from os.path import join
import numpy as np

# TODO：优化正确率
# 搭建网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.output = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x

net = Net(784, 500, 300, 10)
# print(net)

# 优化与损失
# 使用梯度下降法
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
# 选择交叉熵损失
loss_func = torch.nn.CrossEntropyLoss()


# 读取数据集
input_path = 'D://DataBase/MNIST/'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataLoader.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
# print(np.array(x_train).shape)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test)
print(x_train.size(), y_train.size())
# print(y_train[0].type())
# print(x_train[0].type())
# print(x_train[0, 0].type())

# 训练
num_epochs = 500
for t in range(num_epochs):
    prediction = net(x_train)
    print(prediction.argmax(dim=-1))
    loss = loss_func(prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(f'number of feature:{i + 1}  loss:{loss}')
    print(f'epoch:{t+1}  loss:{loss}')

# 验证
prediction = net(x_test)
print(prediction)
result = prediction.argmax(dim=-1)
print(result)
sum = 0
n = y_test.size()[0]
for i in range(n):
    if result[i] == y_test[i]:
        sum += 1
print(f'correct_rate:{sum/n}')
