import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MnistDataLoader
from os.path import join
import time

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

# 处理数据并将数据塞进GPU
data_train = torch.tensor(x_train).float()
label_train = torch.tensor(y_train)
x_train = data_train.cuda()
y_train = label_train.cuda()
print('x_train:', x_train.device)
print('y_train:', x_train.device)
data_test = torch.tensor(x_test).float()
label_test = torch.tensor(y_test)
x_test = data_test.cuda()
y_test = label_test.cuda()
print('x_test:', x_test.device)
print('y_test:', y_test.device)
# print(x_train.size(), y_train.size())
# print(y_train[0].type())
# print(x_train[0].type())
# print(x_train[0, 0].type())

# 训练
num_epochs = 10000
loss_old = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把模型塞进GPu
net.to(device)
time_start = time.time()
for t in range(num_epochs):
    prediction = net(x_train)
    # print(prediction.argmax(dim=-1))
    loss = loss_func(prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if torch.abs(loss_old - loss)< 1e-6:
        break
    # print(f'number of feature:{i + 1}  loss:{loss}')
    if (t+1) % 100 == 0:
        print(f'epoch:{t+1}  loss:{loss}')
    loss_old = loss
time_end = time.time()
print(f'time of training:{time_end - time_start}')

# TODO 绘制损失下降图像

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
