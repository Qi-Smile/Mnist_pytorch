import torch
from torch import nn


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # 生成一个线性的映射，这是一个两个全连接层的神经网络
        # self.rnn = nn.RNN(input_size=28 * 28 * 1, hidden_size=28, num_layers=10, nonlinearity='relu')
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.size())
        # 上面的方法可以用来测试全连接层的输入参数
        x = x.view(-1, 320)  # 可以用来调整大小
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        return x
