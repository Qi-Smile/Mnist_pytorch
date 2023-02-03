import torch.nn.functional
import torch.nn as nn
import Net
from Net import MnistNet
from GetData import *
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
import time as tm
import torch

writer = SummaryWriter('./runs')  # tensorboard进行可视化
mnist_net = MnistNet()
for layer in mnist_net.modules():  # 初始化参数
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
cri = torch.nn.CrossEntropyLoss()
step = 0
loss_value = torch.inf
train_loss_list = []
train_count_list = []
time = 1
device = torch.device('mps')
mnist_net.to(device)


def train(epoch):
    mnist_net.train(True)  # 启用BatchNormalization和 Dropout
    train_dataloader = get_train_data()
    print("Begin Training in {} round".format(epoch))
    global step, loss_value, time
    for idx, (dt, target) in enumerate(train_dataloader):
        dt = dt.to(device)
        target = target.to(device)
        writer.add_graph(mnist_net, dt)
        optimizer.zero_grad()
        output = mnist_net(dt)
        loss = cri(output, target)
        writer.add_scalar('train_loss', loss, time)
        loss.backward()
        optimizer.step()
        time += 1
        print('epoch:{},loss:{}'.format(idx, loss))
        if idx % 20 == 0:
            step = step+1
            train_loss_list.append([step, loss/64])
            if loss < loss_value:
                torch.save(mnist_net, 'model.pth')
                loss_value = loss


def test():
    model_path = 'model.pth'
    model = torch.load(model_path)
    model.eval()
    test_dataloader = get_test_data()
    cor = 0
    global time
    time = 1
    with torch.no_grad():
        for idx, (dt, target) in enumerate(test_dataloader):
            dt = dt.to(device)
            target = target.to(device)
            output = model(dt)
            cur_loss = cri(output, target)
            writer.add_scalar('test_loss', cur_loss, time)
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            writer.add_scalar('test_acc', cur_acc, time)
            time += 1
            print('batch id: {}, accuracy: {}'.format(idx+1, cur_acc))


if __name__ == '__main__':
    """
    data = torch.randn(1, 1, 28, 28)
    test_net = MnistNet()
    test_net(data)
    """
    t0 = tm.time()
    for i in range(10):
        train(i)
    test()
    writer.close()
    t1 = tm.time()
    print('time_cost:', t1-t0)
    #  plt.show()








