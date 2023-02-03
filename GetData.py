import torchvision as tv
from torch.utils import data


# 导入数据集，分成训练集和测试集
def get_train_data():
    dataset = tv.datasets.MNIST('/Users/qisenmao/Library/CloudStorage/OneDrive-个人/python/MNIST by Pytorch/data',
                                train=True, download=True,
                                transform=tv.transforms.Compose([
                                    tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(0.1307, 0.3081)
                                ]))
    train_dataloader = data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    return train_dataloader


def get_test_data():
    dataset = tv.datasets.MNIST('/Users/qisenmao/Library/CloudStorage/OneDrive-个人/python/MNIST by Pytorch/data',
                                train=False, download=True,
                                transform=tv.transforms.Compose([
                                    tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(0.1307, 0.3081)
                                ]))
    test_dataloader = data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    return test_dataloader
