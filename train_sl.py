import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
from skimage import io, transform
import random

from data_loader import TrainDataset
from config import parse_args

import os

#  老的不用了
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_data(imgs_dir, images_num, h, w):
    classes = os.listdir(imgs_dir)
    image_list = np.empty((images_num, 3, h, w))
    label_list = np.empty(images_num)
    i = 0
    for c in (classes):
        image_names = os.listdir(os.path.join(imgs_dir, c))
        label = int(c)
        for name in (image_names):
            image_t = io.imread(os.path.join(imgs_dir, c, name))
            image_t = transform.resize(image_t, (h, w))
            image_t = np.transpose(image_t, (2, 0, 1))
            image_list[i] = image_t
            label_list[i] = label
            print("File load [{}/{}]".format(i + 1, images_num))
            i += 1
    return image_list, label_list


def loadData(data_directory, batch_size, img_H, img_W):
    print("test dataset path: {}".format(os.path.join(data_directory, 'test')))
    imgs_test, labels_test = get_data(imgs_dir=os.path.join(data_directory, 'test'), images_num=240, h=img_H, w=img_W)
    test_ind = random.sample(range(labels_test.shape[0]), int(np.floor(labels_test.shape[0])))
    test_set = TrainDataset(imgs_test[test_ind, ...], labels_test[test_ind, ...])

    print("train dataset A path: {}".format(os.path.join(data_directory, 'node1')))
    imgs_train_A, labels_train_A = get_data(imgs_dir=os.path.join(data_directory, 'node1'), images_num=400, h=img_H,
                                            w=img_W)
    train_ind_A = random.sample(range(labels_train_A.shape[0]), int(np.floor(labels_train_A.shape[0])))
    train_set_A = TrainDataset(imgs_train_A[train_ind_A, ...], labels_train_A[train_ind_A, ...])

    print("train dataset B path: {}".format(os.path.join(data_directory, 'node2')))
    imgs_train_B, labels_train_B = get_data(imgs_dir=os.path.join(data_directory, 'node2'), images_num=400, h=img_H,
                                            w=img_W)
    train_ind_B = random.sample(range(labels_train_B.shape[0]), int(np.floor(labels_train_B.shape[0])))
    train_set_B = TrainDataset(imgs_train_B[train_ind_B, ...], labels_train_B[train_ind_B, ...])

    print("train dataset C path: {}".format(os.path.join(data_directory, 'node3')))
    imgs_train_C, labels_train_C = get_data(imgs_dir=os.path.join(data_directory, 'node3'), images_num=400, h=img_H,
                                            w=img_W)
    train_ind_C = random.sample(range(labels_train_C.shape[0]), int(np.floor(labels_train_C.shape[0])))
    train_set_C = TrainDataset(imgs_train_C[train_ind_C, ...], labels_train_C[train_ind_C, ...])

    train_loader_A = DataLoader(dataset=train_set_A, batch_size=batch_size, shuffle=True)
    train_loader_B = DataLoader(dataset=train_set_B, batch_size=batch_size, shuffle=True)
    train_loader_C = DataLoader(dataset=train_set_C, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    print("train_size_A:", len(train_set_A))
    print("train_size_B:", len(train_set_B))
    print("train_size_C:", len(train_set_C))
    print("test_size:", len(test_set))

    return train_loader_A, train_loader_B, train_loader_C, test_loader


def train_and_test_1(train_loader, test_loader, num_classes, img_H, img_W):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden1_num, hidden2_num, output_num):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Sequential(nn.Linear(input_num, hidden1_num), nn.BatchNorm1d(hidden1_num), nn.ReLU(True))
            self.fc2 = nn.Sequential(nn.Linear(hidden1_num, hidden2_num), nn.BatchNorm1d(hidden2_num), nn.ReLU(True))
            self.fc3 = nn.Sequential(nn.Linear(hidden2_num, output_num))

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    epoches = 20
    lr = 0.001
    input_num = img_H * img_W * 3
    hidden1_num = 1024
    hidden2_num = 128
    output_num = num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        flag = 0
        for i, data in enumerate(train_loader):
            images, labels = data['image'], data['label']
            images = images.reshape(-1, img_H * img_W * 3).to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))

    params = list(model.named_parameters())

    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        images, labels = data['image'], data['label']
        images = images.reshape(-1, img_H * img_W * 3).to(device)
        labels = labels.to(device)
        output = model(images)
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {:.4f}%".format(total, 100 * correct / total))
    return params


def train_and_test_2(test_loader, com_para_fc1, com_para_fc2, com_para_fc3, num_classes, img_H, img_W):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden1_num, hidden2_num, output_num, com_para_fc1, com_para_fc2, com_para_fc3):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_num, hidden1_num)
            self.fc2 = nn.Linear(hidden1_num, hidden2_num)
            self.fc3 = nn.Linear(hidden2_num, output_num)
            self.fc1.weight = Parameter(com_para_fc1)
            self.fc2.weight = Parameter(com_para_fc2)
            self.fc3.weight = Parameter(com_para_fc3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    epoches = 20
    lr = 0.001
    input_num = img_H * img_W * 3
    hidden1_num = 1024
    hidden2_num = 128
    output_num = num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num, com_para_fc1, com_para_fc2, com_para_fc3)
    model.to(device)
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        images, labels = data['image'], data['label']
        images = images.reshape(-1, img_H * img_W * 3).to(device)
        labels = labels.to(device)
        output = model(images)
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {:.4f}%".format(total, 100 * correct / total))
    return


def combine_params(para_A, para_B, para_C):
    fc1_wA = para_A[0][1].data
    fc1_wB = para_B[0][1].data
    fc1_wC = para_C[0][1].data

    fc2_wA = para_A[2][1].data
    fc2_wB = para_B[2][1].data
    fc2_wC = para_C[2][1].data

    fc3_wA = para_A[4][1].data
    fc3_wB = para_B[4][1].data
    fc3_wC = para_C[4][1].data

    com_para_fc1 = (fc1_wA + fc1_wB + fc1_wC) / 3
    com_para_fc2 = (fc2_wA + fc2_wB + fc2_wC) / 3
    com_para_fc3 = (fc3_wA + fc3_wB + fc3_wC) / 3
    return com_para_fc1, com_para_fc2, com_para_fc3


if __name__ == '__main__':
    args_par = parse_args()
    data_directory = args_par.DATASET_DIR
    batch_size = args_par.BATCH_SIZE
    num_classes = args_par.NUM_CLASSES
    img_H = args_par.IMAGE_HEIGHT
    img_W = args_par.IMAGE_WIDTH

    train_loader_A, train_loader_B, train_loader_C, test_loader = loadData(data_directory, batch_size, img_H, img_W)
    para_A = train_and_test_1(train_loader_A, test_loader, num_classes, img_H, img_W)
    para_B = train_and_test_1(train_loader_B, test_loader, num_classes, img_H, img_W)
    para_C = train_and_test_1(train_loader_C, test_loader, num_classes, img_H, img_W)
    for i in range(10):
        print("The {} round to be federated...".format(i + 1))
        com_para_fc1, com_para_fc2, com_para_fc3 = combine_params(para_A, para_B, para_C)
        train_and_test_2(test_loader, com_para_fc1, com_para_fc2, com_para_fc3, num_classes, img_H, img_W)
