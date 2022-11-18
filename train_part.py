import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from skimage import io, transform
import random
import pickle

from data_loader import TrainDataset
from config import parse_args

import os

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
    if os.path.exists('./pickle/test_set.pickle'):
        print("Read the pickle file of test set...")
        with open('./pickle/test_set.pickle', 'rb') as f:
            test_set = pickle.load(f)
    else:
        print("Pickle file of test set do not exist...")
        print("test dataset path: {}".format(os.path.join(data_directory, 'test')))
        imgs_test, labels_test = get_data(imgs_dir=os.path.join(data_directory, 'test'), images_num=240, h=img_H,
                                          w=img_W)
        test_ind = random.sample(range(labels_test.shape[0]), int(np.floor(labels_test.shape[0])))
        test_set = TrainDataset(imgs_test[test_ind, ...], labels_test[test_ind, ...])
        with open('./pickle/test_set.pickle', 'wb') as f:
            pickle.dump(test_set, f)

    train_path = './pickle/node' + node_num + '_train_set.pickle'
    if os.path.exists(train_path):
        print("Read the pickle file of train set...")
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
    else:
        print("Pickle file of train set do not exist...")
        print("train dataset path: {}".format(os.path.join(data_directory, 'node' + node_num)))
        imgs_train, labels_train = get_data(imgs_dir=os.path.join(data_directory, 'node' + node_num), images_num=400,
                                            h=img_H, w=img_W)
        train_ind = random.sample(range(labels_train.shape[0]), int(np.floor(labels_train.shape[0])))
        train_set = TrainDataset(imgs_train[train_ind, ...], labels_train[train_ind, ...])
        with open(train_path, 'wb') as f:
            pickle.dump(train_set, f)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    print("train_size:", len(train_set))
    print("test_size:", len(test_set))
    return train_loader, test_loader


def train_and_test(train_loader, test_loader, num_classes, img_H, img_W):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden1_num, hidden2_num, output_num):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Sequential(nn.Linear(input_num, hidden1_num), nn.ReLU(True))
            self.fc2 = nn.Sequential(nn.Linear(hidden1_num, hidden2_num), nn.ReLU(True))
            self.fc3 = nn.Sequential(nn.Linear(hidden2_num, output_num))

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    # class NeuralNet(nn.Module):
    #     def __init__(self, output_num):
    #         super(NeuralNet, self).__init__()
    #         self.conv1 = nn.Conv2d(1,6,5)
    #         self.conv2 = nn.Conv2d(6,16,5)
    #         self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, output_num)

    #     def forward(self, x):
    #         x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2,2))
    #         x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2,2))

    #         x = x.view(x.size()[0], -1)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    epoches = 20
    lr = 0.001
    input_num = img_H * img_W * 3
    hidden1_num = 64
    hidden2_num = 16
    output_num = num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for i in range(10):
        print("The {} round begin...".format(i + 1))

        for epoch in range(epoches):
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
    end = time.time()
    print('[Info] Parameter upload take %s s' % (end - start))
    print(model.state_dict())
    torch.save(model.state_dict(), './parameter/NNnet_node' + node_num + '.pkl')


if __name__ == '__main__':
    args_par = parse_args()
    data_directory = args_par.DATASET_DIR
    batch_size = args_par.BATCH_SIZE
    num_classes = args_par.NUM_CLASSES
    img_H = args_par.IMAGE_HEIGHT
    img_W = args_par.IMAGE_WIDTH
    # node_num = args_par.NODE_NUM
    node_num = "3"

    print('Start loading...')
    train_loader_A, test_loader = loadData(data_directory, batch_size, img_H, img_W)
    print('Start training...')
    train_and_test(train_loader_A, test_loader, num_classes, img_H, img_W)
