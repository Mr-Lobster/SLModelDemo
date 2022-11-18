import torch
import torch.nn as nn
import numpy as np
from skimage import io, transform
import random
from torch.utils.data import DataLoader
import pickle
from data_loader import TrainDataset
from config import parse_args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def loadTestSet(data_directory, batch_size, img_H, img_W):
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
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    return test_loader


if __name__ == '__main__':

    # 解析命令行参数
    args_par = parse_args()
    data_directory = args_par.DATASET_DIR
    batch_size = args_par.BATCH_SIZE
    num_classes = args_par.NUM_CLASSES
    img_H = args_par.IMAGE_HEIGHT
    img_W = args_par.IMAGE_WIDTH
    node_num = args_par.NODE_NUM
    # node_num = "3"

    # 计算模型所需参数
    input_num = img_H * img_W * 3
    hidden1_num = 64
    hidden2_num = 16
    output_num = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num).to(device)
    state_dict = torch.load('./parameter/NNnet_total' + '.pkl')
    print(state_dict)
    model.load_state_dict(state_dict)

    # 使用测试集进行测试
    correct = 0
    total = 0
    test_loader = loadTestSet(data_directory, batch_size, img_H, img_W)
    for i, data in enumerate(test_loader):
        images, labels = data['image'], data['label']
        images = images.reshape(-1, img_H * img_W * 3).to(device)
        labels = labels.to(device)
        output = model(images)
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {:.4f}%".format(total, 100 * correct / total))
