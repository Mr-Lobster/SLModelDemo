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

    modules = [module for module in model.modules()]
    print("=" * 10)
    print("网络结构如下所示：")
    for i, layer in enumerate(modules):
        print(i, layer)

    print("=" * 10)
    print("其中具有权重参数的层为：")
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            print(i, layer)
            # print(layer.weight)
            # print(layer.weight.data)
