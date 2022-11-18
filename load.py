import torch
import torch.nn as nn

import json

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

input_num = 24 * 36 * 3
hidden1_num = 64
hidden2_num = 16
output_num = 4
model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num)
print('[Info] Loading parameters of NNnet')
model.load_state_dict(torch.load('./parameter/NNnet_node3.pkl'))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
fc1_w = []
fc2_w = []
fc3_w = []
fc1_b = []
fc2_b = []
fc3_b = []
print('----------------------------------------------------------------------------------')
for item in model.state_dict()['fc1.0.weight']:
    tmp = []
    for num in item:
        tmp.append(int(float(num)*10000))
    fc1_w.append(tmp)
for item in model.state_dict()['fc1.0.bias']:
    fc1_b.append(int(float(item)*10000))
print('fc1.weight\n', model.state_dict()['fc1.0.weight'])
print('fc1.bias\n', model.state_dict()['fc1.0.bias'])
print('----------------------------------------------------------------------------------')
for item in model.state_dict()['fc2.0.weight']:
    tmp = []
    for num in item:
        tmp.append(int(float(num)*10000))
    fc2_w.append(tmp)
for item in model.state_dict()['fc2.0.bias']:
    fc2_b.append(int(float(item)*10000))
print('fc2.weight\n', model.state_dict()['fc2.0.weight'])
print('fc2.bias\n', model.state_dict()['fc2.0.bias'])
print('----------------------------------------------------------------------------------')
for item in model.state_dict()['fc3.0.weight']:
    tmp = []
    for num in item:
        tmp.append(int(float(num)*10000))
    fc3_w.append(tmp)
for item in model.state_dict()['fc3.0.bias']:
    fc3_b.append(int(float(item))*10000)
print('fc3.weight\n', model.state_dict()['fc3.0.weight'])
print('fc3.bias\n', model.state_dict()['fc3.0.bias'])

model = {"fc1" : {"weight" : fc1_w, "bias" : fc1_b}, "fc2" : {"weight" : fc2_w, "bias" : fc2_b}, "fc3" : {"weight" : fc3_w, "bias" : fc3_b}}
model = json.dumps(model)
with open('./json/node3.json', 'w') as f:
    f.write(model)
print('[Info] The model information has been written in json/node3.json')